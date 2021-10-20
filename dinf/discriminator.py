from __future__ import annotations
import abc
import pathlib
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras


class Symmetric(keras.layers.Layer):
    """
    Network layer for a permutation invariant cnn. This layer collapses
    a dimension using the specified summary function.
    """

    def __init__(self, summary_function, axis, **kwargs):
        assert summary_function in ("sum", "mean", "min", "max")
        self.summary_function = summary_function
        self.axis = axis
        super().__init__(**kwargs)

    def call(self, x):
        if self.summary_function == "sum":
            f = keras.backend.sum
        elif self.summary_function == "mean":
            f = keras.backend.mean
        elif self.summary_function == "min":
            f = keras.backend.min
        elif self.summary_function == "max":
            f = keras.backend.max
        return f(x, axis=self.axis, keepdims=True)

    def get_config(self):
        # Record internal state, so we can load and save the model.
        config = super().get_config().copy()
        config["summary_function"] = self.summary_function
        config["axis"] = self.axis
        return config


def my_conv2d(**kwargs):
    conv_kwargs = dict(
        kernel_size=(1, 5),
        padding="same",
        strides=(1, 2),
        use_bias=False,
    )
    conv_kwargs.update(**kwargs)
    return keras.layers.Conv2D(**conv_kwargs)


def _build_cnn1(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    """
    Build a permutation invariant discriminator neural network.

    :param input_shape:
        The shape of the data that will be given to the network.
        This should be a 3-tuple of (n, m, c), where n is the number of
        hapotypes, m is the size of the "fixed dimension" after resizing
        along the sequence length, and c is the number of colour channels
        (which should be equal to 1).
    :return: The neural network.
    """
    assert len(input_shape) == 3
    assert input_shape[-1] == 1
    nn = keras.models.Sequential(name="discriminator")
    nn.add(keras.Input(input_shape))
    nn.add(keras.layers.BatchNormalization())
    nn.add(my_conv2d(filters=32))
    nn.add(keras.layers.ELU())
    nn.add(keras.layers.BatchNormalization())
    nn.add(my_conv2d(filters=64))
    nn.add(keras.layers.ELU())
    nn.add(keras.layers.BatchNormalization())
    nn.add(Symmetric("sum", axis=1))
    nn.add(my_conv2d(filters=64, strides=(1, 1)))
    nn.add(keras.layers.ELU())
    nn.add(keras.layers.BatchNormalization())
    nn.add(Symmetric("sum", axis=2))
    nn.add(keras.layers.Flatten())
    # nn.add(keras.layers.Dense(128, activation="elu"))
    nn.add(keras.layers.Dense(1, activation="sigmoid"))

    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    opt = keras.optimizers.Adam()
    nn.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    return nn


def _load_cnn1(filename: str) -> tf.keras.Model:
    """Load neural network from a file."""
    return keras.models.load_model(
        filename,
        custom_objects={
            "Symmetric": Symmetric,
        },
    )


class Discriminator:
    default_strategy = tf.distribute.MirroredStrategy

    def __init__(self, nn: tf.keras.Model):
        """
        Instantiate a discriminator. Not
        :param nn: The neural network.
        """
        self.nn = nn

    @classmethod
    def from_input_shape(
        cls, input_shape: tuple[int, int, int], strategy=None
    ) -> Discriminator:
        """
        Build a neural network with the given input shape.

        :param input_shape:
            The shape of the data that will be given to the network.
            This should be a 3-tuple of (n, m, c), where n is the number of
            hapotypes, m is the size of the "fixed dimension" after resizing
            along the sequence length, and c is the number of colour channels
            (which should be equal to 1).
        :param strategy:
            The tensorflow distribute strategy. If None, the MirroredStrategy
            will be used. See tensorflow documentation:
            https://www.tensorflow.org/tutorials/distribute/keras
        :return: The discriminator object
        """
        if strategy is None:
            strategy = cls.default_strategy()
        with strategy.scope():
            nn = _build_cnn1(input_shape)
        return cls(nn)

    @classmethod
    def from_file(cls, filename, strategy=None) -> Discriminator:
        """
        Load neural network from the given file.

        :param filename: The filename of the saved keras model.
        :param strategy:
            The tensorflow distribute strategy. If None, the MirroredStrategy
            will be used. See tensorflow documentation:
            https://www.tensorflow.org/tutorials/distribute/keras
        :return: The discriminator object
        """
        if strategy is None:
            strategy = cls.default_strategy()
        with strategy.scope():
            nn = _load_cnn1(filename)
        return cls(nn)

    def to_file(self, filename: str) -> None:
        """Save neural network to a file."""
        self.nn.save(filename)

    def fit(
        self,
        *,
        train_x,
        train_y,
        val_x,
        val_y,
        batch_size: int = 128,
        epochs: int = 1,
        tensorboard_log_dir=None,
    ):
        """
        Fit a neural network to labelled training data.

        :param train_x: Training data.
        :param train_y: Labels for training data.
        :param val_x: Validation data.
        :param val_y: Labels for validation data.
        :param batch_size: Size of minibatch for gradient update step.
        :param epochs: The number of full passes over the training data.
        :param tensorboard_log_dir:
            Directory for tensorboard logs. If None, no logs will be recorded.
        """
        assert len(train_y.shape) == len(val_y.shape) == 1
        assert len(train_x.shape) == len(val_x.shape) == 4
        assert train_x.shape[1:] == val_x.shape[1:]
        assert train_x.shape[1] > 1
        assert train_x.shape[2] > 1
        assert train_x.shape[3] == 1

        callbacks = []
        if tensorboard_log_dir is not None:
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = pathlib.Path(tensorboard_log_dir) / now
            cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(cb)

        train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(batch_size)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)

        self.nn.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
        )

    def predict(self, x, *, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions about data using a pre-fitted neural network.

        :param x: The data instances about which to make predictions.
        :param batch_size: Size of data batches for prediction.
        :return: A vector of predictions, one for each input instance.
        """
        if len(x.shape) != 4 or x[1:] != self.nn.input_shape[1:]:
            raise ValueError(
                f"Input data has shape {x.shape} but discriminator network "
                f"has shape {self.nn.input_shape}."
            )

        data = tf.data.Dataset.from_tensor_slices(x)
        data = data.batch(batch_size)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        data = data.with_options(options)

        return self.nn.predict(x)[:, 0]
