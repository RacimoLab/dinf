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


def build(input_shape: tuple[int]) -> tf.keras.Model:
    """
    Build a discriminator neural network.

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
    return nn


def fit(
    nn: tf.keras.Model,
    *,
    train_x,
    train_y,
    val_x,
    val_y,
    batch_size: int = 32,
    epochs: int = 1
):
    """
    Fit a neural network to labelled training data.

    :param nn: The neural network.
    :param train_x: Training data.
    :param train_y: Labels for training data.
    :param val_x: Validation data.
    :param val_y: Labels for validation data.
    :param batch_size: Size of minibatch for gradient update step.
    :param epochs: The number of full passes over the training data.
    """
    assert len(train_y.shape) == len(val_y.shape) == 1
    assert len(train_x.shape) == len(val_x.shape) == 4
    assert train_x.shape[1:] == val_x.shape[1:]
    assert train_x.shape[1] > 1
    assert train_x.shape[2] > 1
    assert train_x.shape[3] == 1
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    opt = keras.optimizers.Adam()
    nn.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    nn.fit(
        train_x,
        train_y,
        batch_size,
        epochs,
        validation_data=(val_x, val_y),  # shuffle=True
    )


def predict(nn: tf.keras.Model, x) -> np.ndarray:
    """
    Make predictions about data using a pre-fitted neural network.

    :param nn: The neural network.
    :param x: The data instances about which to make predictions.
    :return: A vector of predictions, one for each input instance.
    """
    assert len(x.shape) == 4
    assert x.shape[1] > 1
    assert x.shape[2] > 1
    assert x.shape[3] == 1
    return nn.predict(x)[:, 0]


def save(nn: tf.keras.Model, filename: str) -> None:
    """Save neural network to a file."""
    nn.save(filename)


def load(filename: str) -> tf.keras.Model:
    """Load neural network from a file."""
    return keras.models.load_model(
        filename,
        custom_objects={
            "Symmetric": Symmetric,
        },
    )
