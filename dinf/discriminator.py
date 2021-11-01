from __future__ import annotations
import pathlib
import functools
import itertools
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import flax.training.train_state
import optax

# A type for jax PyTrees.
# https://github.com/google/jax/issues/3340
PyTree = Any

# Because we use batch normalistion, the training state needs to also record
# batch_stats to maintain the running mean and variance.
class TrainState(flax.training.train_state.TrainState):
    batch_stats: Any


def batchify(dataset, batch_size):
    """Generate batch_size chunks of the dataset."""
    assert batch_size >= 1
    k0 = tuple(dataset.keys())[0]
    size = len(dataset[k0])
    i, j = 0, batch_size
    while i < size:
        batch = {k: v[i:j, ...] for k, v in dataset.items()}
        yield batch
        i = j
        j += batch_size


class Symmetric(nn.Module):
    """
    Layer that summarises over a given axis in a way that
    is invariant to perumtations of data along that axis.
    """

    axis: int

    @nn.compact
    def __call__(self, x):
        return jnp.sum(x, axis=self.axis, keepdims=True)


class CNN1(nn.Module):
    @nn.compact
    def __call__(self, x, *, train: bool):
        # flax uses channels-last (NHWC) convention
        conv = functools.partial(nn.Conv, kernel_size=(1, 5), use_bias=False)
        # https://flax.readthedocs.io/en/latest/howtos/state_params.html
        norm = functools.partial(nn.BatchNorm, use_running_average=not train)

        x = norm()(x)

        x = conv(features=32, strides=(1, 2))(x)
        x = nn.elu(x)
        x = norm()(x)

        x = conv(features=64, strides=(1, 2))(x)
        x = nn.elu(x)
        x = norm()(x)

        # collapse haplotypes
        x = Symmetric(axis=1)(x)

        x = conv(features=64)(x)
        x = nn.elu(x)
        x = norm()(x)

        # collapse genomic bins
        x = Symmetric(axis=2)(x)

        x = nn.Dense(features=1)(x)

        x = x.reshape((x.shape[0],))  # flatten

        # x = nn.sigmoid(x)

        return x


def binary_accuracy(*, logits, labels):
    p = jax.nn.sigmoid(logits)
    return jnp.mean(labels == (p > 0.5))


class Discriminator:
    def __init__(self, dnn: nn.Module, variables: PyTree, input_shape: tuple):
        """
        Instantiate a discriminator.

        Not intended to be used directly. Use from_file() or from_input_shape()
        class methods instead.

        :param dnn: The neural network. This has an apply() method.
        :param variables: A PyTree of the network parameters.
        """
        self.dnn = dnn
        self.variables = variables
        self.input_shape = input_shape

    @classmethod
    def from_file(cls, filename) -> Discriminator:
        """
        Load neural network from the given file.

        :param filename: The filename of the saved flax model.
        """
        raise RuntimeError("TODO")

    @classmethod
    def from_input_shape(
        cls, input_shape: tuple[int, int, int], rng: np.random.Generator
    ) -> Discriminator:
        """
        Build a neural network with the given input shape.

        :param input_shape:
            The shape of the data that will be given to the network.
            This should be a 3-tuple of (n, m, c), where n is the number of
            hapotypes, m is the size of the "fixed dimension" after resizing
            along the sequence length, and c is the number of colour channels
            (which should be equal to 1).
        """
        dnn = CNN1()
        key = jax.random.PRNGKey(rng.integers(2 ** 63))
        input_shape = (1,) + input_shape  # add leading batch dimension
        dummy_input = jnp.zeros(input_shape, dtype=np.int8)

        @jax.jit
        def init(*args):
            return dnn.init(*args, train=False)

        variables = init(key, dummy_input)
        return cls(dnn, variables, input_shape)

    def summary(self):
        x = jnp.zeros(self.input_shape, dtype=np.int8)
        _, state = self.dnn.apply(
            self.variables,
            x,
            train=False,
            capture_intermediates=True,
            mutable=["intermediates"],
        )
        print(jax.tree_map(lambda x: (x.shape, x.dtype), state["intermediates"]))
        print(jax.tree_map(lambda x: (x.shape, x.dtype), self.variables))

    def fit(
        self,
        rng: np.random.Generator,
        *,
        train_x,
        train_y,
        val_x,
        val_y,
        batch_size: int = 64,
        epochs: int = 1,
        tensorboard_log_dir=None,
    ):
        """
        Fit discriminator to training data.

        :param rng: Numpy random number generator.
        """

        @jax.jit
        def train_step(state, batch):
            """Train for a single step."""

            def loss_fn(params):
                logits, new_model_state = state.apply_fn(
                    dict(params=params, batch_stats=state.batch_stats),
                    batch["image"],
                    mutable=["batch_stats"],
                    train=True,
                )
                loss = jnp.mean(
                    optax.sigmoid_binary_cross_entropy(
                        logits=logits, labels=batch["label"]
                    )
                )
                return loss, (logits, new_model_state)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (logits, new_model_state)), grads = grad_fn(state.params)
            state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state["batch_stats"]
            )
            metrics = dict(
                loss=loss,
                accuracy=binary_accuracy(logits=logits, labels=batch["label"]),
            )
            return state, metrics

        @jax.jit
        def eval_step(state, batch):
            logits = state.apply_fn(
                dict(params=state.params, batch_stats=state.batch_stats),
                batch["image"],
                train=False,
            )
            loss = jnp.mean(
                optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch["label"])
            )
            accuracy = binary_accuracy(logits=logits, labels=batch["label"])
            metrics = dict(loss=loss, accuracy=accuracy)
            return metrics

        def train_epoch(state, train_ds, batch_size, epoch, key):
            """Train for a single epoch."""
            batch_metrics = []
            for i, batch in enumerate(batchify(train_ds, batch_size)):
                state, metrics = train_step(state, batch)
                batch_metrics.append(metrics)
                if i % 20 == 0:
                    line = [f"{k}: {jax.device_get(v):.4f}" for k, v in metrics.items()]
                    print(*line, sep=", ", end="\r")

            # compute mean of metrics across each batch in epoch.
            batch_metrics_np = jax.device_get(batch_metrics)
            epoch_metrics_np = {
                k: np.mean([metrics[k] for metrics in batch_metrics_np])
                for k in batch_metrics_np[0]
            }

            train_loss = epoch_metrics_np["loss"]
            train_acc = epoch_metrics_np["accuracy"]
            print(
                f"train epoch: {epoch}, loss: {train_loss:.4f}, accuracy: {train_acc:.4f}"
            )

            return state

        def eval_model(state, test_ds, batch_size):
            batch_metrics = []
            for batch in batchify(test_ds, batch_size):
                metrics = eval_step(state, batch)
                batch_metrics.append(metrics)
            batch_metrics = jax.device_get(batch_metrics)
            summary = {
                k: np.mean([metrics[k] for metrics in batch_metrics])
                for k in batch_metrics[0]
            }
            return summary["loss"], summary["accuracy"]

        state = TrainState.create(
            apply_fn=self.dnn.apply,
            tx=optax.adam(learning_rate=0.001),
            params=self.variables["params"],
            batch_stats=self.variables.get("batch_stats", {}),
        )

        key = jax.random.PRNGKey(rng.integers(2 ** 63))

        train_ds = dict(image=train_x, label=train_y)
        # train_ds = jax.device_put(train_ds)
        test_ds = dict(image=val_x, label=val_y)
        # test_ds = jax.device_put(test_ds)

        for epoch in range(1, epochs + 1):
            # Use a separate PRNG key to permute image data during shuffling
            key, input_key = jax.random.split(key)
            # Run an optimization step over a training batch
            state = train_epoch(state, train_ds, batch_size, epoch, input_key)
            # print("state", jax.tree_map(jnp.shape, state))
            # Evaluate on the test set after each training epoch
            test_loss, test_acc = eval_model(state, test_ds, batch_size)
            print(
                f"test epoch: {epoch}, loss: {test_loss:.4f}, accuracy: {test_acc:.4f}"
            )

        self.variables = dict(
            params=jax.device_get(state.params),
            batch_stats=jax.device_get(state.batch_stats),
        )
