from __future__ import annotations
import pathlib
import functools
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


class CNN1(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = False):
        # flax uses channels-last (NHWC) convention
        conv = functools.partial(nn.Conv, kernel_size=(1, 5), use_bias=False)
        # https://flax.readthedocs.io/en/latest/howtos/state_params.html
        norm = functools.partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.99,
            epsilon=0.001,
            axis_name="batch",  # Name batch dim
        )
        # permutation invariant layer
        symmetric = lambda axis: functools.partial(jnp.sum, axis=axis, keepdims=True)

        x = norm()(x)

        x = conv(features=32, strides=(1, 2))(x)
        x = nn.elu(x)
        x = norm()(x)

        x = conv(features=64, strides=(1, 2))(x)
        x = nn.elu(x)
        x = norm()(x)

        # collapse haplotypes
        x = symmetric(axis=-3)(x)

        x = conv(features=64)(x)
        x = nn.elu(x)
        x = norm()(x)

        # collapse genomic bins
        x = symmetric(axis=-2)(x)

        x = x.reshape((np.prod(x.shape),))  # flatten

        x = nn.Dense(features=1)(x)
        #x = nn.sigmoid(x)

        return x


def binary_accuracy(logits, labels):
    return jnp.mean(jnp.abs(jax.nn.sigmoid(logits) - labels) < 0.5)


class Discriminator:
    def __init__(self, dnn: nn.Module, variables: PyTree):
        """
        Instantiate a discriminator.

        Not intended to be used directly. Use from_file() or from_input_shape()
        class methods instead.

        :param dnn: The neural network. This has an apply() method.
        :param variables: A PyTree of the network parameters.
        """
        self.dnn = dnn
        self.variables = variables

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
        shape = (1,) + input_shape  # add leading batch dimension
        dummy_input = jnp.zeros(shape, dtype=np.int8)
        params = dnn.init(key, dummy_input)
        return cls(dnn, params)

    def summary(self):
        print(jax.tree_map(lambda x: x.shape, self.variables))

    def fit(
        self,
        rng: np.random.Generator,
        *,
        train_x,
        train_y,
        val_x,
        val_y,
        batch_size: int = 32,
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

            def batch_loss(params):
                def loss_fn(x, y):
                    logits, batch_stats = state.apply_fn(
                        dict(params=params, batch_stats=state.batch_stats),
                        x, mutable=["batch_stats"], train=True
                    )
                    loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=y)
                    return loss, logits, batch_stats

                loss, logits, batch_stats = jax.vmap(
                    loss_fn, out_axes=(0, 0, None), axis_name="batch"
                )(batch["image"], batch["label"])
                return jnp.mean(loss), (logits, batch_stats)

            grad_fn = jax.value_and_grad(batch_loss, has_aux=True)
            (loss, (logits, updated_batch_stats)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads, batch_stats=updated_batch_stats)
            metrics = dict(loss=loss, accuracy=binary_accuracy(logits, batch["label"]))
            return state, metrics

        @jax.jit
        def eval_step(state, batch):
            def loss_fn(x, y):
                params = dict(params=state.params, batch_stats=state.batch_stats)
                logits = state.apply_fn(params, x)
                loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=y)
                return loss, logits

            loss, logits = jax.vmap(loss_fn, out_axes=(0, 0), axis_name="batch")(
                batch["image"], batch["label"]
            )
            loss = jnp.mean(loss)
            accuracy = binary_accuracy(logits, batch["label"])
            metrics = dict(loss=loss, accuracy=accuracy)
            return metrics

        def train_epoch(state, train_ds, batch_size, epoch, key):
            """Train for a single epoch."""
            train_ds_size = len(train_ds["image"])
            steps_per_epoch = train_ds_size // batch_size

            perms = jax.random.permutation(key, train_ds_size)
            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            batch_metrics = []
            for perm in perms:
                batch = {k: v[perm, ...] for k, v in train_ds.items()}
                state, metrics = train_step(state, batch)
                batch_metrics.append(metrics)
                for k, v in metrics.items():
                    print(k, jax.device_get(v), end=", ")
                print("\r")

            # compute mean of metrics across each batch in epoch.
            batch_metrics_np = jax.device_get(batch_metrics)
            epoch_metrics_np = {
                k: np.mean([metrics[k] for metrics in batch_metrics_np])
                for k in batch_metrics_np[0]
            }

            train_loss = epoch_metrics_np["loss"]
            train_acc = epoch_metrics_np["accuracy"]
            print(f"train epoch: {epoch}, loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

            return state

        def eval_model(state, test_ds):
            metrics = eval_step(state, test_ds)
            metrics = jax.device_get(metrics)
            summary = jax.tree_map(lambda x: x.item(), metrics)
            return summary["loss"], summary["accuracy"]

        state = TrainState.create(
            apply_fn=self.dnn.apply,
            tx=optax.adam(learning_rate=0.001),
            params=self.variables["params"],
            batch_stats=self.variables.get("batch_stats", {}),
        )

        key = jax.random.PRNGKey(rng.integers(2 ** 63))

        train_ds = dict(image=train_x, label=train_y)
        test_ds = dict(image=val_x, label=val_y)

        for epoch in range(1, epochs + 1):
            # Use a separate PRNG key to permute image data during shuffling
            key, input_key = jax.random.split(key)
            # Run an optimization step over a training batch
            state = train_epoch(state, train_ds, batch_size, epoch, input_key)
            # Evaluate on the test set after each training epoch
            test_loss, test_acc = eval_model(state, test_ds)
            print(f"test epoch: {epoch}, loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")

        self.variables = dict(params=state.params, batch_stats=state.batch_stats)
