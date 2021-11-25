from __future__ import annotations
import dataclasses
import functools
import pathlib
import pickle
import sys
from typing import Any, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import flax.training.train_state
import optax

# A type for jax PyTrees.
# https://github.com/google/jax/issues/3340
PyTree = Any


# Because we use batch normalisation, the training state needs to also record
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
    Network layer that summarises over a given axis in a way that is invariant
    to permutations of the input along that axis.
    """

    axis: int

    @nn.compact
    def __call__(self, x):
        # TODO: Assess choice of symmetric function more formally.
        # One-shot tests indicated that sum and variance work well very well,
        # but variance trains quicker; mean and median work less well.
        # Chan et al. suggest some alternatives which I haven't tried:
        #  - max
        #  - mean of the top decile
        #  - higher moments
        return jnp.var(x, axis=self.axis, keepdims=True)


class ExchangeableCNN(nn.Module):
    """
    An exchangeable CNN for the discriminator.

    Chan et al. 2018, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7687905/
    """

    @nn.compact
    def __call__(self, x: PyTree, *, train: bool) -> PyTree:  # type: ignore[override]
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

        # flatten
        x = x.reshape((-1,))

        # We output logits on (-inf, inf), rather than a probability on [0, 1],
        # because the jax ecosystem provides better API support for working
        # with logits, e.g. loss functions in optax.
        # So remember to call jax.nn.sigmoid(x) on the output when
        # probabilities are needed.
        return x


@dataclasses.dataclass
class Discriminator:
    """
    A discriminator neural network.

    Not intended to be instantiated directly. Use either the from_file() or
    from_input_shape() class methods instead.

    :ivar dnn: The neural network. This has an apply() method.
    :ivar input_shape: The shape of the input to the neural network.
    :ivar variables: A PyTree of the network parameters.
    :ivar train_metrics:
        A PyTree containing the loss/accuracy metrics obtained when training
        the network.
    """

    dnn: nn.Module
    input_shape: Sequence[int]
    variables: PyTree
    train_metrics: PyTree = None
    # Bump this after making internal changes.
    discriminator_format: str = "0.0.1"
    state = None

    @classmethod
    def from_input_shape(
        cls, input_shape: Sequence[int], rng: np.random.Generator
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
        dnn = ExchangeableCNN()
        key = jax.random.PRNGKey(rng.integers(2 ** 63))
        input_shape = (1,) + tuple(input_shape)  # add leading batch dimension
        dummy_input = jnp.zeros(input_shape, dtype=np.int8)

        @jax.jit
        def init(*args):
            return dnn.init(*args, train=False)

        variables = init(key, dummy_input)
        return cls(dnn=dnn, variables=variables, input_shape=input_shape)

    @classmethod
    def from_file(cls, filename: str | pathlib.Path) -> Discriminator:
        """
        Load discriminator neural network from the given file.

        :param filename: The filename of the saved model.
        :return: The discriminator object.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        if data.pop("discriminator_format") != cls.discriminator_format:
            raise ValueError(
                f"{filename}: saved discriminator is not compatible with this "
                "version of dinf. Either train a new discriminator or use an "
                "older version of dinf."
            )
        expected_fields = set(map(lambda f: f.name, dataclasses.fields(cls)))
        expected_fields.remove("discriminator_format")
        assert data.keys() == expected_fields
        return cls(**data)

    def to_file(self, filename) -> None:
        """
        Save discriminator neural network to the given file.

        :param filename: The filename to which the model will be saved.
        """
        data = dataclasses.asdict(self)
        data["dnn"] = self.dnn  # asdict converts this to a dict
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def summary(self):
        """Print a summary of the neural network."""
        x = jnp.zeros(self.input_shape, dtype=np.int8)
        _, state = self.dnn.apply(
            self.variables,
            x,
            train=False,
            capture_intermediates=True,
            mutable=["intermediates"],
        )
        # TODO: this sucks. The order of layers in the CNN are lost, because of
        # https://github.com/google/jax/issues/4085
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
        # TODO: tensorboard output
        tensorboard_log_dir=None,
        reset_metrics: bool = False,
    ):
        """
        Fit discriminator to training data.

        :param rng: Numpy random number generator.
        :param train_x: Training data.
        :param train_y: Labels for training data.
        :param val_x: Validation data.
        :param val_y: Labels for validation data.
        :param batch_size: Size of minibatch for gradient update step.
        :param epochs: The number of full passes over the training data.
        :param tensorboard_log_dir:
            Directory for tensorboard logs. If None, no logs will be recorded.
        :param reset_metrics:
            If true, remove loss/accuracy metrics from previous calls to
            fit() (if any). If false, loss/accuracy metrics will be appended
            to the existing metrics.
        """
        assert len(train_y.shape) == len(val_y.shape) == 1
        assert len(train_x.shape) == len(val_x.shape) == 4
        assert train_x.shape[1:] == val_x.shape[1:]
        assert train_x.shape[1] > 1
        assert train_x.shape[2] > 1
        assert train_x.shape[3] == 1

        def running_metrics(n, batch_size, current_metrics, metrics):
            new_metrics = jax.tree_map(
                lambda a, b: a + batch_size * b, current_metrics, metrics
            )
            return n + batch_size, new_metrics

        def train_epoch(state, train_ds, batch_size, epoch, key):
            """Train for a single epoch."""

            def print_metrics(n, metrics_sum, end):
                loss = metrics_sum["loss"] / n
                accuracy = metrics_sum["accuracy"] / n
                print(
                    f"[epoch {epoch}|{n}] "
                    f"train loss {loss:.4f}, accuracy {accuracy:.4f}",
                    end=end,
                )
                return loss, accuracy

            metrics_sum = dict(loss=0, accuracy=0)
            n = 0
            for i, batch in enumerate(batchify(train_ds, batch_size)):
                state, batch_metrics = _train_step(state, batch)
                actual_batch_size = len(batch["image"])
                n, metrics_sum = running_metrics(
                    n, actual_batch_size, metrics_sum, batch_metrics
                )
                if i % 20 == 0:
                    print_metrics(n, metrics_sum, end="\r")

            loss, accuracy = print_metrics(n, metrics_sum, end="")
            sys.stdout.flush()
            return loss, accuracy, state

        def eval_model(state, test_ds, batch_size):
            metrics_sum = dict(loss=0, accuracy=0)
            n = 0
            for batch in batchify(test_ds, batch_size):
                batch_metrics = _eval_step(state, batch)
                actual_batch_size = len(batch["image"])
                n, metrics_sum = running_metrics(
                    n, actual_batch_size, metrics_sum, batch_metrics
                )
            loss = metrics_sum["loss"] / n
            accuracy = metrics_sum["accuracy"] / n
            print(f"; test loss {loss:.4f}, accuracy {accuracy:.4f}")
            return loss, accuracy

        state = self.state
        if state is None:
            state = TrainState.create(
                apply_fn=self.dnn.apply,
                tx=optax.adam(learning_rate=0.001),
                params=self.variables["params"],
                batch_stats=self.variables.get("batch_stats", {}),
            )

        train_ds = dict(image=train_x, label=train_y)
        test_ds = dict(image=val_x, label=val_y)
        do_eval = len(val_x) > 0

        if reset_metrics or self.train_metrics is None:
            self.train_metrics = dict(
                train_loss=[],
                train_accuracy=[],
            )
            if do_eval:
                self.train_metrics.update(
                    test_loss=[],
                    test_accuracy=[],
                )

        seed = rng.integers(1 ** 63)
        keys = jax.random.split(jax.random.PRNGKey(seed), epochs)
        for epoch, key in enumerate(keys, 1):
            train_loss, train_accuracy, state = train_epoch(
                state, train_ds, batch_size, epoch, key
            )
            self.train_metrics["train_loss"].append(train_loss)
            self.train_metrics["train_accuracy"].append(train_accuracy)

            if do_eval:
                test_loss, test_accuracy = eval_model(state, test_ds, batch_size)
                self.train_metrics["test_loss"].append(test_loss)
                self.train_metrics["test_accuracy"].append(test_accuracy)
            else:
                print()

        self.state = state
        self.variables = jax.tree_map(
            np.array,
            dict(
                params=jax.device_get(state.params),
                batch_stats=jax.device_get(state.batch_stats),
            ),
        )

    def predict(self, x, *, batch_size: int = 1024) -> np.ndarray:
        """
        Make predictions about data using a pre-fitted neural network.

        :param x: The data instances about which to make predictions.
        :param batch_size: Size of data batches for prediction.
        :return: A vector of predictions, one for each input instance.
        """
        if len(x.shape) != 4 or x.shape[1:] != self.input_shape[1:]:
            # TODO: relax this, because the network is exchangeable.
            raise ValueError(
                f"Input data has shape {x.shape} but discriminator network "
                f"expects shape {self.input_shape}."
            )

        if "batch_stats" not in self.variables:
            raise ValueError(
                "Cannot make predications as the discriminator has not been trained."
            )

        dataset = dict(image=x)
        y = []
        for batch in batchify(dataset, batch_size):
            y.append(_predict_batch(batch, self.variables, self.dnn.apply))
        return np.concatenate(y)


##
# Jitted functions below are at the top level so they only get jitted once.


def binary_accuracy(*, logits, labels):
    """Accuracy of binary classifier, from logits."""
    p = jax.nn.sigmoid(logits)
    return jnp.mean(labels == (p > 0.5))


@jax.jit
def _train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            dict(params=params, batch_stats=state.batch_stats),
            batch["image"],
            mutable=["batch_stats"],
            train=True,
        )
        loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch["label"])
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
def _eval_step(state, batch):
    """Evaluate for a single step."""

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


@functools.partial(jax.jit, static_argnums=(2,))
def _predict_batch(batch, variables, apply_func):
    """Make predictions on a batch."""

    logits = apply_func(
        variables,
        batch["image"],
        train=False,
    )
    return jax.nn.sigmoid(logits)
