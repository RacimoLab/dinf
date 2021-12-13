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

from .misc import Pytree, tree_equal, tree_shape, tree_cons, tree_car, tree_cdr


# Because we use batch normalisation, the training state needs to also record
# batch_stats to maintain the running mean and variance.
class TrainState(flax.training.train_state.TrainState):
    batch_stats: Any


def batchify(dataset, batch_size):
    """Generate batch_size chunks of the dataset."""
    assert batch_size >= 1

    sizes = np.array(jax.tree_flatten(tree_car(tree_shape(dataset)))[0])
    assert np.all(sizes[0] == sizes[1:])
    size = sizes[0]

    i, j = 0, batch_size
    while i < size:
        batch = jax.tree_map(lambda x: x[i:j, ...], dataset)
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
    def __call__(  # type: ignore[override]
        self, inputs: Pytree, *, train: bool
    ) -> Pytree:
        # flax uses channels-last (NHWC) convention
        conv = functools.partial(nn.Conv, kernel_size=(1, 5), use_bias=False)
        # https://flax.readthedocs.io/en/latest/howtos/state_params.html
        norm = functools.partial(nn.BatchNorm, use_running_average=not train)

        combined = []
        for input_feature in jax.tree_leaves(inputs):
            x = norm()(input_feature)

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
            combined.append(x)

        y = nn.Dense(features=1)(combined)

        # flatten
        y = y.reshape((-1,))

        # We output logits on (-inf, inf), rather than a probability on [0, 1],
        # because the jax ecosystem provides better API support for working
        # with logits, e.g. loss functions in optax.
        # So remember to call jax.nn.sigmoid(x) on the output when
        # probabilities are needed.
        return y


@dataclasses.dataclass
class Discriminator:
    """
    A discriminator neural network.

    Not intended to be instantiated directly. Use either the from_file() or
    from_input_shape() class methods instead.

    :ivar dnn: The neural network. This has an apply() method.
    :ivar input_shape: The shape of the input to the neural network.
    :ivar variables: A Pytree of the network parameters.
    :ivar train_metrics:
        A Pytree containing the loss/accuracy metrics obtained when training
        the network.
    """

    dnn: nn.Module
    input_shape: Sequence[int]
    variables: Pytree
    train_metrics: Pytree = None
    # Bump this after making internal changes.
    discriminator_format: int = 1
    state = None

    @classmethod
    def from_input_shape(
        cls, input_shape: Pytree, rng: np.random.Generator
    ) -> Discriminator:
        """
        Build a neural network with the given input shape.

        :param input_shape:
            The shape of the input data for the network. This is a dictionary
            that maps a label to a feature array. Each feature array has shape
            (n, m, c), where
            n >= 2 is the number of (pseudo)haplotypes,
            m >= 4 is the length of the (pseudo)haplotypes,
            and c <= 4 is the number of channels.
        """
        dnn = ExchangeableCNN()
        key = jax.random.PRNGKey(rng.integers(2 ** 63))

        # Sanity checks.
        if not jax.tree_util.tree_all(
            jax.tree_map(
                lambda x: len(x) == 3 and x[0] >= 2 and x[1] >= 4 and x[2] <= 4,
                input_shape,
                is_leaf=lambda x: isinstance(x, tuple),
            )
        ):
            raise ValueError(
                "Input features must each have shape (n, m, c), where "
                "n >= 2 is the number of (pseudo)haplotypes, "
                "m >= 4 is the length of the (pseudo)haplotypes, "
                "and c <= 4 is the number of channels.\n"
                f"input_shape={input_shape}"
            )

        # add leading batch dimension
        input_shape = tree_cons(1, input_shape)
        dummy_input = jax.tree_map(
            lambda x: jnp.zeros(x, dtype=np.int8),
            input_shape,
            is_leaf=lambda x: isinstance(x, tuple),
        )

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
        a = jax.tree_map(
            lambda x: jnp.zeros(x, dtype=np.int8),
            self.input_shape,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        _, state = self.dnn.apply(
            self.variables,
            a,
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
        # tensorboard_log_dir=None,
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
        :param reset_metrics:
            If true, remove loss/accuracy metrics from previous calls to
            fit() (if any). If false, loss/accuracy metrics will be appended
            to the existing metrics.
        """
        train_x_sizes, train_y_sizes, val_x_sizes, val_y_sizes = map(
            lambda tree: np.array(jax.tree_flatten(tree_car(tree_shape(train_x)))[0]),
            [train_x, train_y, val_x, val_y],
        )
        train_size = train_x_sizes[0]
        if not np.all(train_size == train_x_sizes) or not np.all(
            train_size == train_y_sizes
        ):
            raise ValueError(
                "Leading dimensions of train_x and train_y must be the same.\n"
                f"train_x={tree_shape(train_x)}\n"
                f"train_y={tree_shape(train_y)}"
            )
        val_size = val_x_sizes[0]
        if not np.all(val_size == val_x_sizes) or not np.all(val_size == val_y_sizes):
            raise ValueError(
                "Leading dimensions of val_x and val_y must be the same.\n"
                f"val_x={tree_shape(val_x)}\n"
                f"val_y={tree_shape(val_y)}"
            )
        if not tree_equal(
            *map(tree_cdr, [self.input_shape, tree_shape(train_x), tree_shape(val_x)])
        ):
            raise ValueError(
                "Trailing dimensions of train_x and val_x must match input_shape.\n"
                f"input_shape={self.input_shape}\n"
                f"train_x={tree_shape(train_x)}\n"
                f"val_x={tree_shape(val_x)}\n"
                f"input_shape={tree_cdr(self.input_shape)}\n"
                f"train_x={tree_cdr(tree_shape(train_x))}\n"
                f"val_x={tree_cdr(tree_shape(val_x))}"
            )
        if not tree_equal(*map(tree_cdr, map(tree_shape, [train_y, val_y]))):
            raise ValueError(
                "Trailing dimensions of train_y and val_y must match.\n"
                f"train_y={tree_cdr(train_y)}\n"
                f"val_y={tree_cdr(val_y)}"
            )

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
                actual_batch_size = len(batch["input"])
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
                actual_batch_size = len(batch["input"])
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

        train_ds = dict(input=train_x, output=train_y)
        test_ds = dict(input=val_x, output=val_y)
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

        shape_x = tree_shape(x)
        sizes_x = np.array(jax.tree_flatten(tree_car(shape_x))[0])
        if not np.all(sizes_x[0] == sizes_x[1:]):
            raise ValueError(
                f"Leading dimensions of input features must match.\nx={shape_x}"
            )
        if not tree_equal(*map(tree_cdr, [self.input_shape, shape_x])):
            raise ValueError(
                "Trailing dimensions of x must match input_shape.\n"
                f"input_shape={self.input_shape}\n"
                f"x={shape_x}"
            )

        if "batch_stats" not in self.variables:
            raise ValueError(
                "Cannot make predications as the discriminator has not been trained."
            )

        dataset = dict(input=x)
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
            batch["input"],
            mutable=["batch_stats"],
            train=True,
        )
        loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch["output"])
        )
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )
    metrics = dict(
        loss=loss,
        accuracy=binary_accuracy(logits=logits, labels=batch["output"]),
    )
    return state, metrics


@jax.jit
def _eval_step(state, batch):
    """Evaluate for a single step."""

    logits = state.apply_fn(
        dict(params=state.params, batch_stats=state.batch_stats),
        batch["input"],
        train=False,
    )
    loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch["output"])
    )
    accuracy = binary_accuracy(logits=logits, labels=batch["output"])
    metrics = dict(loss=loss, accuracy=accuracy)
    return metrics


@functools.partial(jax.jit, static_argnums=(2,))
def _predict_batch(batch, variables, apply_func):
    """Make predictions on a batch."""

    logits = apply_func(
        variables,
        batch["input"],
        train=False,
    )
    return jax.nn.sigmoid(logits)
