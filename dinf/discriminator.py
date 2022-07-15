from __future__ import annotations
import collections
import dataclasses
import functools
import pathlib
import sys
import time
from typing import Callable, Tuple

from flax import linen as nn
import flax.training.train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.stats

from .misc import (
    Pytree,
    tree_equal,
    tree_shape,
    tree_cons,
    tree_cdr,
    leading_dim_size,
)

# Small fudge-factor to avoid numerical instability.
EPSILON = jnp.finfo(jnp.float32).eps


# Because we use batch normalisation, the training state needs to also record
# batch_stats to maintain the running mean and variance.
class TrainState(flax.training.train_state.TrainState):
    batch_stats: Pytree


def batchify(dataset, batch_size, random=False, rng=None):
    """Generate batch_size chunks of the dataset."""
    assert batch_size >= 1
    size = leading_dim_size(dataset)
    assert size >= 1

    if random:
        assert rng is not None
        indices = rng.permutation(size)

    i, j = 0, batch_size
    while i < size:
        if random:
            batch = jax.tree_map(lambda x: x[indices[i:j], ...], dataset)
        else:
            batch = jax.tree_map(lambda x: x[i:j, ...], dataset)
        yield batch
        i = j
        j += batch_size


class Symmetric(nn.Module):
    """
    Network layer that summarises over a given axis in a way that is invariant
    to permutations of the input along that axis.

    | Chan et al. 2018, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7687905/

    :param func:
        The permutation-invariant function to apply.
        Must be one of:

         * "max": maximum value.
         * "mean": mean value.
         * "sum": sum of the values.
         * "var": variance of the values.
         * "moments": first ``k`` moments, :math:`E[x], E[x^2], ..., E[x^k]`.
         * "central-moments": first ``k`` central moments,
           :math:`E[x], E[(x - E[x])^2], ..., E[(x - E[x])^k]`.

    :param k:
        The number of moments to take when ``func`` is "moments" or "central-moments".
    """

    func: str
    k: int | None = None

    def setup(self):
        choices = ("max", "mean", "sum", "var", "moments", "central-moments")
        if self.func not in choices:
            raise ValueError(
                f"Unexpected func `{self.func}'. Must be one of {choices}."
            )
        needs_k = ("moments", "central-moments")
        if self.func in needs_k and (self.k is None or self.k < 2):
            raise ValueError(f"Must have k >= 2 when func is one of {needs_k}.")

    @nn.compact
    def __call__(self, x, *, axis):
        # Chan et al. also suggests trying (1) the top k,
        # and (2) the mean of the top decile, which I've not tried implementing.
        if self.func == "max":
            return jnp.max(x, axis=axis, keepdims=True)
        elif self.func == "mean":
            return jnp.mean(x, axis=axis, keepdims=True)
        elif self.func == "sum":
            return jnp.sum(x, axis=axis, keepdims=True)
        elif self.func == "var":
            return jnp.var(x, axis=axis, keepdims=True)
        elif self.func == "moments":
            assert self.k is not None and self.k >= 2
            return jnp.concatenate(
                [
                    jnp.mean(x**i, axis=axis, keepdims=True)
                    for i in range(1, self.k + 1)
                ],
                axis=axis,
            )
        elif self.func == "central-moments":
            assert self.k is not None and self.k >= 2
            mean = jnp.mean(x, axis=axis, keepdims=True)
            return jnp.concatenate(
                [mean]
                + [
                    jnp.mean((x - mean) ** i, axis=axis, keepdims=True)
                    for i in range(2, self.k + 1)
                ],
                axis=axis,
            )


class ExchangeablePGGAN(nn.Module):
    """
    An exchangeable CNN for the discriminator using the PG-GAN architecture.

    This is a faithful translation of the network implemented in PG-GAN [1],
    with the addition of batch normalisation.
    Each haplotype is treated as exchangeable with any other
    haplotype within the same feature matrix (i.e. exchangeability
    applies within labelled groups).

    Each feature matrix in the input has shape
    (batch_size, num_haplotypes, num_loci, channels).
    Two 1-d convolution layers are applied along haplotypes, and feature
    size is reduced following each convolution using 1-d max pooling with
    size and stride of 2. The sum function is then used to collapse
    features across the num_haplotypes dimension. Two fully connected layers
    with dropout follow.
    Relu activation is used throughout, except for the final output which has
    no activation.

    The number of trainable parameters in the network is dependent on the
    number of feature matrices in the input, and dependent on the num_loci
    dimension of the feature matrices.

    | [1] Wang et al. 2021, https://doi.org/10.1111/1755-0998.13386

    :param sizes1:
        A tuple of feature sizes for the convolution layers before the
        symmetric function. Convolution weights are tied between convolution
        layers of the distinct labelled feature matrices.
    :param symmetric:
        The symmetric network layer.
    :param sizes2:
        A tuple of feature sizes for the dense (fully connected) layers
        afer the symmetric function.
    """

    sizes1: Tuple[int, ...] = (32, 64)
    symmetric: nn.Module = Symmetric(func="sum")
    sizes2: Tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(  # type: ignore[override]
        self, inputs: Pytree, *, train: bool
    ) -> Pytree:
        kernel_init = nn.initializers.glorot_uniform()
        # Convolution weights are shared across feature matrices.
        # Flax uses channels-last (NHWC) convention.
        conv_layers = [
            nn.Conv(
                features=features,
                strides=(1, 1),
                kernel_size=(1, 5),
                use_bias=True,
                kernel_init=kernel_init,
            )
            for features in self.sizes1
        ]
        pool = functools.partial(nn.max_pool, window_shape=(1, 2), strides=(1, 2))
        # https://flax.readthedocs.io/en/latest/howtos/state_params.html
        norm = functools.partial(nn.BatchNorm, use_running_average=not train)
        activation = nn.relu

        xs = []
        for input_feature in jax.tree_leaves(inputs):
            x = input_feature
            x = norm()(x)

            for conv in conv_layers:
                x = conv(x)
                x = activation(x)
                x = pool(x)
                x = norm()(x)

            # collapse haplotypes
            x = self.symmetric(x, axis=1)
            # flatten
            x = jnp.reshape(x, (x.shape[0], -1))
            xs.append(x)

        x = jnp.concatenate(xs, axis=-1)

        for features in self.sizes2:
            x = nn.Dense(features=features, kernel_init=kernel_init)(x)
            x = activation(x)
            x = norm()(x)
            x = nn.Dropout(rate=0.5)(x, deterministic=not train)

        y = nn.Dense(features=1)(x)
        # flatten
        y = jnp.reshape(y, (-1,))
        # We output logits on (-inf, inf), rather than a probability on [0, 1].
        # So remember to call jax.nn.sigmoid(x) on the output when
        # probabilities are needed.
        return y


class ExchangeableCNN(nn.Module):
    """
    An exchangeable CNN for the discriminator.

    Each feature matrix in the input has shape
    (batch_size, num_haplotypes, num_loci, channels).
    Two 1-d convolution layers are applied along haplotypes and
    feature size is reduced in each convolution using a stride of 2.
    Then the max function is used to collapse
    features across the num_haplotypes dimension. A third 1-d convolution
    follows (with stride of 1), and the max function is used to
    collapse features across the num_loci dimension.
    Elu activation is used throughout, except for the final output which has
    no activation.

    The number of trainable parameters in the network is dependent on the
    number of feature matrices in the input, but independent of the size of
    the feature matrices.

    :param sizes1:
        A tuple of feature sizes for the convolution layers before the
        symmetric function.
    :param symmetric:
        The symmetric network layer.
    :param sizes2:
        A tuple of feature sizes for the convolution layers afer the
        symmetric function.
    """

    sizes1: Tuple[int, ...] = (32, 64)
    symmetric: nn.Module = Symmetric(func="max")
    sizes2: Tuple[int, ...] = (64,)

    @nn.compact
    def __call__(  # type: ignore[override]
        self, inputs: Pytree, *, train: bool
    ) -> Pytree:
        # Flax uses channels-last (NHWC) convention.
        conv = functools.partial(nn.Conv, kernel_size=(1, 5), use_bias=False)
        # https://flax.readthedocs.io/en/latest/howtos/state_params.html
        norm = functools.partial(nn.BatchNorm, use_running_average=not train)
        activation: Callable = nn.elu

        xs = []
        for input_feature in jax.tree_leaves(inputs):
            x = norm()(input_feature)

            for features in self.sizes1:
                x = conv(features=features, strides=(1, 2))(x)
                x = activation(x)
                x = norm()(x)

            # collapse haplotypes
            x = self.symmetric(x, axis=1)

            for features in self.sizes2:
                x = conv(features=features, strides=(1, 1))(x)
                x = activation(x)
                x = norm()(x)

            # collapse loci
            x = self.symmetric(x, axis=2)
            xs.append(x)

        x = jnp.concatenate(xs, axis=-1)
        # flatten
        x = jnp.reshape(
            x,
            (
                x.shape[0],
                -1,
            ),
        )
        y = nn.Dense(features=1)(x)
        # flatten
        y = jnp.reshape(y, (-1,))
        # We output logits on (-inf, inf), rather than a probability on [0, 1].
        # So remember to call jax.nn.sigmoid(x) on the output when
        # probabilities are needed.
        return y


@dataclasses.dataclass
class _NetworkWrapper:
    """
    Base class for Discriminator and Surrogate wrappers.
    """

    input_shape: Pytree
    """The shape of the input to the neural network."""

    network: nn.Module | None = None
    """The flax neural network."""

    state: TrainState | None = None
    """Network training state, e.g. network parameters and batch statistics."""

    trained: bool = False
    """True if the network has been trained, False otherwise."""

    metrics: collections.abc.Mapping = dataclasses.field(
        default_factory=lambda: collections.defaultdict(list)
    )
    """Loss and accuracy metrics obtained when training the network."""

    # Bump this after making internal changes.
    format_version: int = 3
    """Version number for the serialised network on disk."""

    _add_batch_dim: bool = True
    _inited: bool = False

    def __post_init__(self):
        assert self.network is not None

        if self.state is None:
            self.state = TrainState.create(
                apply_fn=self.network.apply,
                tx=optax.chain(
                    optax.clip(1.0),
                    optax.adam(learning_rate=0.001),
                ),
                params={},
                batch_stats={},
            )
        if self.input_shape is not None and self._add_batch_dim:
            # Add leading batch dimension.
            if isinstance(self.input_shape, int):
                self.input_shape = (self.input_shape,)
            self.input_shape = tree_cons(1, self.input_shape)

    def init(self, rng: np.random.Generator):
        """
        Build a neural network with the given input shape.

        :param numpy.random.Generator rng:
            The numpy random number generator.
        :return:
            A new network wrapper object.
        """
        assert self.input_shape is not None
        assert self.state is not None

        @jax.jit
        def init(*args):
            return self.network.init(*args, train=False)

        dummy_input = jax.tree_map(
            lambda x: jnp.zeros(x, dtype=np.float32),
            self.input_shape,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        key = jax.random.PRNGKey(rng.integers(2**63))
        variables = init(key, dummy_input)
        state = type(self.state).create(
            apply_fn=self.state.apply_fn,
            tx=self.state.tx,
            params=variables["params"],
            batch_stats=variables.get("batch_stats", {}),
        )
        return dataclasses.replace(
            self, state=state, _add_batch_dim=False, _inited=True
        )

    def from_file(self, filename: str | pathlib.Path, /):
        """
        Load neural network training state from the given file.

        :param filename:
            The filename of the saved model.
        :return:
            A new network wrapper object.
        """
        with open(filename, "rb") as f:
            data = flax.serialization.msgpack_restore(f.read())
        return self.fromdict(data, _err_prefix=f"{filename}: ")

    def to_file(self, filename: str | pathlib.Path, /) -> None:
        """
        Save neural network to the given file.

        :param filename: The path where the model will be saved.
        """
        data = self.asdict()
        with open(filename, "wb") as f:
            f.write(flax.serialization.msgpack_serialize(data))

    def fromdict(self, data: dict, _err_prefix: str = ""):
        assert self.state is not None

        format_version = data.get("format_version")
        if format_version is not None and format_version != self.format_version:
            raise ValueError(
                f"{_err_prefix}saved network is not compatible with this "
                "version of dinf. Either train a new network or use an "
                "older version of dinf."
            )
        expected_fields = {
            "input_shape",
            "state",
            "trained",
            "metrics",
            "format_version",
        }
        if data.keys() != expected_fields:
            raise ValueError(
                f"{_err_prefix}not recognisable as a Dinf network.\n"
                f"Expected {expected_fields},\nbut got {set(data.keys())}."
            )

        data["input_shape"] = jax.tree_map(
            tuple,
            data["input_shape"],
            is_leaf=lambda x: isinstance(x, list),
        )
        data["state"] = type(self.state).create(
            apply_fn=self.state.apply_fn,
            tx=self.state.tx,
            params=data["state"]["params"],
            batch_stats=data["state"].get("batch_stats", {}),
        )
        data["metrics"] = collections.defaultdict(list, **data["metrics"])
        return dataclasses.replace(self, _add_batch_dim=False, _inited=True, **data)

    def asdict(self) -> dict:
        input_shape = jax.tree_map(
            list,
            self.input_shape,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        state = flax.serialization.to_state_dict(self.state)
        metrics = {k: np.array(v) for k, v in self.metrics.items()}
        return dict(
            input_shape=input_shape,
            state=state,
            trained=self.trained,
            metrics=metrics,
            format_version=self.format_version,
        )

    def summary(self):
        """Print a summary of the neural network."""
        # XXX: The order of layers in the CNN are lost because of
        # https://github.com/google/jax/issues/4085

        a = jax.tree_map(
            lambda x: jnp.zeros(x, dtype=np.float32),
            self.input_shape,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        print(self.network.tabulate(jax.random.PRNGKey(0), a, train=False))


@dataclasses.dataclass
class Discriminator(_NetworkWrapper):
    """
    Wrapper of the discriminator network for classifying genotype matrices.

    :param input_shape:
        The shape of the input data for the network. This is a dictionary
        that maps a label to a feature array. Each feature array has shape
        (n, m, c), where
        n >= 2 is the number of (pseudo)haplotypes,
        m >= 4 is the length of the (pseudo)haplotypes,
        and c <= 4 is the number of channels.
    """

    def __post_init__(self):
        if self.network is None:
            self.network = ExchangeableCNN()
        super().__post_init__()

        if self.input_shape is None:
            return

        # Sanity checks.
        if not jax.tree_util.tree_all(
            jax.tree_map(
                lambda x: np.shape(x) == (4,) and x[1] >= 2 and x[2] >= 4 and x[3] <= 4,
                self.input_shape,
                is_leaf=lambda x: isinstance(x, tuple),
            )
        ):
            raise ValueError(
                "Input features must each have shape (b, n, m, c), where "
                "b is the number of batches,"
                "n >= 2 is the number of (pseudo)haplotypes, "
                "m >= 4 is the length of the (pseudo)haplotypes, "
                "and c <= 4 is the number of channels.\n"
                f"input_shape={self.input_shape}"
            )

    def fit(
        self,
        *,
        train_x,
        train_y,
        val_x=None,
        val_y=None,
        batch_size: int = 64,
        epochs: int = 1,
        rng: np.random.Generator,
        reset_metrics: bool = False,
        entropy_regularisation: bool = False,
    ):
        """
        Fit discriminator to training data.

        :param train_x: Training data.
        :param train_y: Labels for training data.
        :param val_x: Validation data.
        :param val_y: Labels for validation data.
        :param batch_size: Size of minibatch for gradient update step.
        :param epochs: The number of full passes over the training data.
        :param numpy.random.Generator rng:
            The numpy random number generator.
        :param reset_metrics:
            If true, remove loss/accuracy metrics from previous calls to
            fit() (if any). If false, loss/accuracy metrics will be appended
            to the existing metrics.
        """
        assert self.network is not None
        assert self.state is not None
        assert self.input_shape is not None
        assert self._inited
        if leading_dim_size(train_x) != leading_dim_size(train_y):
            raise ValueError(
                "Leading dimensions of train_x and train_y must be the same.\n"
                f"train_x={tree_shape(train_x)}\n"
                f"train_y={tree_shape(train_y)}"
            )
        if not tree_equal(*map(tree_cdr, [self.input_shape, tree_shape(train_x)])):
            raise ValueError(
                "Trailing dimensions of train_x must match input_shape.\n"
                f"input_shape={self.input_shape}\n"
                f"train_x={tree_shape(train_x)}"
            )

        if (val_x is None and val_y is not None) or (
            val_x is not None and val_y is None
        ):
            raise ValueError("Must specify both val_x and val_y or neither.")

        if val_x is not None:
            if leading_dim_size(val_x) != leading_dim_size(val_y):
                raise ValueError(
                    "Leading dimensions of val_x and val_y must be the same.\n"
                    f"val_x={tree_shape(val_x)}\n"
                    f"val_y={tree_shape(val_y)}"
                )

            if not tree_equal(*map(tree_cdr, [self.input_shape, tree_shape(val_x)])):
                raise ValueError(
                    "Trailing dimensions of val_x must match input_shape.\n"
                    f"input_shape={self.input_shape}\n"
                    f"val_x={tree_shape(val_x)}"
                )

            # For a binary classifier, y has no trailing dimensions.
            # if not tree_equal(*map(tree_cdr, map(tree_shape, [train_y, val_y]))):
            #    raise ValueError(
            #        "Trailing dimensions of train_y and val_y must match.\n"
            #        f"train_y={tree_cdr(train_y)}\n"
            #        f"val_y={tree_cdr(val_y)}"
            #    )

        def running_metrics(n, batch_size, current_metrics, metrics):
            new_metrics = jax.tree_map(
                lambda a, b: a + batch_size * b, current_metrics, metrics
            )
            return n + batch_size, new_metrics

        def train_epoch(state, train_ds, batch_size, epoch, dropout_rng, rng=rng):
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
            t_prev = time.time()
            for batch in batchify(train_ds, batch_size, random=True, rng=rng):
                state, batch_metrics = _train_step(
                    state,
                    batch,
                    dropout_rng,
                    entropy_regularisation=entropy_regularisation,
                )
                actual_batch_size = leading_dim_size(batch["input"])
                n, metrics_sum = running_metrics(
                    n, actual_batch_size, metrics_sum, batch_metrics
                )
                t_now = time.time()
                if t_now - t_prev > 0.5:
                    print_metrics(n, metrics_sum, end="\r")
                    t_prev = t_now

            loss, accuracy = print_metrics(n, metrics_sum, end="")
            sys.stdout.flush()
            return loss, accuracy, state

        def eval_model(state, test_ds, batch_size):
            metrics_sum = dict(loss=0, accuracy=0)
            n = 0
            for batch in batchify(test_ds, batch_size):
                batch_metrics = _eval_step(state, batch)
                actual_batch_size = leading_dim_size(batch["input"])
                n, metrics_sum = running_metrics(
                    n, actual_batch_size, metrics_sum, batch_metrics
                )
            loss = metrics_sum["loss"] / n
            accuracy = metrics_sum["accuracy"] / n
            print(f"; test loss {loss:.4f}, accuracy {accuracy:.4f}")
            return loss, accuracy

        train_ds = dict(input=train_x, output=train_y)
        test_ds = dict(input=val_x, output=val_y)
        do_eval = val_x is not None

        if reset_metrics:
            self.metrics = collections.defaultdict(list)

        dropout_rng1 = jax.random.PRNGKey(rng.integers(2**63))

        for epoch in range(1, epochs + 1):
            dropout_rng1, dropout_rng2 = jax.random.split(dropout_rng1)
            train_loss, train_accuracy, self.state = train_epoch(
                self.state, train_ds, batch_size, epoch, dropout_rng2, rng=rng
            )
            self.metrics["train_loss"].append(train_loss)
            self.metrics["train_accuracy"].append(train_accuracy)

            if do_eval:
                test_loss, test_accuracy = eval_model(self.state, test_ds, batch_size)
                self.metrics["test_loss"].append(test_loss)
                self.metrics["test_accuracy"].append(test_accuracy)
            else:
                print()

        self.trained = True

        # Return the metrics from the last epoch of training.
        metrics_conclusion = {k: v[-1] for k, v in self.metrics.items()}
        return metrics_conclusion

    def predict(self, x, *, batch_size: int = 1024) -> np.ndarray:
        """
        Make predictions about data using a pre-fitted neural network.

        :param x: The data instances about which to make predictions.
        :param batch_size: Size of data batches for prediction.
        :return: A vector of predictions, one for each input instance.
        """
        assert self.network is not None
        assert self.state is not None
        assert self.input_shape is not None
        assert leading_dim_size(x) > 0
        if not tree_equal(*map(tree_cdr, [self.input_shape, tree_shape(x)])):
            raise ValueError(
                "Trailing dimensions of x must match input_shape.\n"
                f"input_shape={self.input_shape}\n"
                f"x={tree_shape(x)}"
            )

        if not self.trained:
            raise ValueError(
                "Cannot make predications as the discriminator has not been trained."
            )

        dataset = dict(input=x)
        y = []
        variables = dict(params=self.state.params, batch_stats=self.state.batch_stats)
        for batch in batchify(dataset, batch_size):
            y.append(_predict_batch(batch, variables, self.network.apply))
        return np.concatenate(y)


##
# Jitted functions below are at the top level so they only get jitted once.


def binary_accuracy(*, logits, labels):
    """Accuracy of binary classifier, from logits."""
    p = jax.nn.sigmoid(logits)
    return jnp.mean(labels == (p > 0.5))


@functools.partial(jax.jit, static_argnums=(3,))
def _train_step(state, batch, dropout_rng, entropy_regularisation=False):
    """Train for a single step."""

    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            dict(params=params, batch_stats=state.batch_stats),
            batch["input"],
            mutable=["batch_stats"],
            train=True,
            rngs={"dropout": dropout_rng},
        )
        loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch["output"])
        )
        if entropy_regularisation:
            # Entropy regularisation such that the network doesn't give up and
            # predict all ones or all zeros. We weight the regularisation by a
            # constant, c, according to how balanced the batch labels are.
            # If the batch labels are all 0 or all 1 we don't do regularisation,
            # but if there are equal 0 and 1 labels, then we fully apply
            # the regularisation.
            # Similar to PG-GAN, but see
            #   https://github.com/mathiesonlab/pg-gan/issues/3
            logits_entropy = jnp.mean(
                optax.sigmoid_binary_cross_entropy(
                    logits=logits, labels=jax.nn.sigmoid(logits)
                )
            )
            c = 1.0 - 2 * jnp.abs(0.5 - jnp.mean(batch["output"]))
            regularisation = 0.01 * c * logits_entropy
            loss -= regularisation
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
    logits = apply_func(variables, batch["input"], train=False)
    return jax.nn.sigmoid(logits)


##
# Surrogate network stuff.


class SurrogateMLP(nn.Module):
    layers: Tuple[int, ...] = (64, 64, 32)

    @nn.compact
    def __call__(self, inputs, *, train: bool):  # type: ignore[override]
        num_params = inputs.shape[-1]
        x = inputs
        x = nn.BatchNorm(use_running_average=not train)(x)
        for i, size in enumerate(self.layers):
            if i == 0:
                # Expand capacity of first layer according to input size.
                size *= num_params
            x = nn.Dense(size)(x)
            x = nn.gelu(x)
        x = nn.Dense(2)(x)
        x = EPSILON + jax.nn.softplus(x)
        alpha, beta = x[..., 0], x[..., 1]
        return alpha, beta


@dataclasses.dataclass
class Surrogate(_NetworkWrapper):
    """
    Wrapper of the surrogate network for the beta-estimation model of ALFI.

    This network predicts the output of the discriminator network
    from a set of dinf model parameters, thus bypassing the generator.
    Kim et al. 2020, https://arxiv.org/abs/2004.05803v1
    """

    network: nn.Module = SurrogateMLP()
    """The flax neural network module used as the surrogate."""

    def fit(
        self,
        *,
        train_x,
        train_y,
        val_x=None,
        val_y=None,
        batch_size: int = 64,
        epochs: int = 1,
    ):
        assert self.network is not None
        assert self.state is not None
        assert self.input_shape is not None
        assert self._inited

        # The density of labels is not uniform over [0, 1], so we weight the
        # loss for each instance by the inverse density at that point.
        # Yang et al. 2021, https://arxiv.org/abs/2102.09554
        def weights(p, bandwidth=0.2, num_points=100):
            """
            Get the inverse of the density (Gaussian KDE) at points p.
            """
            kde = scipy.stats.gaussian_kde(p, bandwidth)
            x = np.linspace(0, 1, num_points)
            density = kde(x)
            weights = 1.0 / np.interp(p, x, density)
            weights = np.sqrt(weights)
            return weights

        do_eval = val_x is not None
        train_ds = dict(input=train_x, output=train_y, weights=weights(train_y))
        if do_eval:
            test_ds = dict(input=val_x, output=val_y, weights=weights(val_y))

        for epoch in range(epochs):
            loss_sum = 0
            n = 0
            for batch in batchify(train_ds, batch_size):
                self.state, loss = _train_step_surrogate(self.state, batch)
                loss_sum += loss
                n += 1
            train_loss = loss_sum / n

            if do_eval:
                n = 0
                loss_sum = 0
                for batch in batchify(test_ds, batch_size):
                    loss_sum += _train_step_surrogate(self.state, batch, update=False)
                    n += 1
                test_loss = loss_sum / n

            print(f"Surrogate train loss {train_loss:.4f}", end="")
            if do_eval:
                print(f"; test loss {test_loss:.4f}", end="")
            if epoch < epochs - 1:
                print(end="\r")
            else:
                print()

        self.trained = True

    def predict(self, x, *, batch_size: int = 1024) -> np.ndarray:
        assert self.network is not None
        assert self.state is not None
        assert self.input_shape is not None
        if not self.trained:
            raise ValueError(
                "Cannot make predications as the network has not been trained."
            )
        dataset = dict(input=x)
        y = []
        variables = dict(params=self.state.params, batch_stats=self.state.batch_stats)
        for batch in batchify(dataset, batch_size):
            y.append(_predict_batch_surrogate(batch, variables, self.network.apply))
        return np.concatenate(y, axis=1)


def beta_loss(*, alpha, beta, y):
    y = jnp.where(y < EPSILON, EPSILON, y)
    y = jnp.where(y > 1 - EPSILON, 1 - EPSILON, y)
    loss = -jax.scipy.stats.beta.logpdf(y, alpha, beta)
    return loss


# TODO: The beta distribution seems unnecessary. Try this?
# def l2_loss(*, alpha, beta, y):
#    p = alpha / (alpha + beta)
#    loss = jnp.linalg.norm(p - y, axis=-1)
#    return loss


@functools.partial(jax.jit, static_argnums=(2,))
def _train_step_surrogate(state, batch, update=True):
    """Train for a single step."""

    def loss_fn(params):
        (alpha, beta), new_state = state.apply_fn(
            dict(params=params, batch_stats=state.batch_stats),
            batch["input"],
            mutable=["batch_stats"] if update else [],
            train=update,
        )
        loss = jnp.mean(
            batch["weights"] * beta_loss(alpha=alpha, beta=beta, y=batch["output"])
        )
        return loss, new_state

    if update:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, new_state), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads, batch_stats=new_state["batch_stats"])
        return state, loss
    else:
        loss, _ = loss_fn(state.params)
        return loss


@functools.partial(jax.jit, static_argnums=(2,))
def _predict_batch_surrogate(batch, variables, apply_func):
    """Make predictions on a batch."""
    alpha, beta = apply_func(variables, batch["input"], train=False)
    return alpha, beta
