from __future__ import annotations
import collections
import dataclasses
import functools
import logging
import pathlib
from typing import Callable, Tuple

from flax import linen as nn
import flax.training.train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rich.text

from .misc import (
    Pytree,
    pytree_equal,
    pytree_shape,
    pytree_cons,
    pytree_cdr,
    leading_dim_size,
    is_tuple,
)

logger = logging.getLogger(__name__)


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
            batch = jax.tree_util.tree_map(lambda x: x[indices[i:j], ...], dataset)
        else:
            batch = jax.tree_util.tree_map(lambda x: x[i:j, ...], dataset)
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
        for input_feature in jax.tree_util.tree_leaves(inputs):
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
        for input_feature in jax.tree_util.tree_leaves(inputs):
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
class Discriminator:
    """
    Wrapper for the discriminator neural network that classifies genotype matrices.
    """

    input_shape: Pytree | None = dataclasses.field(init=False, default=None)
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
    format_version: int = 4
    """Version number for the serialised network on disk."""

    _inited: bool = False

    def __post_init__(self):
        if self.network is None:
            self.network = ExchangeableCNN()

        if self.state is None:
            self.state = TrainState.create(
                apply_fn=self.network.apply,
                tx=optax.chain(optax.clip(1.0), optax.adam(learning_rate=0.001)),
                params={},
                batch_stats={},
            )

    def _init(self, input_shape, *, seed: int):
        """
        Initialise the neural network with the given input shape.

        :param input_shape:
            The shape of the input to the neural network.
        :param seed:
            Seed for the jax random number generator.
        """
        assert self.network is not None
        assert self.state is not None
        assert self.input_shape is None

        # Set the leading (batch) dimension to 1.
        self.input_shape = pytree_cons(1, pytree_cdr(input_shape))

        # Sanity checks.
        if not jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda x: np.shape(x) == (4,) and x[1] >= 2 and x[2] >= 4 and x[3] <= 4,
                self.input_shape,
                is_leaf=is_tuple,
            )
        ):
            raise ValueError(
                "Input features must each have shape (b, n, m, c), where "
                "b is the number of data instances in the batch,"
                "n >= 2 is the number of (pseudo)haplotypes, "
                "m >= 4 is the length of the (pseudo)haplotypes, "
                "and c <= 4 is the number of channels.\n"
                f"input_shape={self.input_shape}"
            )

        logger.info("initialising network with shape %s", self.input_shape)

        @jax.jit
        def init(*args):
            return self.network.init(*args, train=False)

        dummy_input = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x, dtype=np.float32), self.input_shape, is_leaf=is_tuple
        )
        key = jax.random.PRNGKey(seed)
        variables = init(key, dummy_input)

        self.state = TrainState.create(
            apply_fn=self.state.apply_fn,
            tx=self.state.tx,
            params=variables["params"],
            batch_stats=variables.get("batch_stats", {}),
        )
        self._inited = True

        logger.debug("%s", self)

    @classmethod
    def from_file(cls, filename: str | pathlib.Path, /, *, network=None, state=None):
        """
        Load neural network weights from the given file.

        :param filename:
            The filename of the saved model.
        :param network:
            The flax neural network.
        :param state:
            The flax train state.
        :return:
            A new Discriminator object.
        """
        logger.info("%s: loading network weights", filename)
        with open(filename, "rb") as f:
            data = flax.serialization.msgpack_restore(f.read())
        discr = cls.fromdict(
            data, network=network, state=state, _err_prefix=f"{filename}: "
        )
        logger.debug("%s", discr)
        return discr

    def to_file(self, filename: str | pathlib.Path, /) -> None:
        """
        Save neural network to the given file.

        :param filename: The path where the model will be saved.
        """
        logger.debug("%s: saving network weights", filename)
        data = self.asdict()
        with open(filename, "wb") as f:
            f.write(flax.serialization.msgpack_serialize(data))

    @classmethod
    def fromdict(
        cls, data: dict, /, *, network=None, state=None, _err_prefix: str = ""
    ):
        """
        Load neural network weights from the given dict.

        :param data:
            The dict from which the network weights will be loaded.
        :param network:
            The flax neural network.
        :return:
            A new Discriminator object.
        """
        format_version = data.get("format_version")
        if format_version is not None and format_version != cls.format_version:
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

        discr = cls(network=network, state=state)
        assert discr.network is not None
        assert discr.state is not None

        discr.input_shape = jax.tree_util.tree_map(
            tuple,
            data["input_shape"],
            is_leaf=lambda x: isinstance(x, list),
        )
        discr.state = TrainState.create(
            apply_fn=discr.state.apply_fn,
            tx=discr.state.tx,
            params=data["state"]["params"],
            batch_stats=data["state"].get("batch_stats", {}),
        )
        discr.trained = data["trained"]
        discr.metrics = collections.defaultdict(list, **data["metrics"])
        discr._inited = True
        return discr

    def asdict(self) -> dict:
        input_shape = jax.tree_util.tree_map(list, self.input_shape, is_leaf=is_tuple)
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
        """Return a summary of the neural network."""
        # XXX: The order of layers in the CNN are lost because of
        # https://github.com/google/jax/issues/4085

        a = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x, dtype=np.float32), self.input_shape, is_leaf=is_tuple
        )
        return self.network.tabulate(jax.random.PRNGKey(0), a, train=False)

    def __str__(self):
        return str(rich.text.Text.from_ansi(self.summary()))

    def fit(
        self,
        *,
        train_x,
        train_y,
        val_x=None,
        val_y=None,
        batch_size: int = 64,
        epochs: int = 1,
        ss: np.random.SeedSequence,
        reset_metrics: bool = False,
        entropy_regularisation: bool = False,
        callbacks: dict | None = None,
    ):
        """
        Fit discriminator to training data.

        :param train_x:
            Training data.
        :param train_y:
            Labels for training data (zeros and ones).
        :param val_x:
            Validation data.
        :param val_y:
            Labels for validation data (zeros and ones).
        :param batch_size:
            Size of minibatch for gradient update step.
        :param epochs:
            The number of full passes over the training data.
        :param numpy.random.SeedSequence ss:
            Numpy random number seed sequence.
        :param reset_metrics:
            If true, remove loss/accuracy metrics from previous calls to
            fit() (if any). If false, loss/accuracy metrics will be appended
            to the existing metrics.
        """
        if callbacks is None:
            callbacks = {}
        assert all(k in ("train_batch", "test_batch", "epoch") for k in callbacks)

        if leading_dim_size(train_x) != leading_dim_size(train_y):
            raise ValueError(
                "Leading dimensions of train_x and train_y must be the same.\n"
                f"train_x={pytree_shape(train_x)}\n"
                f"train_y={pytree_shape(train_y)}"
            )
        if self.input_shape is not None and not pytree_equal(
            *map(pytree_cdr, [self.input_shape, pytree_shape(train_x)])
        ):
            raise ValueError(
                "Trailing dimensions of train_x must match network's input_shape.\n"
                f"input_shape={self.input_shape}\n"
                f"train_x={pytree_shape(train_x)}"
            )

        if (val_x is None and val_y is not None) or (
            val_x is not None and val_y is None
        ):
            raise ValueError("Must specify both val_x and val_y or neither.")

        if val_x is not None:
            if leading_dim_size(val_x) != leading_dim_size(val_y):
                raise ValueError(
                    "Leading dimensions of val_x and val_y must be the same.\n"
                    f"val_x={pytree_shape(val_x)}\n"
                    f"val_y={pytree_shape(val_y)}"
                )

            if not pytree_equal(
                *map(pytree_cdr, [pytree_shape(train_x), pytree_shape(val_x)])
            ):
                raise ValueError(
                    "Trailing dimensions of val_x must match train_x.\n"
                    f"train_x={pytree_shape(train_x)}\n"
                    f"val_x={pytree_shape(val_x)}"
                )

        def running_metrics(n, batch_size, current_metrics, metrics):
            new_metrics = jax.tree_util.tree_map(
                lambda a, b: a + batch_size * b, current_metrics, metrics
            )
            return n + batch_size, new_metrics

        def train_epoch(state, train_ds, batch_size, epoch, dropout_rng, rng):
            """Train for a single epoch."""

            metrics_sum = dict(loss=0, accuracy=0)
            n = 0
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
                loss = metrics_sum["loss"] / n
                accuracy = metrics_sum["accuracy"] / n
                if (cb_train_batch := callbacks.get("train_batch")) is not None:
                    cb_train_batch(n, loss, accuracy)

            return loss, accuracy, state

        def eval_model(state, test_ds, batch_size, rng):
            metrics_sum = dict(loss=0, accuracy=0)
            n = 0
            for batch in batchify(test_ds, batch_size, random=True, rng=rng):
                batch_metrics = _eval_step(state, batch)
                actual_batch_size = leading_dim_size(batch["input"])
                n, metrics_sum = running_metrics(
                    n, actual_batch_size, metrics_sum, batch_metrics
                )
                loss = metrics_sum["loss"] / n
                accuracy = metrics_sum["accuracy"] / n
                if (cb_test_batch := callbacks.get("test_batch")) is not None:
                    cb_test_batch(n, loss, accuracy)

            return loss, accuracy

        ss_jax, ss_train, ss_eval = ss.spawn(3)
        seed_init, seed_dropout = ss_jax.generate_state(2)
        rng_train, rng_eval = (np.random.default_rng(sj) for sj in (ss_train, ss_eval))

        if not self._inited:
            self._init(pytree_shape(train_x), seed=seed_init)

        assert self.network is not None
        assert self.state is not None
        assert self.input_shape is not None
        assert self._inited

        train_ds = dict(input=train_x, output=train_y)
        test_ds = dict(input=val_x, output=val_y)
        do_eval = val_x is not None

        if reset_metrics:
            self.metrics = collections.defaultdict(list)

        dropout_rng1 = jax.random.PRNGKey(seed_dropout)

        if (cb_epoch := callbacks.get("epoch")) is not None:
            cb_epoch(0)

        for epoch in range(1, epochs + 1):
            dropout_rng1, dropout_rng2 = jax.random.split(dropout_rng1)
            train_loss, train_accuracy, self.state = train_epoch(
                self.state, train_ds, batch_size, epoch, dropout_rng2, rng=rng_train
            )
            self.metrics["train_loss"].append(train_loss)
            self.metrics["train_accuracy"].append(train_accuracy)

            if do_eval:
                test_loss, test_accuracy = eval_model(
                    self.state, test_ds, batch_size, rng=rng_eval
                )
                self.metrics["test_loss"].append(test_loss)
                self.metrics["test_accuracy"].append(test_accuracy)

            if (cb_epoch := callbacks.get("epoch")) is not None:
                cb_epoch(epoch)

        self.trained = True

        # Return the metrics from the last epoch of training.
        metrics_conclusion = {k: v[-1] for k, v in self.metrics.items()}
        return metrics_conclusion

    def predict(
        self, x, *, batch_size: int = 1024, callbacks: dict | None = None
    ) -> np.ndarray:
        """
        Make predictions about data using a pre-fitted neural network.

        :param x: The data instances about which to make predictions.
        :param batch_size: Size of data batches for prediction.
        :return: A vector of predictions, one for each input instance.
        """
        if not self.trained:
            raise ValueError(
                "Cannot make predications as the discriminator has not been trained."
            )
        assert self.network is not None
        assert self.state is not None
        assert self.input_shape is not None
        assert leading_dim_size(x) > 0
        if not pytree_equal(*map(pytree_cdr, [self.input_shape, pytree_shape(x)])):
            raise ValueError(
                "Trailing dimensions of x must match input_shape.\n"
                f"input_shape={self.input_shape}\n"
                f"x={pytree_shape(x)}"
            )

        if callbacks is None:
            callbacks = {}
        assert all(k in ("batch",) for k in callbacks)

        dataset = dict(input=x)
        y = []
        n = 0
        variables = dict(params=self.state.params, batch_stats=self.state.batch_stats)
        for batch in batchify(dataset, batch_size):
            y.append(_predict_batch(batch, variables, self.network.apply))
            if (cb_batch := callbacks.get("batch")) is not None:
                actual_batch_size = leading_dim_size(batch)
                n += actual_batch_size
                cb_batch(n)
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
