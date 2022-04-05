from __future__ import annotations
import dataclasses
import functools
import pathlib
import pickle
import sys
import time
from typing import Sequence

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


def batchify(dataset, batch_size):
    """Generate batch_size chunks of the dataset."""
    assert batch_size >= 1
    size = leading_dim_size(dataset)
    assert size >= 1

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
        activation = nn.elu

        combined = []
        for input_feature in jax.tree_leaves(inputs):
            x = norm()(input_feature)

            x = conv(features=32, strides=(1, 2))(x)
            x = activation(x)
            x = norm()(x)

            x = conv(features=64, strides=(1, 2))(x)
            x = activation(x)
            x = norm()(x)

            # collapse haplotypes
            x = Symmetric(axis=1)(x)

            x = conv(features=64)(x)
            x = activation(x)
            x = norm()(x)

            # collapse genomic bins
            x = Symmetric(axis=2)(x)
            combined.append(x)

        ys = jnp.concatenate(combined, axis=-1)
        y = nn.Dense(features=1)(ys)

        # flatten
        y = y.reshape((-1,))

        # We output logits on (-inf, inf), rather than a probability on [0, 1],
        # because the jax ecosystem provides better API support for working
        # with logits, e.g. loss functions in optax.
        # So remember to call jax.nn.sigmoid(x) on the output when
        # probabilities are needed.
        return y


@dataclasses.dataclass
class Network:
    """
    Base class for Discriminator and Surrogate networks.


    :ivar network: The flax neural network. This has an apply() method.
    :ivar input_shape: The shape of the input to the neural network.
    :ivar variables: A Pytree of the network parameters.
    :ivar train_metrics:
        A Pytree containing the loss/accuracy metrics obtained when training
        the network.
    """

    network: nn.Module
    input_shape: Pytree
    variables: Pytree
    train_metrics: Pytree | None = None
    state: TrainState | None = None
    trained: bool = False
    # Bump this after making internal changes.
    format_version: int = 2

    @classmethod
    def from_input_shape(
        cls,
        input_shape: Pytree,
        rng: np.random.Generator,
        network: nn.Module,
    ):
        """
        Build a neural network with the given input shape.

        :param input_shape:
            The shape of the input data for the network.
        :param numpy.random.Generator rng:
            The numpy random number generator.
        :param network:
            A flax neural network.
        """
        key = jax.random.PRNGKey(rng.integers(2**63))

        # Add leading batch dimension.
        input_shape = tree_cons(1, input_shape)
        dummy_input = jax.tree_map(
            lambda x: jnp.zeros(x, dtype=np.int8),
            input_shape,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        @jax.jit
        def init(*args):
            return network.init(*args, train=False)

        variables = init(key, dummy_input)
        return cls(network=network, variables=variables, input_shape=input_shape)

    @classmethod
    def from_file(cls, filename: str | pathlib.Path):
        """
        Load neural network from the given file.

        :param filename: The filename of the saved model.
        :return: The network object.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        if data.pop("format_version", -1) != cls.format_version:
            raise ValueError(
                f"{filename}: saved network is not compatible with this "
                "version of dinf. Either train a new network or use an "
                "older version of dinf."
            )
        expected_fields = set(map(lambda f: f.name, dataclasses.fields(cls)))
        expected_fields.remove("format_version")
        assert data.keys() == expected_fields
        return cls(**data)

    def to_file(self, filename) -> None:
        """
        Save neural network to the given file.

        :param filename: The path where the model will be saved.
        """
        data = dataclasses.asdict(self)
        data["state"] = None
        data["network"] = self.network  # asdict converts this to a dict
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def summary(self):
        """Print a summary of the neural network."""
        a = jax.tree_map(
            lambda x: jnp.zeros(x, dtype=np.int8),
            self.input_shape,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        _, state = self.network.apply(
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

        num_params = np.sum(
            jax.tree_leaves(
                jax.tree_map(lambda x: np.prod(x.shape), self.variables["params"])
            )
        )
        print("Total number of trainable parameters:", num_params)


@dataclasses.dataclass
class Discriminator(Network):
    """
    A discriminator network that classifies the origin of feature matrices.

    To instantiate, use the from_file() or from_input_shape() class methods.
    """

    @classmethod
    def from_input_shape(
        cls,
        input_shape: Pytree,
        rng: np.random.Generator,
        network: nn.Module = None,
    ) -> Discriminator:
        """
        Build a discriminator neural network with the given input shape.

        :param input_shape:
            The shape of the input data for the network. This is a dictionary
            that maps a label to a feature array. Each feature array has shape
            (n, m, c), where
            n >= 2 is the number of (pseudo)haplotypes,
            m >= 4 is the length of the (pseudo)haplotypes,
            and c <= 4 is the number of channels.
        :param numpy.random.Generator rng:
            The numpy random number generator.
        :param network:
            A flax neural network.
            If not specified, an exchangeable CNN will be used.
        """
        if network is None:
            network = ExchangeableCNN()

        # Sanity checks.
        if not jax.tree_util.tree_all(
            jax.tree_map(
                lambda x: np.shape(x) == (3,) and x[0] >= 2 and x[1] >= 4 and x[2] <= 4,
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

        return super().from_input_shape(input_shape, rng, network)

    def fit(
        self,
        *,
        train_x,
        train_y,
        val_x=None,
        val_y=None,
        batch_size: int = 64,
        epochs: int = 1,
        # TODO: tensorboard output
        # tensorboard_log_dir=None,
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
        :param reset_metrics:
            If true, remove loss/accuracy metrics from previous calls to
            fit() (if any). If false, loss/accuracy metrics will be appended
            to the existing metrics.
        """
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

        def train_epoch(state, train_ds, batch_size, epoch):
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
            for batch in batchify(train_ds, batch_size):
                state, batch_metrics = _train_step(
                    state, batch, entropy_regularisation=entropy_regularisation
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

        state = self.state
        if state is None:
            state = TrainState.create(
                apply_fn=self.network.apply,
                tx=optax.adam(learning_rate=0.001),
                params=self.variables["params"],
                batch_stats=self.variables.get("batch_stats", {}),
            )

        train_ds = dict(input=train_x, output=train_y)
        test_ds = dict(input=val_x, output=val_y)
        do_eval = val_x is not None

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

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy, state = train_epoch(
                state, train_ds, batch_size, epoch
            )
            self.train_metrics["train_loss"].append(train_loss)
            self.train_metrics["train_accuracy"].append(train_accuracy)

            if do_eval:
                test_loss, test_accuracy = eval_model(state, test_ds, batch_size)
                self.train_metrics["test_loss"].append(test_loss)
                self.train_metrics["test_accuracy"].append(test_accuracy)
            else:
                print()

        assert state is not None
        self.state = state
        self.trained = True
        self.variables = jax.tree_map(
            np.array,
            dict(
                params=jax.device_get(state.params),
                batch_stats=jax.device_get(state.batch_stats),
            ),
        )

        # Return the metrics from the last epoch of training.
        metrics_conclusion = {k: v[-1] for k, v in self.train_metrics.items()}
        return metrics_conclusion

    def predict(self, x, *, batch_size: int = 1024) -> np.ndarray:
        """
        Make predictions about data using a pre-fitted neural network.

        :param x: The data instances about which to make predictions.
        :param batch_size: Size of data batches for prediction.
        :return: A vector of predictions, one for each input instance.
        """

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
        for batch in batchify(dataset, batch_size):
            y.append(_predict_batch(batch, self.variables, self.network.apply))
        return np.concatenate(y)


##
# Jitted functions below are at the top level so they only get jitted once.


def binary_accuracy(*, logits, labels):
    """Accuracy of binary classifier, from logits."""
    p = jax.nn.sigmoid(logits)
    return jnp.mean(labels == (p > 0.5))


@functools.partial(jax.jit, static_argnums=(2,))
def _train_step(state, batch, entropy_regularisation=False):
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
    layers: Sequence[int] = (64, 64, 32)

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
class Surrogate(Network):
    """
    A surrogate network for the beta-estimation model of ALFI.

    This network predicts the output of the discriminator network
    from a set of dinf model parameters, thus bypassing the generator.
    Kim et al. 2020, https://arxiv.org/abs/2004.05803v1

    To instantiate, use the from_file() or from_input_shape() class methods.
    """

    @classmethod
    def from_input_shape(
        cls,
        input_shape: int,
        rng: np.random.Generator,
        network: nn.Module = None,
    ) -> Surrogate:
        """
        Build a surrogate neural network with the given input shape.

        :param input_shape:
            The shape of the input data for the network.
        :param numpy.random.Generator rng:
            The numpy random number generator.
        :param network:
            A neural network. If not specified, a simple MLP will be used.
        """
        if network is None:
            network = SurrogateMLP()
        key = jax.random.PRNGKey(rng.integers(2**63))

        assert input_shape >= 1

        # Add leading batch dimension.
        input_shape2 = (1, input_shape)
        dummy_input = jnp.zeros(input_shape2)

        @jax.jit
        def init(*args):
            return network.init(*args, train=False)

        variables = init(key, dummy_input)
        return cls(network=network, variables=variables, input_shape=input_shape2)

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
        state = self.state
        if state is None:
            state = TrainState.create(
                apply_fn=self.network.apply,
                tx=optax.adam(learning_rate=0.001),
                params=self.variables["params"],
                batch_stats=self.variables.get("batch_stats", {}),
            )

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
                state, loss = _train_step_surrogate(state, batch)
                loss_sum += loss
                n += 1
            train_loss = loss_sum / n

            if do_eval:
                n = 0
                loss_sum = 0
                for batch in batchify(test_ds, batch_size):
                    loss_sum += _train_step_surrogate(state, batch, update=False)
                    n += 1
                test_loss = loss_sum / n

            print(f"Surrogate train loss {train_loss:.4f}", end="")
            if do_eval:
                print(f"; test loss {test_loss:.4f}", end="")
            if epoch < epochs - 1:
                print(end="\r")
            else:
                print()

        assert state is not None
        self.state = state
        self.trained = True
        self.variables = jax.tree_map(
            np.array,
            dict(
                params=jax.device_get(state.params),
                batch_stats=jax.device_get(state.batch_stats),
            ),
        )

    def predict(self, x, *, batch_size: int = 1024) -> np.ndarray:
        if not self.trained:
            raise ValueError(
                "Cannot make predications as the network has not been trained."
            )
        dataset = dict(input=x)
        y = []
        for batch in batchify(dataset, batch_size):
            y.append(
                _predict_batch_surrogate(batch, self.variables, self.network.apply)
            )
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
