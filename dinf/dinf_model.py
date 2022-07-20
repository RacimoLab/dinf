from __future__ import annotations
import dataclasses
import functools
import importlib
import pathlib
import sys
from typing import Callable, Dict

from flax import linen as nn
import numpy as np

from .parameters import Parameters
from .misc import tree_equal, tree_shape


def _sim_shim(args, *, func, keys):
    """
    Wrapper that takes an argument list, and calls func with keyword args.
    """
    seed, func_args = args
    assert len(keys) == len(func_args)
    kwargs = dict(zip(keys, func_args))
    return func(seed, **kwargs)


@dataclasses.dataclass
class DinfModel:
    """
    A container that describes the components of a Dinf model.

    Constructing a Dinf model requires:

     - specifying the inferrable ``parameters``,
     - a ``generator_func`` function that accepts concrete parameter values,
       produces data under some simulation model, and returns a feature matrix
       (or matrices),
     - a ``target_func`` function that samples from the target dataset,
       and returns a feature matrix (or matrices),
     - the ``feature_shape``, which is the shape of the feature matrix
       (or matrices) that are returned by both the ``generator_func``
       and the ``target_func``.

    .. code::

        import dinf
        import msprime

        parameters = dinf.Parameters(...)
        features = dinf.BinnedHaplotypeMatrix(...)
        vcfs = dinf.BagOfVcf(...)

        def generator(seed, ...):
            ...
            # E.g. simulate using msprime to get a succinct tree sequence (ts).
            ts = msprime.sim_ancestry(...)
            mts = msprime.sim_mutations(ts, ...)
            return features.from_ts(mts)

        def target(seed):
            ...
            return features.from_vcf(vcfs, ...)

        dinf_model = dinf.DinfModel(
            parameters=parameters,
            generator_func=generator,
            target_func=target,
            feature_shape=features.shape,
        )

    :param parameters:
        A collection of inferrable parameters, one for each keyword argument
        provided to the :attr:`.generator_func`.

    :param generator_func:
        A function that returns features for concrete parameter values.

        The first (positional) argument to the function is an integer seed,
        that may be used to seed a random number generator.
        The subsequent (keyword) arguments correspond to concrete values for
        the :attr:`.parameters`.
        The return value is either:

         - a feature matrix, i.e. an n-dimensional numpy array, or
         - multiple feature matrices, i.e. a dictionary of n-dimensional
           numpy arrays, where the dictionary keys are arbitrary labels
           such as population names.

        The signature of the function will follow the patterns below,
        where ``p0``, ``p1``, etc. are the names of the parameters in
        :attr:`.parameters`.
        Type annotations are provided here for clarity, but are not required
        in user-defined functions. In function signatures,
        `positional-only parameters <https://peps.python.org/pep-0570/>`_
        preceed a ``/`` and
        `keyword-only parameters <https://peps.python.org/pep-3102/>`_
        follow a ``*``.

        .. code::

            parameters = dinf.Parameters(
                p0=dinf.Param(...),
                p1=dinf.Param(...),
                ...
            )

            # Signature for returning a single feature matrix.
            def generator_func1(
                seed: int, /, *, p0: float, p1: float, ...
            ) -> np.ndarray:
                ...

            # Signature for returning multiple feature matrices.
            def generator_func2(
                seed: int, /, *, p0: float, p1: float, ...
            ) -> dict[str, np.ndarray]:
                ...

            # For generator functions accepting large numbers of parameters,
            # the following pattern using ``**kwargs`` may be preferred.
            def generator_func3(
                seed: int, /, **kwargs: float
            )-> dict[str, np.ndarray]:
                assert kwargs.keys() == parameters.keys()
                # do something with p0
                do_something(kwargs["p0"])
                # do something with p1
                do_something_else(kwargs["p1"])
                ...

    :param target_func:
        A function that returns features sampled from the target dataset.

        If ``None`` (which must be specified explicitly),
        the :attr:`.generator_func` will be used to simulate
        the target dataset using each parameter's :attr:`truth <Param.truth>`
        value.

        The function takes a single (positional) argument, an integer seed,
        that may be used to seed a random number generator.
        The return value must match the return value of :attr:`generator_func`.

    :param feature_shape:
        The shape of the feature, or features, returned by
        :attr:`.generator_func` and :attr:`.target_func`.

         - When a single feature matrix is returned, i.e. an n-dimensional
           numpy array, the ``feature_shape`` is a tuple of array dimensions
           (c.f. :attr:`numpy.ndarray.shape`).
         - When multiple feature matrices are returned,
           i.e. a dictionary of n-dimensional numpy arrays,
           the ``feature_shape`` is also a dictionary, which maps
           feature labels to the shape of the given feature matrix.

    :param discriminator_network:
        A :doc:`flax <flax:index>` neural network.
        If not specified, :class:`ExchangeableCNN` will be used.
    """

    parameters: Parameters
    """
    The inferrable parameters.
    """

    generator_func: Callable
    """
    Function that simulates features using concrete parameter values.
    """

    generator_func_v: Callable = dataclasses.field(init=False)
    """
    Wrapper for ``generator_func`` that accepts a single argument containing
    the seed and a vector of parameter values (as opposed to keyword arguments).
    The signature is ``generator_func_v(a: Tuple[int, v: np.ndarray])``,
    where the argument is a 2-tuple of ``(seed, vector)``.
    """

    target_func: Callable | None
    """
    Function that samples features from the target distribution.
    """

    feature_shape: tuple | Dict[str, tuple]
    """
    Shape of the feature, or features, produced by
    :attr:`.generator_func` and :attr:`.target_func`.
    """

    discriminator_network: nn.Module | None = None
    """
    A :doc:`flax <flax:index>` neural network. May be ``None``.
    """

    filename: pathlib.Path | None = dataclasses.field(init=False, default=None)
    """
    Path to the file from which the model was loaded (if any).
    May be ``None``.
    """

    def __post_init__(self):
        if len(self.parameters) == 0:
            raise ValueError("Must define one or more parameters")

        if self.target_func is None:
            # Use the generator function with the parameter's "truth" values
            # as the target function. I.e. do a simulation study.
            theta_truth = {k: p.truth for k, p in self.parameters.items()}
            truth_missing = [k for k, truth in theta_truth.items() if truth is None]
            if len(truth_missing) > 0:
                raise ValueError(
                    "For a simulation-only model "
                    "(with dinf_model.target_func=None), "
                    "all parameters must have `truth' values defined.\n"
                    f"Truth values missing for: {', '.join(truth_missing)}."
                )
            self.target_func = functools.partial(self.generator_func, **theta_truth)

        # Transform generator_func from a function accepting arbitrary kwargs
        # (which limits user error) into a function accepting a sequence of
        # args (which is easier to pass to the mcmc).
        self.generator_func_v = functools.update_wrapper(
            functools.partial(
                _sim_shim, func=self.generator_func, keys=tuple(self.parameters)
            ),
            self.generator_func,
        )

    def check(self, seed=None):
        """
        Basic health checks: draw parameters and call the functions.

        We don't do this when initialising the DinfModel, because calling
        :attr:`.generator_func()` and :attr:`.target_func()` are potentially
        time consuming, which could lead to annoying delays for the command
        line interface.
        """
        if seed is None:
            seed = 1234
        rng = np.random.default_rng(seed)
        thetas = self.parameters.draw_prior(5, rng=rng)
        if thetas.shape != (5, len(self.parameters)):
            raise ValueError(
                "parameters.draw_prior(5) produced output with shape "
                f"{thetas.shape}, expected shape {(5, len(self.parameters))}."
            )

        x_g = self.generator_func_v((rng.integers(low=0, high=2**31), thetas[0]))
        if not tree_equal(tree_shape(x_g), self.feature_shape):
            raise ValueError(
                f"generator_func produced feature shape {tree_shape(x_g)}, "
                f"but feature_shape is {self.feature_shape}"
            )

        x_t = self.target_func(rng.integers(low=0, high=2**31))
        if not tree_equal(tree_shape(x_t), self.feature_shape):
            raise ValueError(
                f"target_func produced feature shape {tree_shape(x_t)}, "
                f"but feature_shape is {self.feature_shape}"
            )

    @staticmethod
    def from_file(filename: str | pathlib.Path) -> DinfModel:
        """
        Load the symbol "dinf_model" from a file.

        https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        """
        module_name = "_dinf_user_module"
        spec = importlib.util.spec_from_file_location(module_name, filename)
        if spec is None:
            raise ImportError(
                f"{filename}: couldn't load spec. "
                "Check that the file exists and has a .py extension."
            )
        # Pacify mypy.
        assert isinstance(spec.loader, importlib.abc.Loader)

        user_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = user_module
        spec.loader.exec_module(user_module)

        dinf_model = getattr(user_module, "dinf_model", None)
        if dinf_model is None:
            raise AttributeError(f"{filename}: variable 'dinf_model' not found")
        if not isinstance(dinf_model, DinfModel):
            raise TypeError(f"{filename}: dinf_model is not a dinf.DinfModel object")
        dinf_model.filename = pathlib.Path(filename)
        return dinf_model
