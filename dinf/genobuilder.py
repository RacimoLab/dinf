from __future__ import annotations
import collections
import dataclasses
import functools
import importlib
import pathlib
import sys
from typing import Callable, Tuple

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
class Genobuilder:
    """
    User-defined parts of the GAN.

    :param target_func:
        A function that samples a feature from the target distribution.
        Typically this would be drawn from empirical data, but simulations with
        fixed parameter values can be used instead.

        This function accepts a single (positional) argument, an integer seed,
        and returns an n-dimensional feature array, whose shape is given in
        :attr:`.feature_shape`. The signature of the function should be:

        .. code::

            def target_func(seed: int) -> np.ndarray:
                ...

    :param generator_func:
        A function that generates features using the given parameter values.

        The first (positional) argument to the function is an integer seed.
        The subsequent arguments correspond to values for the
        :attr:`.parameters`, and will be passed to the generator
        function by name (as keyword arguments). This function returns
        an n-dimensional feature array, whose shape is given in
        :attr:`.feature_shape`. The signature of the function will
        follow the pattern below, where ``p0``, ``p1``, etc. are the
        names of the parameters in :attr:`.parameters`.

        .. code::

            def generator_func(seed: int, *, p0: float, p1: float, ...) -> np.ndarray:
                ...

        For generator functions accepting large numbers of parameters,
        the following pattern may be preferred instead.

        .. code::

            def generator_func(seed: int, **kwargs)-> np.ndarray:
                assert kwargs.keys() == parameters.keys()
                # do something with p0
                do_something(kwargs["p0"])
                # do something with p1
                do_something_else(kwargs["p1"])

    :param parameters:
        A collection of inferrable parameters, one for each keyword argument
        provided to the :attr:`.generator_func`.

    :param feature_shape:
        The shape of the n-dimensional arrays produced by :attr:`.target_func`
        and :attr:`.generator_func`.

    :param discriminator_network:
        A flax neural network module. If not specified, :class:`ExchangeableCNN`
        will be used.
    """

    target_func: Callable  # Callable[[int], np.ndarray]
    """
    Function that samples a feature from the target distribution.
    """

    generator_func: Callable  # Callable[[int, ...], np.ndarray]
    """
    Function that generates features using the given parameter values.
    """

    parameters: Parameters
    """
    The inferrable parameters.
    """

    feature_shape: collections.abc.Mapping[str, Tuple]
    """
    Shape of the n-dimensional arrays produced by :attr:`.target_func`
    and :attr:`.generator_func`.
    """

    discriminator_network: nn.Module | None = None
    """
    A flax neural network module. May be ``None``.
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
                    "For a simulation study (with genobuilder.target_func=None), "
                    "all parameters must have `truth' values defined.\n"
                    f"Truth values missing for: {', '.join(truth_missing)}."
                )
            self.target_func = functools.partial(self.generator_func, **theta_truth)

        # Transform generator_func from a function accepting arbitrary kwargs
        # (which limits user error) into a function accepting a sequence of
        # args (which is easier to pass to the mcmc).
        f = self.generator_func
        self.generator_func = functools.update_wrapper(
            functools.partial(_sim_shim, func=f, keys=tuple(self.parameters)), f
        )
        self._orig_generator_func = f
        self._filename = None

    def check(self, seed=None):
        """
        Basic health checks: draw parameters and call the functions.

        We don't do this when initialising the Genobuilder, because calling
        :meth:`.generator_func()` is potentially time consuming, which could
        lead to annoying delays for the command line interface.
        """
        if seed is None:
            seed = 1234
        rng = np.random.default_rng(seed)
        thetas = self.parameters.draw_prior(num_replicates=5, rng=rng)
        if thetas.shape != (5, len(self.parameters)):
            raise ValueError(
                "parameters.draw_prior(num_replicates=5) produced output with shape "
                f"{thetas.shape}, expected shape {(5, len(self.parameters))}."
            )

        x_g = self.generator_func((rng.integers(low=0, high=2**31), thetas[0]))
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
    def _from_file(filename: str | pathlib.Path) -> Genobuilder:
        """
        Load the symbol "genobuilder" from a file.

        https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        """
        module_name = "_dinf_user_module"
        spec = importlib.util.spec_from_file_location(module_name, filename)
        if spec is None:
            raise ImportError(
                f"Could not load spec for `{filename}'. "
                "Check that the file exists and has a .py extension."
            )
        # Pacify mypy.
        assert isinstance(spec.loader, importlib.abc.Loader)

        user_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = user_module
        spec.loader.exec_module(user_module)

        genobuilder = getattr(user_module, "genobuilder", None)
        if genobuilder is None:
            raise AttributeError(f"genobuilder not found in {filename}")
        if not isinstance(genobuilder, Genobuilder):
            raise TypeError(f"{filename}: genobuilder is not a dinf.Genobuilder object")
        genobuilder._filename = filename
        return genobuilder
