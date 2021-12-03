import dataclasses
import functools
from typing import Callable, Tuple

import numpy as np

from .parameters import Parameters


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

    feature_shape: Tuple
    """
    Shape of the n-dimensional arrays produced by :attr:`.target_func`
    and :attr:`.generator_func`.
    """

    def __post_init__(self):
        if len(self.parameters) == 0:
            raise ValueError("Must define one or more parameters")
        # Transform generator_func from a function accepting arbitrary kwargs
        # (which limits user error) into a function accepting a sequence of
        # args (which is easier to pass to the mcmc).
        f = self.generator_func
        self.generator_func = functools.update_wrapper(
            functools.partial(_sim_shim, func=f, keys=tuple(self.parameters)), f
        )
        self._orig_generator_func = f

    def check(self, seed=1234):
        """
        Basic health checks: draw parameters and call the functions.
        """
        rng = np.random.default_rng(seed)
        thetas = self.parameters.draw(num_replicates=5, rng=rng)
        assert thetas.shape == (5, len(self.parameters))
        x_g = self.generator_func((rng.integers(low=0, high=2 ** 31), thetas[0]))
        if not np.array_equal(x_g.shape, self.feature_shape):
            raise ValueError(
                f"Output of generator_func has shape {x_g.shape}, "
                f"but feature_shape is {self.feature_shape}."
            )
        x_t = self.target_func(rng.integers(low=0, high=2 ** 31))
        if not np.array_equal(x_t.shape, self.feature_shape):
            raise ValueError(
                f"Output of target_func has shape {x_t.shape}, "
                f"but feature_shape is {self.feature_shape}."
            )
