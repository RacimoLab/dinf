import dataclasses
import functools
import typing

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
    """

    empirical_func: typing.Callable
    generator_func: typing.Callable
    parameters: Parameters
    feature_shape: typing.Any

    def __post_init__(self):
        # Transform generator_func from a function accepting arbitrary kwargs
        # (which limits user error) into a function accepting a sequence of
        # args (which is easier to pass to zeus-mcmc).
        f = self.generator_func
        self.generator_func = functools.update_wrapper(
            functools.partial(_sim_shim, func=f, keys=tuple(self.parameters)), f
        )
        self._orig_generator_func = f
