from __future__ import annotations
import collections.abc
import copy
import dataclasses

import numpy as np


def logit(x):
    # Warning: divide by zero at x=0.
    return np.log(x / (1 - x))


def expit(x):
    # Warning: overflow when x is negative and very large
    return 1 / (1 + np.exp(-x))


@dataclasses.dataclass
class Param:
    """
    An inferrable parameter with ``Uniform(low, high)`` prior distribution.

    :param low:
        The lower bound for the parameter.
    :param high:
        The uppper bound for the parameter.
    :param truth:
        The true value of the parameter. Set to None if not known
        (e.g. for parameters corresponding to an empirical dataset).
        If the value is not None, this value may be used as the
        truth in a simulation study.
    """

    low: float
    """Parameter's lower bound."""

    high: float
    """Parameter's upper bound."""

    truth: float | None = None
    """The true value of the parameter. May be None."""

    name: str | None = None
    """Name of the parameter."""

    def __post_init__(self):
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError("Bounds must be finite.")
        if self.truth is not None and not self.bounds_contain(self.truth):
            raise ValueError(
                f"truth={self.truth} not in bounds [{self.low}, {self.high}]"
            )

    def draw_prior(self, size: int, /, rng: np.random.Generator) -> np.ndarray:
        """
        A random sample from the ``Uniform(low, high)`` prior distribution.

        :param size:
            The sample size.
        :param numpy.random.Generator rng:
            The numpy random number generator.
        :return:
            A numpy array of parameter values.
        """
        return rng.uniform(low=self.low, high=self.high, size=size)

    def bounds_contain(self, x: np.ndarray, /) -> np.ndarray:
        """
        Test if values are within in the parameter bounds.

        :param x:
            The values that will be bounds checked.
            May be a single number or an array.
        :return:
            A boolean value, or array of booleans with the same length as
            the input. ret[j] is True if x[j] is in bounds, False otherwise.
        """
        x = np.atleast_1d(x)
        return np.logical_and(self.low <= x, x <= self.high)

    @np.errstate(divide="ignore")
    def transform(self, x: np.ndarray, /) -> np.ndarray:
        """
        Transform bounded values on [low, high] to unbounded values on [-inf, inf].

        Performs a logit transformation.

        :param x:
            The values to be transformed.
        :return:
            Transformed parameter values.
        """
        x = np.atleast_1d(x)
        return logit((x - self.low) / (self.high - self.low))

    @np.errstate(over="ignore")
    def itransform(self, x: np.ndarray, /) -> np.ndarray:
        """
        Transform unbounded values on [-inf, inf] to bounded values on [low, high].

        Performs an inverse logit transformation (aka expit, aka sigmoid).

        :param x:
            The values to be transformed.
        :return:
            Transformed parameter values.
        """
        x = np.atleast_1d(x)
        return self.low + expit(x) * (self.high - self.low)

    def truncate(self, x: np.ndarray, /) -> np.ndarray:
        """
        Truncate values that are out of bounds.

        :param x:
            The values to be truncated.
        :return:
            Truncated parameter values.
        """
        return np.clip(x, self.low, self.high)

    def reflect(self, x: np.ndarray, /) -> np.ndarray:
        """
        Reflect values that are out of bounds by the amount they are out.

        As reflecting does not gaurantee values will be within the bounds,
        values are first truncated to (2*low - high, 2*high - low),
        then reflected. For example, with bounds low=0, high=10,
        a value of -11 will be truncated to -10, then reflected to attain
        a final value of 10.

        :param x:
            The values to be reflected.
        :return:
            Reflected parameter values.
        """
        # Truncate values that can't be reflected into the domain.
        width = self.high - self.low
        x = np.clip(x, self.low - width, self.high + width)

        idx_lo = np.where(x < self.low)
        x[idx_lo] += 2 * (self.low - x[idx_lo])
        idx_hi = np.where(x > self.high)
        x[idx_hi] -= 2 * (x[idx_hi] - self.high)
        return x


class Parameters(collections.abc.Mapping):
    """
    An ordered collection of inferrable parameters.

    A collection of parameters is built by passing :class:`Param` objects
    to the constructor by keyword. The keyword will be used as the
    :attr:`name <Param.name>` for the given parameter. This class implements
    Python's :term:`python:mapping` protocol.

    .. code::

        import dinf

        # Parameters named "a", "b", and "c".
        parameters = dinf.Parameters(
            a=dinf.Param(low=0, high=1),
            b=dinf.Param(low=5, high=100),
            c=dinf.Param(low=1e-6, high=1e-3),
        )

        # Iterate over parameters.
        assert len(parameters) == 3
        assert list(parameters) == ["a", "b", "c"]
        assert [p.high for p in parameters.values()] == [1, 100, 1e-3]

        # Lookup a parameter by name.
        a = parameters["a"]
        assert a.low == 0
        assert a.high == 1
        assert a.name == "a"
    """

    def __init__(self, **kwargs: Param):
        """
        :param kwargs:
            The :class:`Param` objects, named by keyword.
        """
        self._params = copy.deepcopy(kwargs)
        for k, v in self._params.items():
            if not isinstance(v, Param):
                raise TypeError(f"{k} must be a (sub)class of Param")
            if v.name is None:
                # Set the name.
                v.name = k
            if v.name != k:
                raise ValueError(f"Name mismatch for {k}=Param(..., name={v.name})")
            if v.name == "_Pr":
                # Used for the discriminator probabilites in npz files.
                raise ValueError(
                    "Parameter name '_Pr' is reserved by Dinf for internal use."
                )

    def __getitem__(self, key):
        return self._params[key]

    def __iter__(self):
        # yield the parameter names
        yield from self._params

    def __len__(self):
        return len(self._params)

    def draw_prior(self, size: int, /, rng: np.random.Generator) -> np.ndarray:
        """
        A random sample of parameter values.

        :param size:
            The sample size.
        :param numpy.random.Generator rng:
            The numpy random number generator.
        :return:
            A 2d numpy array of parameter values, where ret[j][k] is the
            j'th draw for the k'th parameter.
        """
        return np.transpose([p.draw_prior(size, rng) for p in self.values()])

    def bounds_contain(self, xs: np.ndarray, /) -> np.ndarray:
        """
        Test if values are within in the parameter bounds.

        :param xs:
            A 2d numpy array, where xs[j][k] is the j'th sample for the
            k'th parameter.
        :return:
            A 1d numpy array of boolean values, where ret[j] is True if the
            j'th sample is within the parameter space, and False otherwise.
        """
        xs = np.atleast_2d(xs)
        assert xs.shape[-1] == len(self)
        ret = np.all(
            [p.bounds_contain(xs[..., k]) for k, p in enumerate(self.values())],
            axis=0,
        )
        return ret

    def transform(self, xs: np.ndarray, /) -> np.ndarray:
        """
        Transform values bounded by [param.low, param.high] to [-inf, inf].

        See :meth:`.itransform` for the inverse transformation.

        :param xs:
            The values to be transformed.
        :return:
            Transformed parameter values.
        """
        xs = np.atleast_2d(xs)
        assert xs.shape[-1] == len(self)
        return np.transpose(
            [p.transform(xs[:, k]) for k, p in enumerate(self.values())]
        )

    def itransform(self, xs: np.ndarray, /) -> np.ndarray:
        """
        Transform values on [-inf, inf] to be bounded by [param.low, param.high].

        Performs the inverse of :meth:`.transform`.

        :param xs:
            The values to be transformed.
        :return:
            Transformed parameter values.
        """
        xs = np.atleast_2d(xs)
        assert xs.shape[-1] == len(self)
        return np.transpose(
            [p.itransform(xs[:, k]) for k, p in enumerate(self.values())]
        )

    def truncate(self, xs: np.ndarray, /) -> np.ndarray:
        """
        Truncate values that are out of bounds.

        :param xs:
            The values to be truncated.
        :return:
            Truncated parameter values.
        """
        xs = np.atleast_2d(xs)
        assert xs.shape[-1] == len(self)
        return np.transpose([p.truncate(xs[:, k]) for k, p in enumerate(self.values())])

    def reflect(self, xs: np.ndarray, /) -> np.ndarray:
        """
        Reflect values that are out of bounds by the amount they are out.

        As reflecting does not gaurantee values will be within the bounds,
        values are first truncated to (2*low - high, 2*high - low),
        then reflected. For example, with bounds low=0, high=10,
        a value of -11 will be truncated to -10, then reflected to attain
        a final value of 10.

        :param xs:
            The values to be reflected.
        :return:
            Reflected parameter values.
        """
        xs = np.atleast_2d(xs)
        assert xs.shape[-1] == len(self)
        return np.transpose([p.reflect(xs[:, k]) for k, p in enumerate(self.values())])
