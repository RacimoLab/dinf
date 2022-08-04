from __future__ import annotations
import collections.abc
import copy
import dataclasses
import logging
from typing import Tuple

import numpy as np
import scipy

logger = logging.getLogger(__name__)


def _logit(x):
    # Warning: divide by zero at x=0.
    return np.log(x / (1 - x))


def _expit(x):
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

    def sample_prior(self, *, size: int, rng: np.random.Generator) -> np.ndarray:
        """
        Get a random sample from the ``Uniform(low, high)`` prior distribution.

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
        return _logit((x - self.low) / (self.high - self.low))

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
        return self.low + _expit(x) * (self.high - self.low)

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

    def bounds_contain(self, thetas: np.ndarray, /) -> np.ndarray:
        """
        Test if values are within in the parameter bounds.

        :param thetas:
            A 2d numpy array, where thetas[j][k] is the j'th sample for the
            k'th parameter.
        :return:
            A 1d numpy array of boolean values, where ret[j] is True if the
            j'th sample is within the parameter space, and False otherwise.
        """
        thetas = np.atleast_2d(thetas)
        assert thetas.shape[-1] == len(self)
        ret = np.all(
            [p.bounds_contain(thetas[..., k]) for k, p in enumerate(self.values())],
            axis=0,
        )
        return ret

    def sample_prior(self, *, size: int, rng: np.random.Generator) -> np.ndarray:
        """
        Get a random sample of parameter values from their prior distributions.

        :param size:
            The sample size.
        :param numpy.random.Generator rng:
            The numpy random number generator.
        :return:
            A 2d numpy array of parameter values, where ret[j][k] is the
            j'th draw for the k'th parameter.
        """
        return np.moveaxis(
            np.array([p.sample_prior(size=size, rng=rng) for p in self.values()]),
            0,
            -1,
        )

    def sample_kde(
        self,
        thetas: np.ndarray,
        /,
        *,
        probs: np.ndarray,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample from a smoothed set of weighted observations.

        Samples are drawn from ``thetas``, weighted by their probability.
        New points are drawn within a neighbourhood of the sampled thetas
        using a mulivariate normal whose covariance is calculated from the
        thetas. This is equivalent to sampling from a Gaussian KDE, but
        avoids doing an explicit density estimation.
        Values are sampled until they are within the parameter bounds.
        Scott's rule of thumb is used for bandwidth selection.

        :param thetas:
            Parameter values to sample from.
        :param probs:
            Discriminator predictions corresponding to the ``thetas``.
        :param size:
            Number of samples to draw.
        :param numpy.random.Generator rng:
            Numpy random generator.
        :return:
            The sampled values.
        """
        thetas = np.atleast_2d(thetas)
        assert len(thetas) > 1
        assert thetas.shape[-1] == len(self)
        probs = np.atleast_1d(probs)
        assert thetas.shape[0] == probs.shape[0]
        weights = probs / np.sum(probs)

        # Calculate bandwidth.
        _, d = thetas.shape
        neff = np.sum(probs) ** 2 / np.sum(probs**2)
        bw_scott = neff ** (-1.0 / (d + 4))  # bandwidth multiplier
        cov = bw_scott**2 * np.cov(thetas.T, aweights=weights)
        assert not np.any(np.isnan(cov))
        assert not np.any(np.isinf(cov))

        sample = np.empty((size, d))
        idx: slice | np.ndarray = slice(size)
        size_remaining = size
        draws = 0
        while size_remaining > 0:
            draws += size_remaining
            # Weighted sampling of points from the thetas.
            mean = rng.choice(thetas, size=size_remaining, replace=True, p=weights)
            # Disperse the sample with an MVN.
            disp = rng.multivariate_normal(np.zeros(d), cov, size=size_remaining)
            sample[idx] = mean + disp

            # Get indices of samples that are out of bounds.
            idx = np.nonzero(~self.bounds_contain(sample))[0]
            size_remaining = len(idx)

            if draws > 1000 * size:
                raise RuntimeError(
                    f"Failed to get {size} samples from KDE that are within the "
                    f"parameter bounds after {draws} draws. "
                )

        if draws > 10 * size:
            logger.warning(
                "Excessive KDE samples out of parameter bounds: %d of %d draws",
                draws - size,
                draws,
            )

        return sample

    def sample_ball(
        self,
        theta: np.ndarray,
        /,
        *,
        size: int,
        cov: np.ndarray = None,
        cov_factor: float = 1.0,
        rng: np.random.Generator,
    ):
        """
        Sample a ball around the given ``theta`` value.

        The size of the ball is controlled by the covariance matrix, ``cov``,
        which is multiplied by ``cov_factor``.

        .. warning::
            Samples are not gauranteed to be within the parameter bounds.

        :param theta:
            Parameter value around which the ball is centered.
        :param size:
            Number of samples to draw.
        :param cov:
            The covariance matrix.
            If not specified, a diagonal matrix will be used that has
            entry cov[j][j] set to the squared range of parameter j:
            (high - low)**2.
        :param cov_factor:
            Left-multiply the covariance matrix by this value.
        :param numpy.random.Generator rng:
            Numpy random generator.
        :return:
            The sampled values.
        """
        assert len(theta) == len(self)
        if cov is None:
            cov = np.diag([(p.high - p.low) ** 2 for p in self.values()])
        cov = cov_factor * cov
        return rng.multivariate_normal(theta, cov, size=size)

    def transform(self, thetas: np.ndarray, /) -> np.ndarray:
        """
        Transform values bounded by [param.low, param.high] to [-inf, inf].

        See :meth:`.itransform` for the inverse transformation.

        :param thetas:
            The values to be transformed.
        :return:
            Transformed parameter values.
        """
        thetas = np.atleast_2d(thetas)
        assert thetas.shape[-1] == len(self)
        return np.moveaxis(
            np.array(
                [p.transform(thetas[..., k]) for k, p in enumerate(self.values())]
            ),
            0,
            -1,
        )

    def itransform(self, thetas: np.ndarray, /) -> np.ndarray:
        """
        Transform values on [-inf, inf] to be bounded by [param.low, param.high].

        Performs the inverse of :meth:`.transform`.

        :param thetas:
            The values to be transformed.
        :return:
            Transformed parameter values.
        """
        thetas = np.atleast_2d(thetas)
        assert thetas.shape[-1] == len(self)
        return np.moveaxis(
            np.array(
                [p.itransform(thetas[..., k]) for k, p in enumerate(self.values())]
            ),
            0,
            -1,
        )

    def truncate(self, thetas: np.ndarray, /) -> np.ndarray:
        """
        Truncate values that are out of bounds.

        :param thetas:
            The values to be truncated.
        :return:
            Truncated parameter values.
        """
        thetas = np.atleast_2d(thetas)
        assert thetas.shape[-1] == len(self)
        return np.moveaxis(
            np.array([p.truncate(thetas[..., k]) for k, p in enumerate(self.values())]),
            0,
            -1,
        )

    def reflect(self, thetas: np.ndarray, /) -> np.ndarray:
        """
        Reflect values that are out of bounds by the amount they are out.

        As reflecting does not gaurantee values will be within the bounds,
        values are first truncated to (2*low - high, 2*high - low),
        then reflected. For example, with bounds low=0, high=10,
        a value of -11 will be truncated to -10, then reflected to attain
        a final value of 10.

        :param thetas:
            The values to be reflected.
        :return:
            Reflected parameter values.
        """
        thetas = np.atleast_2d(thetas)
        assert thetas.shape[-1] == len(self)
        return np.moveaxis(
            np.array([p.reflect(thetas[..., k]) for k, p in enumerate(self.values())]),
            0,
            -1,
        )

    @staticmethod
    def geometric_median(
        thetas: np.ndarray,
        /,
        *,
        probs: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Get the multivariate median of a weighted sample.

        :param thetas:
            Parameter values. thetas[j][k] is the value of the k'th parameter
            for the j'th multivariate sample.
        :param probs:
            Discriminator predictions corresponding to the ``thetas``.
        :return:
            Median position in multivariate space.
        """

        # Normalize by mean/stddev so each parameter is treated equally
        # in the objective function.
        mean = np.mean(thetas, axis=0)
        stddev = np.std(thetas, axis=0)
        v = (thetas - mean) / stddev

        def objective(x):
            """Minimise the sum of distances from v to x."""
            d = np.linalg.norm(x - v, axis=1)
            # Minimise the mean distance rather than the sum, to avoid extremely
            # large values that trigger the notorious scipy error:
            #   "Desired error not necessarily achieved due to precision loss."
            return np.average(d, weights=probs)

        x0 = np.zeros(v.shape[1])
        opt = scipy.optimize.minimize(objective, x0)
        if not opt.success:
            raise RuntimeError(f"Failed to find geometric median: {opt.message}")
        return mean + stddev * opt.x

    @staticmethod
    def top_n(
        thetas: np.ndarray,
        /,
        *,
        probs: np.ndarray,
        n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the top ``n`` values of ``thetas``, ranked by ``probs``.

        :param thetas:
            Parameter values. thetas[j][k] is the value of the k'th parameter
            for the j'th multivariate sample.
        :param probs:
            Discriminator predictions corresponding to the ``thetas``.
        :return:
            A 2-tuple of (top_thetas, top_probs).
        """
        assert n >= 1
        k = len(probs) - n
        assert k >= 1
        idx = np.argpartition(probs, k)[k:]
        return thetas[idx], probs[idx]
