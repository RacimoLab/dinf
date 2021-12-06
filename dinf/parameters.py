from __future__ import annotations
import collections.abc
import copy
import dataclasses

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True)
class Param:
    """
    A parameter whose value is to be inferred.
    """

    low: float
    """Parameter's lower bound."""

    high: float
    """Parameter's upper bound."""

    truth: float | None = None
    """
    The true value of the parameter. Set to None when the value is not known
    (e.g. when inferring parameters of empirical data). If the value is not
    None, this value may be used as the truth in a simulation study.
    """

    def __post_init__(self):
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError("Bounds must be finite.")
        if self.truth is not None and not self.bounds_contain(self.truth):
            raise ValueError(
                f"True value ({self.truth}) not in bounds [{self.low}, {self.high}]"
            )

    def draw_prior(
        self, size: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """
        Draw a random sample from the prior distribution.

        :param size: The sample size.
        :param rng: The numpy random number generator.
        :return: A numpy array of parameter values.
        """
        return rng.uniform(low=self.low, high=self.high, size=size)

    def bounds_contain(self, x: np.ndarray) -> npt.NDArray[np.bool_]:
        """
        Test if values are within in the parameter bounds.

        :param x: The value(s) that will be bounds checked.
        :return:
            A boolean value, or array of booleans with the same shape as
            the input. ret[j] is True if x[j] is in bounds, False otherwise.
        """
        x = np.atleast_1d(x)
        return np.logical_and(self.low <= x, x <= self.high)


class Parameters(collections.abc.Mapping):
    """
    An ordered collection of parameters whose values are to be inferred.
    """

    def __init__(self, **kwargs: Param):
        self._posterior: npt.NDArray | None = None
        self._params = copy.deepcopy(kwargs)
        for k, v in self._params.items():
            if not isinstance(v, Param):
                raise TypeError(f"{k} must be a (sub)class of Param")

    def __getitem__(self, key):
        return self._params[key]

    def __iter__(self):
        # yield the parameter names
        yield from self._params

    def __len__(self):
        return len(self._params)

    def draw_prior(
        self, num_replicates: int, rng: np.random.Generator
    ) -> npt.NDArray[np.float_]:
        """
        Draw a random sample for the parameters, from the prior distribution.

        :param num_replicates: The sample size.
        :param rng: The numpy random number generator.
        :return:
            A 2d numpy array of parameter values, where ret[j][k] is the
            j'th draw for the k'th parameter.
        """
        return np.transpose([p.draw_prior(num_replicates, rng) for p in self.values()])

    def draw(
        self, num_replicates: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """
        Draw a random sample for the parameters.

        If there is a posterior sample associated with the parameters,
        the sample will be drawn from the posterior with replacement.
        Otherise, the sample is drawn from the prior distribution.

        :param num_replicates: The sample size.
        :param rng: The numpy random number generator.
        :return:
            A 2d numpy array of parameter values, where ret[j][k] is the
            j'th draw for the k'th parameter.
        """
        if self._posterior is None:
            # sample from the prior
            return self.draw_prior(num_replicates, rng)
        else:
            # sample from the posterior
            idx = rng.integers(low=0, high=len(self._posterior), size=num_replicates)
            return self._posterior[idx]

    def bounds_contain(self, x: np.ndarray) -> npt.NDArray[np.bool_]:
        """
        Test if values are within in the parameter bounds.

        :param x:
            A 2d numpy array, where x[j][k] is the j'th coordinate for the
            k'th parameter.
        :return:
            A 1d numpy array of boolean values, where ret[j] is True if the
            j'th coordinate is within the parameter space, and False otherwise.
        """
        x = np.atleast_2d(x)
        assert x.shape[-1] == len(self)
        ret = np.all(
            [p.bounds_contain(x[..., k]) for k, p in enumerate(self.values())],
            axis=0,
        )
        return ret

    def update_posterior(self, posterior: npt.ArrayLike) -> None:
        """
        Set the posterior sample for these parameters.

        :param posterior:
            A 2d array of parameter values, where posterior[j][k] is the
            j'th draw for the k'th parameter.
        """
        posterior = np.array(posterior)
        assert len(posterior.shape) == 2
        assert posterior.shape[-1] == len(self)
        self._posterior = posterior
