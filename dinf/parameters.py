import collections.abc
import copy
from dataclasses import dataclass

import numpy as np


@dataclass
class Param:
    """
    A single parameter whose value is to be inferred.
    """

    low: float
    high: float
    truth: float = None

    def draw_prior(self, num_replicates: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw a random sample from the prior distribution.

        :param num_replicates: The sample size.
        :param rng: The numpy random number generator.
        :return: A 1d numpy array of parameter values.
        """
        return rng.uniform(low=self.low, high=self.high, size=num_replicates)

    def bounds_contain(self, x: float) -> bool:
        """Returns True if x is within the parameter bounds, False otherwise."""
        return self.low <= x <= self.high


class Parameters(collections.abc.Mapping):
    """
    A collection of parameters whose values are to be inferred.
    """

    def __init__(self, **kwargs):
        self.posterior = None
        self.params = copy.deepcopy(kwargs)
        for k, v in self.params.items():
            if not isinstance(v, Param):
                raise TypeError("{k} must be a (sub)class of Param")

    def __getitem__(self, key):
        return self.params[key]

    def __iter__(self):
        yield from self.params

    def __len__(self):
        return len(self.params)

    def draw_prior(self, num_replicates: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw a random sample for the parameters, from the prior distribution.

        :param num_replicates: The sample size.
        :param rng: The numpy random number generator.
        :return:
            A 2d numpy array of parameter values, where ret[j][k] is the
            j'th draw for the k'th parameter.
        """
        np.transpose([p.draw_prior(num_replicates, rng) for p in self.values()])

    def draw(self, num_replicates: int, rng: np.random.Generator) -> np.ndarray:
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
        if self.posterior is None:
            # sample from the prior
            return self.draw_prior(num_replicates, rng)
        else:
            # sample from the posterior
            idx = rng.integers(low=0, high=len(self.posterior), size=num_replicates)
            return self.posterior[idx]

    def bounds_contain(self, x) -> bool:
        """Returns True if x is within the parameters' bounds, False otherwise."""
        return all(p.bounds_contain(xp) for p, xp in zip(self.values(), x))

    def bounds_contain_vec(self, xs: np.ndarray) -> np.ndarray:
        """
        Find which coordinates are contained in the parameter space.

        This is just a vectorised version of :meth:`bounds_contain()`.

        :param xs:
            A 2d numpy array, where xs[j][k] is the j'th coordinate for the
            k'th parameter.
        :return:
            A 1d numpy array of boolean values, where ret[j] is True if the
            j'th coordinate is within the parameter space, and False otherwise.
        """
        low = tuple(p.low for p in self.values())
        high = tuple(p.high for p in self.values())
        return np.all(np.logical_and(low <= xs, xs <= high), axis=1)

    def update_posterior(self, posterior) -> None:
        """
        Set the posterior sample for these parameters.

        :param posterior:
            A 2d array of parameter values, where posterior[j][k] is the
            j'th draw for the k'th parameter.
        """
        assert len(posterior.shape) == 2
        assert posterior.shape[-1] == len(self)
        self.posterior = posterior
