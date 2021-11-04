import abc
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import demes
import tskit
import msprime

from . import feature_extractor


@dataclass
class Parameter:
    name: str
    value: float
    bounds: Tuple[float, float]


class Generator(abc.ABC):
    """
    Abstract base class for generators.

    A generator consists of:
     - model parameters
     - a demographic model
     - a genetic simulator
     - a feature extractor
    """

    def __init__(
        self,
        *,
        num_samples: int,
        sequence_length: int,
        feature_extractor: feature_extractor.FeatureExtractor,
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.feature_extractor = feature_extractor

    @property
    @abc.abstractmethod
    def params(self) -> Sequence[Parameter]:
        """The list of model parameters."""
        raise NotImplementedError

    def draw_params(
        self, *, num_replicates: int, random: bool, rng: np.random.Generator
    ):
        if random:
            args = np.transpose(
                [rng.uniform(*p.bounds, size=num_replicates) for p in self.params]
            )
        else:
            # fixed value
            args = np.tile([p.value for p in self.params], (num_replicates, 1))
        return args

    @abc.abstractmethod
    def sim_ts(self, *args, rng: np.random.Generator) -> tskit.TreeSequence:
        """
        Simulates the model, returning a tree sequence.
        """
        raise NotImplementedError

    def sim(self, args: Tuple[int, Sequence[float]]) -> np.ndarray:
        """
        Simulates the model, returning a genotype matrix.

        :param args:
            A 2-tuple of (seed, sim_args), where seed is the seed for the
            random number generator and sim_args is a list of parameter
            values to be used during simulation.
        """
        seed, sim_args = args
        rng = np.random.default_rng(seed)
        ts = self.sim_ts(*sim_args, rng=rng)
        features = self.feature_extractor.from_ts(ts, rng=rng)
        return features

    # TODO: fix semantics
    def print(self):
        demography = self.demography(*[p.value for p in self.params])
        print(demography)


class MsprimeHudsonSimulator(abc.ABC):
    """
    Msprime simulation for Generator subclasses.
    """

    @abc.abstractmethod
    def demography(self, *args) -> demes.Graph:
        """Return a Demes demographic model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mutation_rate(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def recombination_rate(self) -> float:
        raise NotImplementedError

    def sim_ts(self, *args, rng: np.random.Generator) -> tskit.TreeSequence:
        """
        Simulate using msprime's default Hudson model with infinite sites.
        """
        graph = self.demography(*args)
        demography = msprime.Demography.from_demes(graph)
        seed1, seed2 = rng.integers(low=1, high=2 ** 31, size=2)
        ts = msprime.sim_ancestry(
            samples=[msprime.SampleSet(self.num_samples, ploidy=1)],
            demography=demography,
            sequence_length=self.sequence_length,
            recombination_rate=self.recombination_rate,
            random_seed=seed1,
        )
        ts = msprime.sim_mutations(
            ts,
            rate=self.mutation_rate,
            random_seed=seed2,
            # TODO: relax these and fix feature matrix.
            model=msprime.BinaryMutationModel(),
            discrete_genome=False,
        )
        return ts
