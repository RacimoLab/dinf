import string

import demes
import msprime
import numpy as np

import dinf


recombination_rate = 1.25e-8
mutation_rate = 1.25e-8
num_individuals = 16
sequence_length = 1_000_000
parameters = dinf.Parameters(
    N0=dinf.Param(low=10, high=30_000, truth=10_000),
    N1=dinf.Param(low=10, high=30_000, truth=200),
)


def demography(*, N0, N1):
    model = string.Template(
        """
        description: Two-epoch model with recent bottleneck.
        time_units: generations
        demes:
          - name: A
            epochs:
              - start_size: $N0
                end_time: 100
              - start_size: $N1
                end_time: 0
        """
    ).substitute(N0=N0, N1=N1)
    return demes.loads(model)


features = dinf.BinnedHaplotypeMatrix(
    num_individuals=num_individuals,
    num_loci=64,
    ploidy=2,
    phased=False,
    maf_thresh=0.05,
)


def generator(seed, *, N0, N1):
    """Simulate a two-epoch model with msprime."""
    rng = np.random.default_rng(seed)
    graph = demography(N0=N0, N1=N1)
    demog = msprime.Demography.from_demes(graph)
    seed1, seed2 = rng.integers(low=1, high=2**31, size=2)

    ts = msprime.sim_ancestry(
        samples=num_individuals,
        demography=demog,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed1,
        record_provenance=False,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed2)

    feature_matrix = features.from_ts(ts)
    return feature_matrix


dinf_model = dinf.DinfModel(
    target_func=None,
    generator_func=generator,
    parameters=parameters,
    feature_shape=features.shape,
)
