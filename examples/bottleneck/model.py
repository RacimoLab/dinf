import demes
import msprime
import numpy as np

import dinf


recombination_rate = 1.25e-8
mutation_rate = 1.25e-8
num_individuals = 64
sequence_length = 1_000_000
parameters = dinf.Parameters(
    N0=dinf.Param(low=10, high=30_000, truth=10_000),
    N1=dinf.Param(low=10, high=30_000, truth=200),
)

bh_matrix = dinf.BinnedHaplotypeMatrix(
    num_individuals=num_individuals,
    num_bins=128,
    ploidy=2,
    phased=True,
    maf_thresh=0.05,
)


def demography(*, N0, N1) -> demes.Graph:
    b = demes.Builder(description="bottleneck")
    b.add_deme(
        "A",
        epochs=[
            dict(start_size=N0, end_time=100),
            dict(start_size=N1, end_time=0),
        ],
    )
    graph = b.resolve()
    return graph


def generator(seed, *, N0, N1):
    """Simulate with the parameters provided to us."""
    rng = np.random.default_rng(seed)
    graph = demography(N0=N0, N1=N1)
    demog = msprime.Demography.from_demes(graph)
    seed1, seed2 = rng.integers(low=1, high=2 ** 31, size=2)

    ts = msprime.sim_ancestry(
        samples=num_individuals,
        demography=demog,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed1,
        record_provenance=False,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed2)

    feature_matrix = bh_matrix.from_ts(ts, rng=rng)
    return feature_matrix


def target(seed):
    """Simulate with fixed values. I.e. the "true" parameter values."""
    assert all(p.truth is not None for p in parameters.values())
    sim_kwargs = {k: v.truth for k, v in parameters.items()}
    return generator(seed, **sim_kwargs)


genobuilder = dinf.Genobuilder(
    target_func=target,
    generator_func=generator,
    parameters=parameters,
    feature_shape=bh_matrix.shape,
)
