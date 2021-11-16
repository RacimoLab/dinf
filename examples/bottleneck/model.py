import demes
import numpy as np

import dinf


recombination_rate = 1.25e-8
mutation_rate = 1.25e-8
num_samples = 128
sequence_length = 1_000_000
parameters = dinf.Parameters(
    N0=dinf.Param(low=10, high=30000, truth=10000),
    N1=dinf.Param(low=10, high=30000, truth=200),
)

bh_matrix = dinf.BinnedHaplotypeMatrix(
    num_samples=num_samples, num_bins=128, maf_thresh=0.05
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
    ts = dinf.msprime_hudson_simulator(
        graph=graph,
        num_samples=num_samples,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        rng=rng,
    )
    feature_matrix = bh_matrix.from_ts(ts, rng=rng)
    return feature_matrix


def empirical(seed):
    """Simulate with fixed values. I.e. the "true" parameter values."""
    assert all(p.truth is not None for p in parameters.values())
    sim_kwargs = {k: v.truth for k, v in parameters.items()}
    return generator(seed, **sim_kwargs)


genobuilder = dinf.Genobuilder(
    empirical_func=empirical,
    generator_func=generator,
    parameters=parameters,
    feature_shape=bh_matrix.shape,
)
rng = np.random.default_rng(123)
dinf.mcmc_gan(
    genobuilder=genobuilder,
    iterations=3,
    training_replicates=1_000_000,
    test_replicates=10_000,
    epochs=1,
    walkers=64,
    steps=1000,
    Dx_replicates=64,
    working_directory="out",
    rng=rng,
)
