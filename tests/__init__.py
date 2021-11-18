import os

# test on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"


import functools

import demes
import numpy as np

import dinf


def _generator(seed, N0, N1, *, bh_matrix, num_samples):
    """Simulate demography with the parameters provided to us."""

    b = demes.Builder(description="bottleneck")
    b.add_deme(
        "A",
        epochs=[
            dict(start_size=N0, end_time=100),
            dict(start_size=N1, end_time=0),
        ],
    )
    graph = b.resolve()

    rng = np.random.default_rng(seed)
    ts = dinf.msprime_hudson_simulator(
        graph=graph,
        num_samples=num_samples,
        sequence_length=1_000_000,
        recombination_rate=1.25e-8,
        mutation_rate=1.25e-8,
        rng=rng,
    )
    return bh_matrix.from_ts(ts, rng=rng)


def _truth_generator(f, parameters):
    assert all(p.truth is not None for p in parameters.values())
    kwargs = {k: p.truth for k, p in parameters.items()}
    return functools.update_wrapper(functools.partial(f, **kwargs), f)


def get_genobuilder() -> dinf.Genobuilder:
    """Create a genobuilder for tests."""
    num_samples = 128
    parameters = dinf.Parameters(
        N0=dinf.Param(low=10, high=30000, truth=10000),
        N1=dinf.Param(low=10, high=3000, truth=200),
    )
    bh_matrix = dinf.BinnedHaplotypeMatrix(
        num_samples=num_samples, num_bins=128, maf_thresh=0.05
    )
    generator = functools.partial(
        _generator, bh_matrix=bh_matrix, num_samples=num_samples
    )
    return dinf.Genobuilder(
        empirical_func=_truth_generator(generator, parameters),
        generator_func=generator,
        parameters=parameters,
        feature_shape=bh_matrix.shape,
    )
