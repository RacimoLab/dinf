# EXP model from PG-GAN.
# Wang et al. 2021, https://doi.org/10.1111/1755-0998.13386
import pathlib

import demes
import msprime
import numpy as np

import dinf


population_name = "CEU"
samples = dinf.get_samples_from_1kgp_metadata(
    "20130606_g1k_3202_samples_ped_population.txt", [population_name]
)
contig_lengths = dinf.get_contig_lengths(
    "GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
    keep_contigs={f"chr{c + 1}" for c in range(21)},  # Exclude chrX, etc.
)
recombination_rate = 1.25e-8
mutation_rate = 1.25e-8
num_individuals = 64
sequence_length = 100_000
parameters = dinf.Parameters(
    N1=dinf.Param(low=1_000, high=30_000, truth=9_000),
    N2=dinf.Param(low=1_000, high=30_000, truth=5_000),
    # Growth rate starting from N2 at T2.
    growth=dinf.Param(low=0, high=0.05, truth=0.005),
    # Time units are generations ago.
    T1=dinf.Param(low=1_500, high=5_000, truth=2_000),
    T2=dinf.Param(low=100, high=1_500, truth=350),
)

features = dinf.BinnedHaplotypeMatrix(
    num_individuals=num_individuals,
    num_bins=128,
    maf_thresh=0.05,
    phased=True,
    ploidy=2,
)


def demography(*, N1, N2, growth, T1, T2):
    N3 = N2 * np.exp(T2 * growth)
    b = demes.Builder(description="One population with recent exponential growth")
    b.add_deme(
        population_name,
        epochs=[
            dict(start_size=N1, end_time=T1),
            dict(start_size=N2, end_time=T2),
            dict(start_size=N2, end_size=N3, end_time=0),
        ],
    )
    graph = b.resolve()
    return graph


def generator(seed, **theta):
    """Simulate with the parameters provided to us."""
    rng = np.random.default_rng(seed)
    graph = demography(**theta)
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

    feature_matrix = features.from_ts(ts, rng=rng)
    return feature_matrix


vb = dinf.BagOfVcf(
    pathlib.Path("bcf/").glob("*.bcf"),
    samples=samples,
    contig_lengths=contig_lengths,
)


def target(seed):
    """Sample genotype matrices from bcf files."""
    rng = np.random.default_rng(seed)
    feature_matrix = features.from_vcf(
        vb,
        sequence_length=sequence_length,
        max_missing_genotypes=0,
        min_seg_sites=20,
        rng=rng,
    )
    return feature_matrix


genobuilder = dinf.Genobuilder(
    target_func=target,
    generator_func=generator,
    parameters=parameters,
    feature_shape=features.shape,
)
