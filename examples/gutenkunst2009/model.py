from __future__ import annotations
import pathlib

import demes
import msprime
import numpy as np

import dinf
import dinf.misc


def load_samples_from_g1k_metadata(filename: str, populations: list) -> dict:
    """Return a dictionary mapping population name to a list of sample IDs."""
    data = np.recfromtxt(filename, names=True, encoding="ascii")
    # Remove related individuals.
    data = data[data.FatherID == "0"]
    data = data[data.MotherID == "0"]
    return {pop: data.SampleID[data.Population == pop].tolist() for pop in populations}


contig_lengths = dinf.get_contig_lengths(
    "GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
    keep_contigs={f"chr{c + 1}" for c in range(21)},  # Exclude chrX, etc.
)

populations = ["YRI", "CEU", "CHB"]
samples = load_samples_from_g1k_metadata(
    "20130606_g1k_3202_samples_ped_population.txt", populations
)
num_individuals = 64
recombination_rate = 1.25e-8
mutation_rate = 1.25e-8
sequence_length = 1_000_000

parameters = dinf.Parameters(
    # population sizes
    N_ancestral=dinf.Param(low=100, high=30_000),
    N_AMH=dinf.Param(low=100, high=30_000),
    N_OOA=dinf.Param(low=100, high=10_000),
    N_YRI=dinf.Param(low=100, high=100_000),
    N_CEU_start=dinf.Param(low=100, high=10_000),
    N_CEU_end=dinf.Param(low=1000, high=100_000),
    N_CHB_start=dinf.Param(low=100, high=10_000),
    N_CHB_end=dinf.Param(low=1000, high=100_000),
    # Time units match the demography, which we specified in "years".
    # To avoid explicitly defining constraints such as
    #   "CEU/CHB split more recently than the OOA event",
    # we parameterise times as time spans, rather than absolute times.
    # time span of AMH
    dT_AMH=dinf.Param(low=10_000, high=200_000),
    # time span of OOA
    dT_OOA=dinf.Param(low=5_000, high=200_000),
    # time span of CEU and CHB.
    dT_CEU_CHB=dinf.Param(low=10_000, high=50_000),
    # migration rates
    m_YRI_OOA=dinf.Param(low=1e-6, high=1e-2),
    m_YRI_CEU=dinf.Param(low=1e-6, high=1e-2),
    m_YRI_CHB=dinf.Param(low=1e-6, high=1e-2),
    m_CEU_CHB=dinf.Param(low=1e-6, high=1e-2),
)


def demography(**params) -> demes.Graph:
    assert params.keys() == parameters.keys()

    b = demes.Builder(
        description="Gutenkunst et al. (2009) three-population model.",
        doi=["10.1371/journal.pgen.1000695"],
        time_units="years",
        generation_time=25,
    )
    b.add_deme(
        "ancestral",
        epochs=[
            dict(
                end_time=params["dT_CEU_CHB"] + params["dT_OOA"] + params["dT_AMH"],
                start_size=params["N_ancestral"],
            )
        ],
    )
    b.add_deme(
        "AMH",
        ancestors=["ancestral"],
        epochs=[
            dict(
                end_time=params["dT_CEU_CHB"] + params["dT_OOA"],
                start_size=params["N_AMH"],
            )
        ],
    )
    b.add_deme(
        "OOA",
        ancestors=["AMH"],
        epochs=[dict(end_time=params["dT_CEU_CHB"], start_size=params["N_OOA"])],
    )
    b.add_deme("YRI", ancestors=["AMH"], epochs=[dict(start_size=params["N_YRI"])])
    b.add_deme(
        "CEU",
        ancestors=["OOA"],
        epochs=[dict(start_size=params["N_CEU_start"], end_size=params["N_CEU_end"])],
    )
    b.add_deme(
        "CHB",
        ancestors=["OOA"],
        epochs=[dict(start_size=params["N_CHB_start"], end_size=params["N_CHB_end"])],
    )
    b.add_migration(demes=["YRI", "OOA"], rate=params["m_YRI_OOA"])
    b.add_migration(demes=["YRI", "CEU"], rate=params["m_YRI_CEU"])
    b.add_migration(demes=["YRI", "CHB"], rate=params["m_YRI_CHB"])
    b.add_migration(demes=["CEU", "CHB"], rate=params["m_CEU_CHB"])
    return b.resolve()


features = dinf.MultipleBinnedHaplotypeMatrices(
    num_individuals={pop: num_individuals for pop in populations},
    num_bins={pop: 128 for pop in populations},
    ploidy={pop: 2 for pop in populations},
    # The so-called "phased" 1kG vcfs also contain unphased genotypes
    # for some individuals at some sites.
    global_phased=False,
    global_maf_thresh=0.05,
)


def generator(seed, **params):
    """Simulate with the parameters provided to us."""
    rng = np.random.default_rng(seed)
    graph = demography(**params)
    demog = msprime.Demography.from_demes(graph)
    seed1, seed2 = rng.integers(low=1, high=2**31, size=2)

    populations = list(samples)

    ts = msprime.sim_ancestry(
        samples={pop: num_individuals for pop in populations},
        demography=demog,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed1,
        record_provenance=False,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed2)
    individuals = {pop: dinf.misc.ts_individuals(ts, pop) for pop in populations}
    labelled_matrices = features.from_ts(ts, rng=rng, individuals=individuals)
    return labelled_matrices


vb = dinf.BagOfVcf(
    pathlib.Path("bcf/").glob("*.bcf"),
    samples=samples,
    contig_lengths=contig_lengths,
)


def target(seed):
    rng = np.random.default_rng(seed)
    labelled_matrices = features.from_vcf(
        vb,
        sequence_length=sequence_length,
        min_seg_sites=20,
        max_missing_genotypes=0,
        rng=rng,
    )
    return labelled_matrices


genobuilder = dinf.Genobuilder(
    target_func=target,
    generator_func=generator,
    parameters=parameters,
    feature_shape=features.shape,
)
