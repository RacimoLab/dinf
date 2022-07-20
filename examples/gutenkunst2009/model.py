import pathlib
import string

import demes
import msprime
import numpy as np

import dinf


populations = ["YRI", "CEU", "CHB"]
samples = dinf.get_samples_from_1kgp_metadata(
    "20130606_g1k_3202_samples_ped_population.txt", populations=populations
)
contig_lengths = dinf.get_contig_lengths(
    "GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
    keep_contigs={f"chr{c + 1}" for c in range(21)},  # Exclude chrX, etc.
)
num_individuals = 64
recombination_rate = 1.25e-8
mutation_rate = 1.25e-8
sequence_length = 5_000_000

parameters = dinf.Parameters(
    # population sizes
    N_anc=dinf.Param(low=100, high=30_000),
    N_AMH=dinf.Param(low=100, high=30_000),
    N_OOA=dinf.Param(low=100, high=10_000),
    N_YRI=dinf.Param(low=100, high=100_000),
    N_CEU_start=dinf.Param(low=100, high=10_000),
    N_CEU_end=dinf.Param(low=1000, high=100_000),
    N_CHB_start=dinf.Param(low=100, high=10_000),
    N_CHB_end=dinf.Param(low=1000, high=100_000),
    # Time units match the demography, which are specified in "years".
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


def demography(**theta):
    # Arguments are expected to match the parameter names.
    assert theta.keys() == parameters.keys()

    theta["T_OOA_end"] = theta.pop("dT_CEU_CHB")
    theta["T_AMH_end"] = theta["T_OOA_end"] + theta.pop("dT_OOA")
    theta["T_anc_end"] = theta["T_AMH_end"] + theta.pop("dT_AMH")

    model = string.Template(
        """
        description: The Gutenkunst et al. (2009) out-of-Africa model.
        doi:
          - https://doi.org/10.1371/journal.pgen.1000695
        time_units: years
        generation_time: 25

        demes:
          - name: ancestral
            epochs:
              - {end_time: $T_anc_end, start_size: $N_anc}
          - name: AMH
            ancestors: [ancestral]
            epochs:
              - {end_time: $T_AMH_end, start_size: $N_AMH}
          - name: OOA
            ancestors: [AMH]
            epochs:
              - {end_time: $T_OOA_end, start_size: $N_OOA}
          - name: YRI
            ancestors: [AMH]
            epochs:
              - start_size: 12300
          - name: CEU
            ancestors: [OOA]
            epochs:
              - {start_size: $N_CEU_start, end_size: $N_CEU_end}
          - name: CHB
            ancestors: [OOA]
            epochs:
              - {start_size: $N_CHB_start, end_size: $N_CHB_end}

        migrations:
          - {demes: [YRI, OOA], rate: $m_YRI_OOA}
          - {demes: [YRI, CEU], rate: $m_YRI_CEU}
          - {demes: [YRI, CHB], rate: $m_YRI_CHB}
          - {demes: [CEU, CHB], rate: $m_CEU_CHB}
        """
    ).substitute(**theta)
    return demes.loads(model)


features = dinf.MultipleBinnedHaplotypeMatrices(
    num_individuals={pop: num_individuals for pop in populations},
    num_loci={pop: 128 for pop in populations},
    ploidy={pop: 2 for pop in populations},
    # The so-called "phased" 1kG vcfs also contain unphased genotypes
    # for some individuals at some sites.
    global_phased=False,
    global_maf_thresh=0.05,
)


def generator(seed, **theta):
    """Simulate the Gutenkunst out-of-Africa model with msprime."""
    rng = np.random.default_rng(seed)
    graph = demography(**theta)
    demog = msprime.Demography.from_demes(graph)
    seed1, seed2 = rng.integers(low=1, high=2**31, size=2)

    ts = msprime.sim_ancestry(
        samples={pop: num_individuals for pop in populations},
        demography=demog,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed1,
        record_provenance=False,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed2)
    individuals = {pop: dinf.ts_individuals(ts, pop) for pop in populations}
    labelled_matrices = features.from_ts(ts, individuals=individuals)
    return labelled_matrices


vcfs = dinf.BagOfVcf(
    pathlib.Path("bcf/").glob("*.bcf"),
    samples=samples,
    contig_lengths=contig_lengths,
)


def target(seed):
    rng = np.random.default_rng(seed)
    labelled_matrices = features.from_vcf(
        vcfs,
        sequence_length=sequence_length,
        min_seg_sites=20,
        max_missing_genotypes=0,
        rng=rng,
    )
    return labelled_matrices


dinf_model = dinf.DinfModel(
    target_func=target,
    generator_func=generator,
    parameters=parameters,
    feature_shape=features.shape,
)
