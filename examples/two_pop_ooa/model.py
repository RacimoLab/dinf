# OOA2 model from PG-GAN.
# Wang et al. 2021, https://doi.org/10.1111/1755-0998.13386
import pathlib
import string

import demes
import msprime
import numpy as np

import dinf


populations = ["YRI", "CEU"]
samples = dinf.get_samples_from_1kgp_metadata(
    "20130606_g1k_3202_samples_ped_population.txt", populations
)
contig_lengths = dinf.get_contig_lengths(
    "GRCh38_full_analysis_set_plus_decoy_hla.fa.fai",
    keep_contigs={f"chr{c + 1}" for c in range(21)},  # Exclude chrX, etc.
)
recombination_rate = 1.25e-8
mutation_rate = 1.25e-8
num_individuals = 48  # per population
sequence_length = 50_000
parameters = dinf.Parameters(
    N_anc=dinf.Param(low=1_000, high=25_000),
    N1=dinf.Param(low=1_000, high=30_000),
    N2=dinf.Param(low=1_000, high=30_000),
    N3=dinf.Param(low=1_000, high=30_000),
    T1=dinf.Param(low=1_500, high=5_000),
    T2=dinf.Param(low=100, high=1_500),
    mig=dinf.Param(low=-0.2, high=0.2),
)


def demography(*, N_anc, N1, N2, N3, T1, T2, mig):
    source = populations[0]
    dest = populations[1]
    if mig < 0:
        source, dest = dest, source
        mig = -mig

    model = string.Template(
        """
        description:
          Out-of-Africa model with two extant populations.
        time_units: generations
        doi:
          - Wang et al. 2021, https://doi.org/10.1111/1755-0998.13386
        demes:
          - name: anc
            epochs:
              - start_size: $N_anc
                end_time: $T1
          - name: YRI
            ancestors: [anc]
            epochs:
              - start_size: $N3
          - name: CEU
            ancestors: [anc]
            epochs:
              - start_size: $N1
                end_time: $T2
              - start_size: $N2
                end_time: 0
        pulses:
          - sources: [$source]
            dest: $dest
            time: $T2
            proportions: [$mig]
        """
    ).substitute(
        N_anc=N_anc,
        N1=N1,
        N2=N2,
        N3=N3,
        T1=T1,
        T2=T2,
        source=source,
        dest=dest,
        mig=mig,
    )
    return demes.loads(model)


features = dinf.MultipleBinnedHaplotypeMatrices(
    num_individuals={pop: num_individuals for pop in populations},
    num_loci={pop: 36 for pop in populations},
    ploidy={pop: 2 for pop in populations},
    global_phased=True,
)


def generator(seed, **theta):
    """Simulate with the parameters provided to us."""
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


dinf_model = dinf.DinfModel(
    target_func=target,
    generator_func=generator,
    parameters=parameters,
    feature_shape=features.shape,
)
