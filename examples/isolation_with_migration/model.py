# IM model from PG-GAN.
# Wang et al. 2021, https://doi.org/10.1111/1755-0998.13386
import string

import demes
import msprime
import numpy as np

import dinf

populations = ["deme1", "deme2"]
mutation_rate = 1.25e-8
num_individuals = 48  # per population
sequence_length = 50_000
parameters = dinf.Parameters(
    # Recombination rate.
    reco=dinf.Param(low=1e-9, high=1e-7, truth=1.25e-8),
    N_anc=dinf.Param(low=1_000, high=25_000, truth=15_000),
    N1=dinf.Param(low=1_000, high=30_000, truth=9_000),
    N2=dinf.Param(low=1_000, high=30_000, truth=5_000),
    T_split=dinf.Param(low=500, high=20_000, truth=2_000),
    # Asymmetric migration.
    mig=dinf.Param(low=-0.2, high=0.2, truth=0.05),
)


def demography(*, N_anc, N1, N2, T_split, mig):
    T_mig = T_split / 2
    source = "deme1"
    dest = "deme2"
    if mig < 0:
        # negative migration rate means the opposite direction
        source, dest = dest, source
        mig = -mig

    template = string.Template(
        """
        description: Isolation with Migration
        time_units: generations
        doi:
          - Wang et al. 2021, https://doi.org/10.1111/1755-0998.13386
        demes:
          - name: ancestral
            epochs:
              - start_size: $N_anc
                end_time: $T_split
          - name: deme1
            ancestors: [ancestral]
            epochs:
              - start_size: $N1
          - name: deme2
            ancestors: [ancestral]
            epochs:
              - start_size: $N2
        pulses:
          - sources: [$source]
            dest: $dest
            time: $T_mig
            proportions: [$mig]
        """
    )
    model = template.substitute(
        N_anc=N_anc,
        N1=N1,
        N2=N2,
        T_split=T_split,
        mig=mig,
        source=source,
        dest=dest,
        T_mig=T_mig,
    )
    return demes.loads(model)


features = dinf.MultipleBinnedHaplotypeMatrices(
    num_individuals={pop: num_individuals for pop in populations},
    num_loci={pop: 36 for pop in populations},
    ploidy={pop: 2 for pop in populations},
    global_phased=True,
)


def generator(seed, **params):
    """Simulate two-deme isolation-with-migration model with msprime."""
    rng = np.random.default_rng(seed)
    recombination_rate = params.pop("reco")
    graph = demography(**params)
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


dinf_model = dinf.DinfModel(
    target_func=None,
    generator_func=generator,
    parameters=parameters,
    feature_shape=features.shape,
)
