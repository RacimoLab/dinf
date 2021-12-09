import demes
import msprime
import numpy as np

import dinf

recombination_rate = 1.25e-8
mutation_rate = 1.25e-8
num_individuals = 128
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
    # Time units match the demography, which are in "years".
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

bh_matrix = dinf.BinnedHaplotypeMatrix(
    num_individuals=num_individuals,
    num_bins=128,
    maf_thresh=0.05,
    ploidy=2,
    phased=False,
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


ts = None


def generator(seed, **params):
    """Simulate with the parameters provided to us."""
    rng = np.random.default_rng(seed)
    graph = demography(**params)
    demog = msprime.Demography.from_demes(graph)
    seed1, seed2 = rng.integers(low=1, high=2 ** 31, size=2)

    samples = {"CHB": 10, "CEU": num_individuals, "YRI": 20}
    global ts
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demog,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed1,
        record_provenance=False,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed2)

    feature_matrix = bh_matrix.from_ts(ts, rng=rng, population="CEU")
    return feature_matrix


def target(seed):
    pass


genobuilder = dinf.Genobuilder(
    target_func=target,
    generator_func=generator,
    parameters=parameters,
    feature_shape=bh_matrix.shape,
)

rng = np.random.default_rng(123)
sim_kwargs = {k: v.draw_prior(1, rng)[0] for k, v in genobuilder.parameters.items()}
generator(123, **sim_kwargs)
pop2idx = {p.metadata.get("name"): p.id for p in ts.populations()}
population = pop2idx["CEU"]
nodes = ts.samples(population)
ploidy = 2
individual = np.reshape(ts.tables.nodes.individual[nodes], (-1, 2))
