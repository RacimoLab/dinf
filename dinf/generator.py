import numpy as np
import demes
import tskit
import msprime


def msprime_hudson_simulator(
    *,
    graph: demes.Graph,
    num_samples: int,
    sequence_length: int,
    recombination_rate: float,
    mutation_rate: float,
    rng: np.random.Generator,
) -> tskit.TreeSequence:
    """
    Simulate using msprime's default Hudson model with infinite sites.
    """
    demography = msprime.Demography.from_demes(graph)
    seed1, seed2 = rng.integers(low=1, high=2 ** 31, size=2)
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(num_samples, ploidy=1)],
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed1,
        # Disabled to avoid nasty memory growth due to circular refs.
        # https://github.com/tskit-dev/msprime/issues/1899
        # We're not saving the tree sequences, so we don't really need
        # provenance anyhow.
        record_provenance=False,
    )
    ts = msprime.sim_mutations(
        ts,
        rate=mutation_rate,
        random_seed=seed2,
        # TODO: relax these and fix the feature matrix.
        model=msprime.BinaryMutationModel(),
        discrete_genome=False,
    )
    return ts
