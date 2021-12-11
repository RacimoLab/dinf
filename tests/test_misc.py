import msprime
import pytest

import dinf.misc


def sim(ploidies: list):
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=10_000)
    demography.add_population(name="B", initial_size=10_000)
    demography.add_population(name="C", initial_size=1_000)
    demography.add_population_split(time=1000, derived=["A", "B"], ancestral="C")

    return msprime.sim_ancestry(
        demography=demography,
        samples=[
            msprime.SampleSet(5, population=population, ploidy=ploidy)
            for ploidy in ploidies
            for population in ("A", "B")
        ],
    )


def test_ts_individuals_with_population_name():
    ts = sim([2])
    individuals = dinf.misc.ts_individuals(ts, "A")
    for j in ts.samples():
        n = ts.node(j)
        pop_name = ts.population(n.population).metadata.get("name")
        assert (n.individual in individuals) == (pop_name == "A")


def test_ts_individuals_with_population_id():
    ts = sim([2])
    individuals = dinf.misc.ts_individuals(ts, 0)
    for j in ts.samples():
        n = ts.node(j)
        assert (n.individual in individuals) == (n.population == 0)


def test_ts_individuals_population_not_found():
    ts = sim([2])
    with pytest.raises(ValueError, match="'X' not found in the population table"):
        dinf.misc.ts_individuals(ts, "X")


def test_nodes_of_individuals():
    ts = sim([2])
    individuals = dinf.misc.ts_individuals(ts, "A")
    nodes = dinf.misc.ts_nodes_of_individuals(ts, individuals)
    for j in ts.samples():
        n = ts.node(j)
        assert (n.individual in individuals) == (j in nodes)


@pytest.mark.parametrize("sim_ploidy", [(2,), (1, 2, 3)])
def test_ploidy_of_individuals(sim_ploidy):
    ts = sim(sim_ploidy)
    individuals = dinf.misc.ts_individuals(ts, "A")
    ploidies = dinf.misc.ts_ploidy_of_individuals(ts, individuals)
    for i, ploidy in zip(individuals, ploidies):
        assert len(ts.individual(i).nodes) == ploidy
