import jax
import msprime
import numpy as np
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


def test_tree_equal():
    assert dinf.misc.tree_equal(1, 1)
    assert dinf.misc.tree_equal(1, 1, 1, 1)
    assert dinf.misc.tree_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert dinf.misc.tree_equal({"x": 1}, {"x": 1})
    assert dinf.misc.tree_equal({"x": np.array([1, 2, 3])}, {"x": np.array((1, 2, 3))})
    assert dinf.misc.tree_equal({"x": np.array([1, 2, 3])}, {"x": np.array((1, 2, 3))})
    assert dinf.misc.tree_equal(
        {"x": np.array([1, 2, 3]), "y": {"w": {"z": np.zeros((1, 2))}}},
        {"x": np.array([1, 2, 3]), "y": {"w": {"z": np.zeros((1, 2))}}},
    )
    assert not dinf.misc.tree_equal(1, 2)
    assert not dinf.misc.tree_equal(1, 2, 1)
    assert not dinf.misc.tree_equal(1, 2, 2)
    assert not dinf.misc.tree_equal(np.array([2, 2, 3]), np.array([1, 2, 3]))
    assert not dinf.misc.tree_equal(
        np.array([2, 2, 3]), np.array([1, 2, 3]), np.array([2, 2, 3])
    )
    assert not dinf.misc.tree_equal(np.array([2, 2, 3]), np.array([2, 2, 3, 3]))
    assert not dinf.misc.tree_equal({"x": 1}, {"y": 1})
    assert not dinf.misc.tree_equal({"x": 1}, {"y": 1}, {"x": 1})
    assert not dinf.misc.tree_equal({"x": 1}, {"x": 2})
    assert not dinf.misc.tree_equal({"x": 1}, {"x": 2}, {"x": 1})
    assert not dinf.misc.tree_equal(
        {"x": np.array([1, 2, 3])}, {"y": np.array([1, 2, 3])}
    )
    assert not dinf.misc.tree_equal(
        {"x": np.array([2, 2, 3])}, {"x": np.array([1, 2, 3])}
    )
    assert not dinf.misc.tree_equal(
        {"x": np.array([1, 2, 3]), "y": 4}, {"x": np.array([1, 2, 3])}
    )
    assert not dinf.misc.tree_equal(
        {"x": np.array([1, 2, 3]), "y": {"w": {"z": np.zeros([1, 2])}}},
        {"x": np.array([1, 2, 3]), "y": {"w": {"a": np.zeros([1, 2])}}},
    )
    assert not dinf.misc.tree_equal(
        {"x": np.array([1, 2, 3]), "y": {"w": {"z": np.zeros([1, 2])}}},
        {"x": np.array([1, 2, 3]), "y": {"w": {"a": np.zeros([1, 2])}}},
        {"x": np.array([1, 2, 3]), "y": {"w": {"z": np.zeros([1, 2])}}},
    )


@pytest.mark.parametrize(
    "a",
    [
        np.zeros((1, 2, 3)),
        {"x": np.zeros((1, 2, 3)), "y": np.zeros((4, 5))},
        {"x": {"y": np.zeros((1, 2, 3)), "w": {"z": np.zeros((4, 5))}}},
    ],
)
def test_tree_shape(a):
    a_shape = dinf.misc.tree_shape(a)
    assert dinf.misc.tree_equal(jax.tree_structure(a), jax.tree_structure(a_shape))
    b = jax.tree_map(np.zeros, a_shape)
    assert dinf.misc.tree_equal(a, b)
    assert dinf.misc.tree_equal(a_shape, dinf.misc.tree_shape(a))


def test_tree_cons():
    assert dinf.misc.tree_cons(1, (2, 3)) == (1, 2, 3)
    assert dinf.misc.tree_cons(3, {"x": (2, 1)}) == {"x": (3, 2, 1)}
    assert dinf.misc.tree_cons(5, {"x": (2, 1), "y": {"z": (3, 4)}}) == {
        "x": (5, 2, 1),
        "y": {"z": (5, 3, 4)},
    }


def test_tree_car():
    assert dinf.misc.tree_car((1, 2, 3)) == 1
    assert dinf.misc.tree_car({"x": (3, 2, 1)}) == {"x": 3}
    assert dinf.misc.tree_car({"x": (5, 2, 1), "y": {"z": (5, 3, 4)}}) == {
        "x": 5,
        "y": {"z": 5},
    }


def test_tree_cdr():
    assert dinf.misc.tree_cdr((1, 2, 3)) == (2, 3)
    assert dinf.misc.tree_cdr({"x": (3, 2, 1)}) == {"x": (2, 1)}
    assert dinf.misc.tree_cdr({"x": (5, 2, 1), "y": {"z": (5, 3, 4)}}) == {
        "x": (2, 1),
        "y": {"z": (3, 4)},
    }


@pytest.mark.parametrize(
    "a,d",
    [
        (1, (2, 3)),
        (3, {"x": (2, 1)}),
        (5, {"x": (2, 1), "y": {"z": (3, 4)}}),
    ],
)
def test_cons_car_cdr(a, d):
    cons = dinf.misc.tree_cons(a, d)
    assert dinf.misc.tree_car(cons) == jax.tree_map(
        lambda _: a, cons, is_leaf=lambda x: isinstance(x, tuple)
    )
    assert dinf.misc.tree_cdr(cons) == d
