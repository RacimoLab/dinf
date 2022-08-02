import jax
import msprime
import numpy as np
import pytest

from dinf.misc import (
    ts_individuals,
    ts_nodes_of_individuals,
    ts_ploidy_of_individuals,
    pytree_equal,
    pytree_shape,
    pytree_dtype,
    pytree_cons,
    pytree_car,
    pytree_cdr,
    leading_dim_size,
)


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
    individuals = ts_individuals(ts, "A")
    for j in ts.samples():
        n = ts.node(j)
        pop_name = ts.population(n.population).metadata.get("name")
        assert (n.individual in individuals) == (pop_name == "A")


def test_ts_individuals_with_population_id():
    ts = sim([2])
    individuals = ts_individuals(ts, 0)
    for j in ts.samples():
        n = ts.node(j)
        assert (n.individual in individuals) == (n.population == 0)


def test_ts_individuals_population_not_found():
    ts = sim([2])
    with pytest.raises(ValueError, match="'X' not found in the population table"):
        ts_individuals(ts, "X")


def test_nodes_of_individuals():
    ts = sim([2])
    individuals = ts_individuals(ts, "A")
    nodes = ts_nodes_of_individuals(ts, individuals)
    for j in ts.samples():
        n = ts.node(j)
        assert (n.individual in individuals) == (j in nodes)


@pytest.mark.parametrize("sim_ploidy", [(2,), (1, 2, 3)])
def test_ploidy_of_individuals(sim_ploidy):
    ts = sim(sim_ploidy)
    individuals = ts_individuals(ts, "A")
    ploidies = ts_ploidy_of_individuals(ts, individuals)
    for i, ploidy in zip(individuals, ploidies):
        assert len(ts.individual(i).nodes) == ploidy


def test_pytree_equal():
    assert pytree_equal(1, 1)
    assert pytree_equal(1, 1, 1, 1)
    assert pytree_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert pytree_equal({"x": 1}, {"x": 1})
    assert pytree_equal({"x": np.array([1, 2, 3])}, {"x": np.array((1, 2, 3))})
    assert pytree_equal({"x": np.array([1, 2, 3])}, {"x": np.array((1, 2, 3))})
    assert pytree_equal(
        {"x": np.array([1, 2, 3]), "y": {"w": {"z": np.zeros((1, 2))}}},
        {"x": np.array([1, 2, 3]), "y": {"w": {"z": np.zeros((1, 2))}}},
    )
    assert not pytree_equal(1, 2)
    assert not pytree_equal(1, 2, 1)
    assert not pytree_equal(1, 2, 2)
    assert not pytree_equal(np.array([2, 2, 3]), np.array([1, 2, 3]))
    assert not pytree_equal(
        np.array([2, 2, 3]), np.array([1, 2, 3]), np.array([2, 2, 3])
    )
    assert not pytree_equal(np.array([2, 2, 3]), np.array([2, 2, 3, 3]))
    assert not pytree_equal({"x": 1}, {"y": 1})
    assert not pytree_equal({"x": 1}, {"y": 1}, {"x": 1})
    assert not pytree_equal({"x": 1}, {"x": 2})
    assert not pytree_equal({"x": 1}, {"x": 2}, {"x": 1})
    assert not pytree_equal({"x": np.array([1, 2, 3])}, {"y": np.array([1, 2, 3])})
    assert not pytree_equal({"x": np.array([2, 2, 3])}, {"x": np.array([1, 2, 3])})
    assert not pytree_equal(
        {"x": np.array([1, 2, 3]), "y": 4}, {"x": np.array([1, 2, 3])}
    )
    assert not pytree_equal(
        {"x": np.array([1, 2, 3]), "y": {"w": {"z": np.zeros([1, 2])}}},
        {"x": np.array([1, 2, 3]), "y": {"w": {"a": np.zeros([1, 2])}}},
    )
    assert not pytree_equal(
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
def test_pytree_shape_1(a):
    a_shape = pytree_shape(a)
    assert pytree_equal(a, a)
    assert pytree_equal(a_shape, a_shape)
    assert pytree_equal(
        jax.tree_util.tree_structure(a, is_leaf=lambda x: isinstance(x, tuple)),
        jax.tree_util.tree_structure(a_shape, is_leaf=lambda x: isinstance(x, tuple)),
    )
    assert pytree_equal(
        jax.tree_util.tree_structure(a_shape, is_leaf=lambda x: isinstance(x, tuple)),
        jax.tree_util.tree_structure(a, is_leaf=lambda x: isinstance(x, tuple)),
    )
    b = jax.tree_util.tree_map(
        np.zeros, a_shape, is_leaf=lambda x: isinstance(x, tuple)
    )
    assert pytree_equal(a, b)
    assert pytree_equal(b, a)
    assert pytree_equal(a_shape, pytree_shape(a))
    assert pytree_equal(pytree_shape(a), a_shape)


def test_pytree_shape_2():
    a = {"a": np.zeros((1, 2, 3)), "b": np.zeros((4, 5))}
    assert pytree_equal(pytree_shape(a), {"a": (1, 2, 3), "b": (4, 5)})
    assert pytree_equal(
        pytree_shape(a), {"a": np.array((1, 2, 3)), "b": np.array((4, 5))}
    )


def test_pytree_dtype():
    for a, dt in [
        (np.zeros((1, 2, 3), dtype=int), int),
        (np.zeros((1, 2, 3), dtype=np.int8), np.int8),
        (np.zeros((1, 2, 3), dtype=np.float32), np.float32),
        (
            {
                "x": np.zeros((1, 2, 3), dtype=int),
                "y": np.zeros((4, 5), dtype=np.float32),
            },
            {"x": int, "y": np.float32},
        ),
    ]:
        assert pytree_equal(pytree_dtype(a), dt)


def test_pytree_cons():
    assert pytree_cons(1, (2, 3)) == (1, 2, 3)
    assert pytree_cons(3, {"x": (2, 1)}) == {"x": (3, 2, 1)}
    assert pytree_cons(5, {"x": (2, 1), "y": {"z": (3, 4)}}) == {
        "x": (5, 2, 1),
        "y": {"z": (5, 3, 4)},
    }


def test_pytree_car():
    assert pytree_car((1, 2, 3)) == 1
    assert pytree_car({"x": (3, 2, 1)}) == {"x": 3}
    assert pytree_car({"x": (5, 2, 1), "y": {"z": (5, 3, 4)}}) == {
        "x": 5,
        "y": {"z": 5},
    }


def test_pytree_cdr():
    assert pytree_cdr((1, 2, 3)) == (2, 3)
    assert pytree_cdr({"x": (3, 2, 1)}) == {"x": (2, 1)}
    assert pytree_cdr({"x": (5, 2, 1), "y": {"z": (5, 3, 4)}}) == {
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
    cons = pytree_cons(a, d)
    assert pytree_car(cons) == jax.tree_util.tree_map(
        lambda _: a, cons, is_leaf=lambda x: isinstance(x, tuple)
    )
    assert pytree_cdr(cons) == d


@pytest.mark.parametrize(
    "a,b",
    [
        (np.zeros((2, 3)), 2),
        ({"x": np.zeros((2, 1))}, 2),
        ({"x": np.zeros((5, 2, 1)), "y": {"z": np.zeros((5, 2, 1))}}, 5),
    ],
)
def test_leading_dim_size(a, b):
    assert leading_dim_size(a) == b
