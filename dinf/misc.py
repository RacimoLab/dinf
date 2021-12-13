from __future__ import annotations
import collections
from typing import Any

import jax
import numpy as np
import numpy.typing as npt
import tskit


def ts_individuals(
    ts: tskit.TreeSequence, population: str | int | None = None
) -> npt.NDArray[np.integer]:
    """
    Get the individuals corresponding to the tree sequence's samples.

    :param ts:
        The tree sequence.
    :param population:
        Only return individuals from this population. The population may be
        a string identifier (which will be matched against the 'name'
        metadata field in the population table of the tree sequence),
        or an integer population id.
        If not specified, all sampled individuals will be returned.
    :return:
        An array of individual IDs (indices into the individuals table).
    """
    if isinstance(population, str):
        pop2idx = {p.metadata.get("name"): p.id for p in ts.populations()}
        if population not in pop2idx:
            raise ValueError(f"'{population}' not found in the population table")
        population = pop2idx[population]
    nodes = ts.samples(population)
    individuals = ts.tables.nodes.individual[nodes]
    return np.unique(individuals)


def ts_nodes_of_individuals(
    ts: tskit.TreeSequence, individuals: npt.NDArray[np.integer]
) -> npt.NDArray[np.integer]:
    """
    Get the nodes for the individuals.

    :param ts:
        The tree sequence.
    :param individuals:
        An array of individual IDs (indices into the individuals table).
    :return:
        An array of node IDs (indices into the nodes table).
    """
    return np.concatenate([ts.individual(i).nodes for i in individuals])


def ts_ploidy_of_individuals(
    ts: tskit.TreeSequence, individuals: npt.NDArray[np.integer]
) -> npt.NDArray[np.integer]:
    """
    Get the ploidy of the individuals.

    :param ts:
        The tree sequence.
    :param individuals:
        An array of individual IDs (indices into the individuals table).
    :return:
        An array of ploidies, one for each individual.
    """
    return np.array([len(ts.individual(i).nodes) for i in individuals])


# A type for jax pytrees.
# https://github.com/google/jax/issues/3340
Pytree = Any


def tree_equal(tree: Pytree, *others: Pytree) -> bool:
    """
    Return True if tree is the same as all the others, False otherwise.
    """
    tree_structure = jax.tree_structure(tree)
    return all(
        tree_structure == jax.tree_structure(other)
        and jax.tree_util.tree_all(jax.tree_map(np.array_equal, tree, other))
        for other in others
    )


class _OpaqueSequence(collections.abc.Sequence):
    """
    A wrapper for tuples so they're treated leaves in a pytree.
    """
    def __init__(self, t):
        self._t = t

    def __len__(self):
        return len(self._t)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._t[key]
        return _OpaqueSequence(self._t[key])

    def __repr__(self):
        return f"{self.__class__.__name__}({self._t})"

    def __eq__(self, other):
        if isinstance(other, _OpaqueSequence):
            other = other._t
        return self._t == other


def tree_shape(tree: Pytree) -> Pytree:
    """
    Return a pytree with the same dictionary structure as the given tree,
    but with non-dictionaries replaced by their shape.
    """
    return jax.tree_map(
        lambda x: _OpaqueSequence(np.shape(x)),
        tree,
        is_leaf=lambda x: isinstance(x, (list, tuple)),
    )


def tree_cons(a, tree: Pytree) -> Pytree:
    """
    Prepend ``a`` in all tuples of the given tree.
    """
    return jax.tree_map(
        lambda x: _OpaqueSequence((a,) + tuple(x)),
        tree,
        is_leaf=lambda x: isinstance(x, tuple),
    )


def tree_car(tree: Pytree) -> Pytree:
    """
    Return a tree of the leading values of all tuples in the given tree.
    """
    return jax.tree_map(lambda x: x[0], tree, is_leaf=lambda x: isinstance(x, tuple))


def tree_cdr(tree: Pytree) -> Pytree:
    """
    Return a tree of the trailing values of all tuples in the given tree.
    """
    return jax.tree_map(lambda x: x[1:], tree, is_leaf=lambda x: isinstance(x, tuple))
