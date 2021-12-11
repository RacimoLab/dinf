from __future__ import annotations

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
