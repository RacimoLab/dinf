from __future__ import annotations
import collections
import functools
import io
import pathlib
import sqlite3
from typing import Any, Callable
import zlib

import jax
import numpy as np
import numpy.typing as npt
import tskit


def ts_individuals(
    ts: tskit.TreeSequence,
    /,
    population: str | int | None = None,
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
    ts: tskit.TreeSequence,
    /,
    individuals: npt.NDArray[np.integer],
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
    ts: tskit.TreeSequence,
    /,
    individuals: npt.NDArray[np.integer],
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


def _dtree_map(func, *trees: Pytree) -> Pytree:
    return jax.tree_map(
        func, *trees, is_leaf=lambda x: not isinstance(x, collections.abc.Mapping)
    )


def _dtree_structure(tree: Pytree) -> Pytree:
    return jax.tree_structure(_dtree_map(lambda _: (), tree))


def tree_equal(tree: Pytree, *others: Pytree) -> bool:
    """
    Return True if tree is the same as all the others, False otherwise.
    """
    structure = _dtree_structure(tree)
    return all(
        structure == _dtree_structure(other)
        and jax.tree_util.tree_all(_dtree_map(np.array_equal, tree, other))
        for other in others
    )


def tree_shape(tree: Pytree) -> Pytree:
    """
    Return a pytree with the same dictionary structure as the given tree,
    but with non-dictionaries replaced by their shape.
    """
    return jax.tree_map(
        lambda x: np.shape(x),
        tree,
        is_leaf=lambda x: isinstance(x, (list, tuple)),
    )


def tree_cons(a, tree: Pytree) -> Pytree:
    """
    Prepend ``a`` in all tuples of the given tree.
    """
    return jax.tree_map(
        lambda x: tuple((a,) + tuple(x)),
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


def leading_dim_size(tree: Pytree) -> int:
    """Size of the leading dimension (e.g. batch dimension) of each feature."""
    sizes = np.array(jax.tree_flatten(tree_car(tree_shape(tree)))[0])
    # All features should have the same size for the leading dimension.
    assert np.all(sizes[0] == sizes[1:])
    return sizes[0]


def cache(path: str | pathlib.Path, /, *, split: int = 1000):
    """
    A decorator to cache the output of generator and/or target functions.

    This is analogous to :func:`functools.cache`, except each function's
    result is stored in a file under the given directory. Caching can create
    a large number of small files, so the files are split into subdirectories
    to mitigate possible problems.

    :param path:
        The directory to use for the cache.
    :param split:
        The number of subdirectories into which the cache files will be split.
    """
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True)
    # Split cache into subfolders.
    for i in range(split):
        (path / str(i)).mkdir(exist_ok=True)

    def outer_wrapper(func: Callable):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            filename = (
                func.__name__
                + "."
                + ".".join(map(str, args))
                + "."
                + ".".join(f"{k}_{v}" for k, v in kwargs.items())
            )
            # Use a fast hash of the filename to choose a subdirectory.
            subdir = str(zlib.adler32(bytes(filename, "ascii")) % split)
            cache_file = path / subdir / filename
            if cache_file.exists():
                data = np.load(cache_file)
                if isinstance(data, np.lib.npyio.NpzFile):
                    data = dict(data)
            else:
                data = func(*args, **kwargs)
                # We open the file ourselves, to stop numpy from adding
                # a .npy or .npz extension to the filename.
                with open(cache_file, "wb") as f:
                    if isinstance(data, collections.abc.Mapping):
                        np.savez(f, **data)
                    else:
                        np.save(f, data)
            return data

        return inner_wrapper

    return outer_wrapper


def sqlite_cache(db_file: str | pathlib.Path, shape, /):
    """
    A decorator for generator or target functions that caches features to disk.

    This is analogous to :func:`functools.cache`, except the cache is
    persisted to disk in an sqlite database.

    .. warning::

        Do not use a network filesystem for the database filename.
        Writing to an sqlite database depends on file locking support to
        safely deal with concurrent writers. File locking support is known
        to be broken for many network filesystems (e.g. NFS).
        After dinf has finished writing to the database file, it can be
        safely copied to a network filesystem for storage---just remember
        to copy it back to a local filesystem when updating the cache.

    :param db_file:
        The filename of the sqlite database.
    """

    def adapt_array(arr):
        out = io.BytesIO()
        np.save(out, arr)
        return sqlite3.Binary(out.getvalue())

    # Register numpy arrays to be converted to an sqlite BLOB.
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("ARRAY", lambda x: np.load(io.BytesIO(x)))

    con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)

    cur = con.cursor()
    if isinstance(shape, collections.abc.Mapping):
        columns = ", ".join(f"{k} ARRAY" for k in shape.keys())
        qry_columns = ", ".join("?" for k in shape.keys())
    else:
        columns = "feature ARRAY"
        qry_columns = "?"
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS features (
            function TEXT,
            args TEXT,
            kwargs TEXT,
            {columns},
            PRIMARY KEY (function, args, kwargs)
        )
        """
    )
    con.commit()

    def outer_wrapper(func: Callable):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            func.__name__
            args_str = ",".join(map(str, args))
            kwargs_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
            cur = con.cursor()

            cur.execute(
                f"""
                SELECT {columns} FROM features WHERE
                    function = :function AND args = :args AND kwargs = :kwargs
                """,
                dict(function=func.__name__, args=args_str, kwargs=kwargs_str),
            )
            results = cur.fetchall()

            if len(results) > 0:
                (features,) = results
                if len(features) == 1:
                    (feature,) = features
                else:
                    feature = {k: f for k, f in zip(shape, features)}
            else:
                feature = func(*args, **kwargs)
                if isinstance(feature, collections.abc.Mapping):
                    features = tuple(feature.values())
                else:
                    features = (feature,)
                cur.execute(
                    f"""
                    INSERT INTO features VALUES (?, ?, ?, {qry_columns})
                    """,
                    (func.__name__, args_str, kwargs_str, *features),
                )
                con.commit()

            return feature

        return inner_wrapper

    return outer_wrapper
