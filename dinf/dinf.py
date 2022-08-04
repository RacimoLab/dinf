from __future__ import annotations
import contextlib
import functools
import itertools
import logging
import os
import pathlib
import signal
from typing import Callable, Iterable, Protocol, Tuple
import zipfile

import emcee
import jax
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

# We're compatible with the standard lib's ``multiprocessing`` module,
# but ``multiprocess`` uses ``dill`` to pickle functions which provides
# greater flexibility to users (and fewer confusing errors).
# In particular, generator and target functions must be top-level functions
# when using the standard lib's ``multiprocessing``, but ``dill`` is able to
# pickle nested functions and class methods.
import multiprocess as multiprocessing

from .discriminator import Discriminator
from .dinf_model import DinfModel
from .parameters import Parameters
from .store import Store
from .misc import pytree_shape, pytree_dtype, pytree_cdr, is_tuple, spawn_key

logger = logging.getLogger(__name__)


def _worker_init(filename):
    # Ignore ctrl-c in workers. This is handled in the main process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Ensure that symbols from the user's dinf_model are avilable to workers.
    if filename is not None:
        DinfModel.from_file(filename)


class _SupportsImap(Protocol):
    """Type of a processing pool---an object with an ``imap()`` method."""

    def imap(self, func, iterable):
        ...


class _DummyPool:
    """Uniprocessor processing pool."""

    imap = map


@contextlib.contextmanager
def process_pool(parallelism: int | None, dinf_model: DinfModel):
    """
    A context manager to open a process pool with the "spawn" start method.

    This is a wrapper for multiprocessing.Pool, that gracefully exits
    when the user presses control-c to quit. We also don't use Pool's
    context manager because it exits with pool.terminate() and we'd rather
    pool.close() and pool.join(). See
    https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html

    The process pool must be started before the GPU has been initialised,
    or using the "spawn" start method, otherwise we get weird GPU resource
    issues because the fork()'ed subprocesses inherit CUDA resource locks.
    Note also that concurrent.futures uses fork() on unix, and the initial
    processes are forked on demand (https://bugs.python.org/issue39207),
    which means they can be forked after the GPU has been initialised.
    """
    if parallelism == 1:
        yield _DummyPool()
        return

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(
        processes=parallelism,
        initializer=_worker_init,
        initargs=(dinf_model.filename,),
        # Workers don't release resources properly, so recycle them
        # periodically to reduce memory consumption.
        maxtasksperchild=1000,
    )
    terminate = False
    try:
        yield pool
    except KeyboardInterrupt:
        terminate = True
        raise
    finally:
        if terminate:
            pool.terminate()
        else:
            pool.close()
        pool.join()


def _get_dataset_parallel(
    *,
    func: Callable,
    args: Iterable,
    num_replicates: int,
    pool: _SupportsImap,
    callbacks: dict | None = None,
    out=None,
):
    """
    Get features from a generator or target function.

    The function will be called in parallel using the given process pool.

    :param func:
        The function that will produce the features.
    :param args:
        Arguments that will be passed to ``func``.
    :param num_replicates:
        The number of replicates.
    :param pool:
        An object with an ``imap()`` method.
        E.g. a ``multiprocessing.Pool`` object.
    """

    if callbacks is None:
        callbacks = {}
    assert all(k in ("feature",) for k in callbacks)

    result = None
    treedef = None
    if out is not None:
        result = jax.tree_util.tree_leaves(out)
        treedef = jax.tree_util.tree_structure(out)

    if (cb := callbacks.get("feature")) is not None:
        cb(0)

    for j, M in enumerate(pool.imap(func, args)):
        if result is None:
            treedef = jax.tree_util.tree_structure(M)
            result = []
            for m in jax.tree_util.tree_leaves(M):
                result.append(np.empty((num_replicates, *m.shape), dtype=m.dtype))
        for res, m in zip(result, jax.tree_util.tree_leaves(M)):
            res[j] = m

        if (cb := callbacks.get("feature")) is not None:
            cb(j + 1)

    if out is None:
        out = jax.tree_util.tree_unflatten(treedef, result)
    return out


def _get_generator_dataset(
    *,
    generator: Callable,
    thetas: np.ndarray,
    pool: _SupportsImap,
    ss: np.random.SeedSequence,
    callbacks: dict | None = None,
    out=None,
):
    """
    Get features from the generator function.

    :param generator:
        The generator function.
    :param thetas:
        Parameter values to pass to the generator function.
    :param pool:
        An object with an ``imap()`` method.
        E.g. a ``multiprocessing.Pool`` object.
    :param numpy.random.SeedSequence ss:
        Numpy random number seed sequence.
    :return:
        A collection of features.
    """
    num_replicates = len(thetas)
    seeds = ss.spawn(num_replicates)
    data = _get_dataset_parallel(
        func=generator,
        args=zip(seeds, thetas),
        num_replicates=num_replicates,
        pool=pool,
        callbacks=callbacks,
        out=out,
    )
    return data


def _get_target_dataset(
    *,
    target: Callable,
    num_replicates: int,
    pool: _SupportsImap,
    ss: np.random.SeedSequence,
    callbacks: dict | None = None,
    out=None,
):
    """
    Get features from the target function.

    :param target:
        The target function.
    :param num_replicates:
        Number of replicates to sample from the target.
    :param pool:
        An object with an ``imap()`` method.
        E.g. a ``multiprocessing.Pool`` object.
    :param numpy.random.SeedSequence ss:
        Numpy random number seed sequence.
    :return:
        A collection of features.
    """
    seeds = ss.spawn(num_replicates)
    data = _get_dataset_parallel(
        func=target,
        args=seeds,
        num_replicates=num_replicates,
        pool=pool,
        callbacks=callbacks,
        out=out,
    )
    return data


def _alloc_features(
    feature_shape,
    feature_dtype,
    num_replicates,
):
    """
    Allocate empty numpy array(s) for ``num_replicates`` features.
    """
    leaves = []
    for shape, dtype in zip(
        jax.tree_util.tree_leaves(feature_shape, is_leaf=is_tuple),
        jax.tree_util.tree_leaves(feature_dtype, is_leaf=is_tuple),
    ):
        leaves.append(np.empty((num_replicates, *shape), dtype=dtype))

    treedef = jax.tree_util.tree_structure(feature_shape, is_leaf=is_tuple)
    x = jax.tree_util.tree_unflatten(treedef, leaves)
    return x


def _get_combined_dataset(
    *,
    target: Callable,
    generator: Callable,
    thetas: np.ndarray,
    pool: _SupportsImap,
    ss: np.random.SeedSequence,
    callbacks: dict | None = None,
):
    """
    Get a dataset comprised of both generator and target data.

    The returned data are not shuffled. We rely on Discriminator.fit()
    to extract data batches at random.

    :param target:
        The target function.
    :param generator:
        The generator function.
    :param thetas:
        Parameter values to pass to the generator function.
        An equivalent number of replicates will be sampled from the target.
    :param pool:
        An object with an ``imap()`` method.
        E.g. a ``multiprocessing.Pool`` object.
    :param numpy.random.SeedSequence ss:
        Numpy random number seed sequence.
    :return:
        A collection of features.
    """
    if callbacks is None:
        callbacks = {}
    assert all(k in ("generator/feature", "target/feature") for k in callbacks)
    ss_generator1, ss_generator2, ss_target = ss.spawn(3)

    num_replicates = len(thetas)

    # Get a small initial dataset to infer the shape and dtype of the data.
    if (n := os.cpu_count()) is None:
        n = 10
    n = min(n, num_replicates)
    x_generator_init = _get_generator_dataset(
        generator=generator,
        thetas=thetas[:n],
        pool=pool,
        ss=ss_generator1,
        callbacks=dict(feature=callbacks.get("generator/feature")),
    )

    # Allocate enough space for both the generator and target datasets,
    # so that we don't have to copy them to create a concatenated dataset.
    # For very large datasets, concatenating would double peak memory use.
    feature_shape = pytree_cdr(pytree_shape(x_generator_init))
    feature_dtype = pytree_dtype(x_generator_init)
    x = _alloc_features(feature_shape, feature_dtype, 2 * num_replicates)

    # Copy the initial generator dataset into the allocated memory.
    for xj, xj_init in zip(
        jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(x_generator_init)
    ):
        xj[:n] = xj_init

    if n < num_replicates:
        _get_generator_dataset(
            generator=generator,
            thetas=thetas[n:],
            pool=pool,
            ss=ss_generator2,
            callbacks=dict(feature=callbacks.get("generator/feature")),
            out=jax.tree_util.tree_map(lambda a: a[n:num_replicates], x),
        )

    _get_target_dataset(
        target=target,
        num_replicates=num_replicates,
        pool=pool,
        ss=ss_target,
        callbacks=dict(feature=callbacks.get("target/feature")),
        out=jax.tree_util.tree_map(lambda a: a[num_replicates:], x),
    )

    y = np.concatenate((np.zeros(num_replicates), np.ones(num_replicates)))
    return x, y


def save_results(
    filename: str | pathlib.Path,
    /,
    *,
    thetas: np.ndarray | None,
    probs: np.ndarray,
    parameters: Parameters,
):
    """
    Save discriminator predictions and parameter values to a .npz file.

    :param filename:
        The output file.
    :param thetas:
        Parameter values.
    :param probs:
        Discriminator predictions corresponding to the ``thetas``.
    :param parameters:
        Sequence of parameter names.
    """
    if thetas is not None:
        if thetas.shape[-1] != len(parameters):
            raise ValueError(
                f"thetas.shape={thetas.shape}, but got {len(parameters)} parameters"
            )
        if thetas.shape[:-1] != probs.shape:
            raise ValueError(
                f"thetas.shape={thetas.shape}, but got probs.shape={probs.shape}"
            )
    assert "_Pr" not in parameters
    kw = {"_Pr": probs}
    if thetas is not None:
        kw.update(
            **{par_name: thetas[..., j] for j, par_name in enumerate(parameters)},
        )
    # We open the file ourselves to stop numpy from adding a .npz extension.
    with open(filename, "wb") as f:
        np.savez(f, **kw)


def _npz_array_metadata(filename: str | pathlib.Path, /):
    """Read array metadata from an npz file without loading the arrays."""

    def read_array_header(f, version):
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            return np.lib.format.read_array_header_2_0(f)
        else:
            # Probably version 3.0, created with utf8 array names.
            # Numpy doesn't have a public API for this, so just use
            # the internal numpy function that has existed since 2014.
            return np.lib.format._read_array_header(f, version=version)

    names = []
    formats = []
    shapes = []
    with zipfile.ZipFile(filename) as zf:
        for name in zf.namelist():
            assert name.endswith(".npy")
            with zf.open(name) as f:
                version = np.lib.format.read_magic(f)
                shape, _, dtype = read_array_header(f, version)
            short_name = name[: len(name) - len(".npy")]
            names.append(short_name)
            formats.append(dtype.str)
            shapes.append(shape)

    return names, formats, shapes


def load_results(
    filename: str | pathlib.Path, /, *, parameters: Parameters | None = None
) -> np.ndarray:
    """
    Load descriminator predictions and parameter values from a .npz file.

    :param filename:
        The input file.
    :param parameters:
        Sequence of parameter names against which the file's
        arrays will be checked. A ValueError exception is raised
        if the names are not the same (and in the same order).
        If None, the parameter names are not checked.
    :return:
        A numpy structured array, where the first column is the probabilities
        (named ``_Pr``), and the subsequence columns are the
        parameter values (if any).
    """
    names, formats, shapes = _npz_array_metadata(filename)
    # All arrays had the same shape when the file was saved.
    assert all(shape == shapes[0] for shape in shapes[1:])

    if parameters is not None:
        expected_names = ["_Pr"] + list(parameters)
        if names != expected_names:
            raise ValueError(
                f"{filename}: expected arrays {expected_names}, but got {names}"
            )
    elif names[0] != "_Pr":
        raise ValueError(f"{filename}: expected array '_Pr', but got {names}")

    npzfile = np.load(filename)
    arr = np.empty(shapes[0], dtype=dict(names=names, formats=formats))
    for name in names:
        arr[name] = npzfile[name]
    return arr


def _load_results_unstructured(
    filename: str | pathlib.Path, /, *, parameters: Parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """Split results into thetas and probs."""
    data = load_results(filename, parameters=parameters)
    names = list(data.dtype.names)
    probs = data["_Pr"]
    thetas = structured_to_unstructured(data[names[1:]])
    return thetas, probs


def _train_discriminator(
    *,
    discriminator: Discriminator,
    dinf_model: DinfModel,
    training_thetas: np.ndarray,
    test_thetas: np.ndarray,
    epochs: int,
    pool: _SupportsImap,
    ss: np.random.SeedSequence,
    entropy_regularisation: bool = False,
    callbacks: dict | None = None,
):
    if callbacks is None:
        callbacks = {}
    assert all(
        k
        in (
            "train/generator/feature",
            "train/target/feature",
            "test/generator/feature",
            "test/target/feature",
            "discriminator/fit/epoch",
            "discriminator/fit/train_batch",
            "discriminator/fit/test_batch",
        )
        for k in callbacks
    )

    ss_train, ss_val, ss_fit = ss.spawn(3)

    assert dinf_model.target_func is not None
    train_x, train_y = _get_combined_dataset(
        target=dinf_model.target_func,
        generator=dinf_model.generator_func_v,
        thetas=training_thetas,
        pool=pool,
        ss=ss_train,
        callbacks={
            "generator/feature": callbacks.get("train/generator/feature"),
            "target/feature": callbacks.get("train/target/feature"),
        },
    )

    val_x, val_y = None, None
    if test_thetas is not None and len(test_thetas) > 0:
        val_x, val_y = _get_combined_dataset(
            target=dinf_model.target_func,
            generator=dinf_model.generator_func_v,
            thetas=test_thetas,
            pool=pool,
            ss=ss_val,
            callbacks={
                "generator/feature": callbacks.get("test/generator/feature"),
                "target/feature": callbacks.get("test/target/feature"),
            },
        )

    metrics = discriminator.fit(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        epochs=epochs,
        ss=ss_fit,
        # Clear the training loss/accuracy metrics from last iteration.
        reset_metrics=True,
        entropy_regularisation=entropy_regularisation,
        callbacks={
            "epoch": callbacks.get("discriminator/fit/epoch"),
            "train_batch": callbacks.get("discriminator/fit/train_batch"),
            "test_batch": callbacks.get("discriminator/fit/test_batch"),
        },
    )

    return metrics


def train(
    *,
    dinf_model: DinfModel,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    parallelism: None | int = None,
    seed: None | int = None,
    callbacks: dict | None = None,
) -> Discriminator:
    """
    Train a discriminator network.

    :param dinf_model:
        DinfModel object that describes the model components.
    :param training_replicates:
        Size of the dataset used to train the discriminator.
    :param test_replicates:
        Size of the test dataset used to evaluate the discriminator after
        each training epoch.
    :param epochs:
        Number of full passes over the training dataset when training
        the discriminator.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`DinfModel.generator_func` and
        :meth:`DinfModel.target_func`.
    :param seed:
        Seed for the random number generator.
    :return:
        The trained discriminator.
    """
    assert dinf_model.target_func is not None
    ss = np.random.SeedSequence(seed, spawn_key=spawn_key("train"))
    ss_thetas, ss_discr = ss.spawn(2)
    rng_train, rng_test = (np.random.default_rng(sj) for sj in ss_thetas.spawn(2))

    training_thetas = dinf_model.parameters.sample_prior(
        size=training_replicates // 2, rng=rng_train
    )
    test_thetas = dinf_model.parameters.sample_prior(
        size=test_replicates // 2, rng=rng_test
    )

    discriminator = Discriminator(network=dinf_model.discriminator_network)

    with process_pool(parallelism, dinf_model) as pool:
        _train_discriminator(
            discriminator=discriminator,
            dinf_model=dinf_model,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            pool=pool,
            ss=ss_discr,
            callbacks=callbacks,
        )
    return discriminator


def predict(
    *,
    discriminator: Discriminator,
    dinf_model: DinfModel,
    replicates: int,
    sample_target: bool = False,
    parallelism: None | int = None,
    seed: int | None = None,
    callbacks: dict | None = None,
) -> Tuple[np.ndarray | None, np.ndarray]:
    """
    Sample features and make predictions using the discriminator.

    Features are sampled from the generator by default.
    To instead sample features from the target dataset, use the
    ``sample_target`` option.

    :param dinf_model:
        DinfModel object that describes the model components.
    :param replicates:
        Number of features to extract.
    :param sample_target:
        If True, sample features from the target dataset.
        If False (the default), features are sampled from the generator.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`DinfModel.generator_func` and
        :meth:`DinfModel.target_func`.
    :param seed:
        Seed for the random number generator.
    :return:
        A 2-tuple of (thetas, probs), where ``thetas`` are the drawn parameters
        (or ``None`` if ``sample_target`` is True) and ``probs`` are the
        discriminator predictions.
        thetas[j][k] is the j'th draw for the k'th parameter, and
        probs[j] is the discriminator prediction for the j'th draw.
    """
    assert dinf_model.target_func is not None
    ss = np.random.SeedSequence(seed, spawn_key=spawn_key("predict"))
    ss_target, ss_thetas, ss_generator = ss.spawn(3)

    if callbacks is None:
        callbacks = {}
    assert all(
        k
        in (
            "predict/generator/feature",
            "predict/target/feature",
            "discriminator/predict/batch",
        )
        for k in callbacks
    )

    with process_pool(parallelism, dinf_model) as pool:
        if sample_target:
            x = _get_target_dataset(
                target=dinf_model.target_func,
                num_replicates=replicates,
                pool=pool,
                ss=ss_target,
                callbacks={"feature": callbacks.get("predict/target/feature")},
            )
            thetas = None
        else:
            thetas = dinf_model.parameters.sample_prior(
                size=replicates, rng=np.random.default_rng(ss_thetas)
            )
            x = _get_generator_dataset(
                generator=dinf_model.generator_func_v,
                thetas=thetas,
                pool=pool,
                ss=ss_generator,
                callbacks={"feature": callbacks.get("predict/generator/feature")},
            )
    probs = discriminator.predict(
        x, callbacks={"batch": callbacks.get("discriminator/predict/batch")}
    )
    return thetas, probs


def _log_prob(
    thetas: np.ndarray,
    *,
    discriminator: Discriminator,
    generator: Callable,
    parameters: Parameters,
    ss: np.random.SeedSequence,
    num_replicates: int,
    pool: _SupportsImap,
) -> np.ndarray:
    """
    Function to be maximised by mcmc. Vectorised version.
    """
    num_walkers, num_params = thetas.shape
    assert num_params == len(parameters)
    log_D = np.full(num_walkers, -np.inf)

    # Identify workers with one or more out-of-bounds parameters.
    in_bounds = parameters.bounds_contain(thetas)
    num_in_bounds = np.sum(in_bounds)
    if num_in_bounds == 0:
        return log_D

    seeds = ss.spawn(num_replicates * num_in_bounds)
    thetas_reps = np.repeat(thetas[in_bounds], num_replicates, axis=0)
    assert len(seeds) == len(thetas_reps)
    M = _get_dataset_parallel(
        func=generator,
        args=zip(seeds, thetas_reps),
        num_replicates=len(seeds),
        pool=pool,
    )
    Dreps = discriminator.predict(M).reshape(num_in_bounds, num_replicates)
    D = np.mean(Dreps, axis=1)
    assert len(D) == num_in_bounds
    with np.errstate(divide="ignore"):
        log_D[in_bounds] = np.log(D)

    return log_D


def _run_mcmc_emcee(
    start: np.ndarray,
    discriminator: Discriminator,
    generator: Callable,
    parameters: Parameters,
    pool: _SupportsImap,
    walkers: int,
    steps: int,
    Dx_replicates: int,
    ss: np.random.SeedSequence,
    callbacks: dict | None = None,
):
    if callbacks is None:
        callbacks = {}
    assert all(k in ("mcmc",) for k in callbacks)

    ss_log_prob, ss_mt = ss.spawn(2)

    sampler = emcee.EnsembleSampler(
        walkers,
        len(parameters),
        _log_prob,
        vectorize=True,
        kwargs=dict(
            discriminator=discriminator,
            generator=generator,
            parameters=parameters,
            pool=pool,
            num_replicates=Dx_replicates,
            ss=ss_log_prob,
        ),
    )

    mt_seed = ss_mt.generate_state(1)[0]
    mt_initial_state = np.random.mtrand.RandomState(mt_seed).get_state()
    state = emcee.State(start, random_state=mt_initial_state)

    if (cb_mcmc := callbacks.get("mcmc")) is not None:
        cb_mcmc(0)

    for j, _ in enumerate(sampler.sample(state, iterations=steps)):
        if cb_mcmc is not None:
            cb_mcmc(j + 1)

    thetas = sampler.get_chain()
    assert thetas.shape == (steps, walkers, len(parameters))

    with np.errstate(over="ignore"):
        probs = np.exp(sampler.get_log_prob())
    assert probs.shape == (steps, walkers)

    logger.info("MCMC acceptance rate: %s", np.mean(sampler.acceptance_fraction))

    return thetas, probs


def mcmc_gan(
    *,
    dinf_model: DinfModel,
    iterations: int,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    walkers: int,
    steps: int,
    Dx_replicates: int,
    output_folder: None | str | pathlib.Path = None,
    parallelism: None | int = None,
    seed: None | int = None,
    callbacks: dict | None = None,
):
    """
        Run the MCMC GAN.

    Conceptually, the GAN takes the following steps for iteration j:

      - sample training dataset from the prior[j] distribution,
      - train the discriminator,
      - run the MCMC,
      - obtain posterior[j] as weighted KDE of MCMC sample,
      - set prior[j+1] = posterior[j].

    In the first iteration, the parameter values given to the generator
    to produce the training dataset are drawn from the parameters' prior
    distribution. In subsequent iterations, the parameter values are drawn
    from a weighted gaussian KDE of the previous iteration's MCMC chains.

    :param dinf_model:
        DinfModel object that describes the model components.
    :param iterations:
        Number of GAN iterations.
    :param training_replicates:
        Size of the dataset used to train the discriminator.
        This dataset is constructed once each GAN iteration.
    :param test_replicates:
        Size of the test dataset used to evaluate the discriminator after
        each training epoch.
    :param epochs:
        Number of full passes over the training dataset when training
        the discriminator.
    :param walkers:
        Number of independent MCMC chains.
    :param steps:
        The chain length for each MCMC walker.
    :param Dx_replicates:
        Number of generator replicates for approximating E[D(x)|θ].
    :param output_folder:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`DinfModel.generator_func` and
        :meth:`DinfModel.target_func`.
    :param seed:
        Seed for the random number generator.
    """
    assert dinf_model.target_func is not None
    if callbacks is None:
        callbacks = {}
    assert all(
        k
        in (
            "test/generator/feature",
            "test/target/feature",
            "iteration",
            "train/generator/feature",
            "train/target/feature",
            "fit/epoch",
            "fit/train_batch",
            "fit/test_batch",
            "mcmc",
        )
        for k in callbacks
    )

    if output_folder is None:
        output_folder = "."
    store = Store(output_folder, create=True)
    store.assert_complete(["discriminator.nn", "mcmc.npz"])
    resume = len(store) > 0

    ss = np.random.SeedSequence(seed, spawn_key=spawn_key("mcmc-gan"))
    ss_loop, ss_thetas, ss_test = ss.spawn(3)
    rng_thetas = np.random.default_rng(ss_thetas)

    parameters = dinf_model.parameters

    if resume:
        discriminator = Discriminator.from_file(
            store[-1] / "discriminator.nn", network=dinf_model.discriminator_network
        )
        thetas, y = _load_results_unstructured(
            store[-1] / "mcmc.npz", parameters=parameters
        )
        assert len(thetas.shape) == 3
        start = thetas[-1]
        if len(start) != walkers:
            # TODO: allow this by sampling start points for the walkers?
            raise ValueError(
                f"request for {walkers} walkers, but resuming from "
                f"{store[-1] / 'mcmc.npz'} which used {len(start)} walkers."
            )

        training_thetas = parameters.sample_kde(
            thetas.reshape(-1, thetas.shape[-1]),
            probs=y.reshape(-1),
            size=training_replicates // 2,
            rng=rng_thetas,
        )
    else:
        discriminator = Discriminator(network=dinf_model.discriminator_network)

        # Starting point for the mcmc chain.
        start = parameters.sample_prior(size=walkers, rng=rng_thetas)

        training_thetas = parameters.sample_prior(
            size=training_replicates // 2, rng=rng_thetas
        )

    # If start values are linearly dependent, emcee complains loudly.
    assert not np.any((start[0] == start[1:]).all(axis=-1))

    n_target_calls = 0
    n_generator_calls = 0

    with process_pool(parallelism, dinf_model) as pool:

        val_x, val_y = None, None
        if test_replicates >= 2:
            ss_val, ss_test_thetas = ss_test.spawn(2)
            test_thetas = parameters.sample_prior(
                size=test_replicates // 2, rng=np.random.default_rng(ss_test_thetas)
            )
            val_x, val_y = _get_combined_dataset(
                target=dinf_model.target_func,
                generator=dinf_model.generator_func_v,
                thetas=test_thetas,
                pool=pool,
                ss=ss_val,
                callbacks={
                    "generator/feature": callbacks.get("test/generator/feature"),
                    "target/feature": callbacks.get("test/target/feature"),
                },
            )

            n_target_calls += test_replicates // 2
            n_generator_calls += test_replicates // 2

        if (cb_iter := callbacks.get("iteration")) is not None:
            cb_iter(len(store))

        for i in range(len(store), len(store) + iterations):
            logger.info("MCMC GAN iteration %s", i)

            ss_train, ss_fit, ss_mcmc = ss_loop.spawn(3)

            train_x, train_y = _get_combined_dataset(
                target=dinf_model.target_func,
                generator=dinf_model.generator_func_v,
                thetas=training_thetas,
                pool=pool,
                ss=ss_train,
                callbacks={
                    "generator/feature": callbacks.get("train/generator/feature"),
                    "target/feature": callbacks.get("train/target/feature"),
                },
            )
            n_target_calls += training_replicates // 2
            n_generator_calls += training_replicates // 2

            discriminator.fit(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                epochs=epochs,
                ss=ss_fit,
                # Clear the training loss/accuracy metrics from last iteration.
                reset_metrics=True,
                callbacks={
                    "epoch": callbacks.get("fit/epoch"),
                    "train_batch": callbacks.get("fit/train_batch"),
                    "test_batch": callbacks.get("fit/test_batch"),
                },
            )
            del train_x
            del train_y

            store.increment()
            discriminator.to_file(store[-1] / "discriminator.nn")

            thetas, probs = _run_mcmc_emcee(
                start=start,
                discriminator=discriminator,
                generator=dinf_model.generator_func_v,
                parameters=parameters,
                pool=pool,
                walkers=walkers,
                steps=steps,
                Dx_replicates=Dx_replicates,
                ss=ss_mcmc,
                callbacks={"mcmc": callbacks.get("mcmc")},
            )
            assert thetas.shape == (steps, walkers, len(parameters))
            save_results(
                store[-1] / "mcmc.npz",
                thetas=thetas,
                probs=probs,
                parameters=parameters,
            )

            # The chain for next iteration starts at the end of this chain.
            start = thetas[-1]

            training_thetas = parameters.sample_kde(
                thetas.reshape(-1, thetas.shape[-1]),
                probs=probs.reshape(-1),
                size=training_replicates // 2,
                rng=rng_thetas,
            )

            n_generator_calls += walkers * steps * Dx_replicates
            logger.info("Target called %s times.", n_target_calls)
            logger.info("Generator called %s times.", n_generator_calls)

            if (cb_iter := callbacks.get("iteration")) is not None:
                cb_iter(len(store))


def abc_gan(
    *,
    dinf_model: DinfModel,
    iterations: int,
    training_replicates: int,
    test_replicates: int,
    proposal_replicates: int,
    epochs: int,
    top_n: int | None = None,
    output_folder: None | str | pathlib.Path = None,
    parallelism: None | int = None,
    seed: None | int = None,
    callbacks: dict | None = None,
):
    """
    Adversarial Abstract Bayesian Computation.

    Conceptually, the GAN takes the following steps for iteration j:

      - sample training and proposal datasets from the prior[j] distribution,
      - train the discriminator,
      - make predictions with the discriminator on the proposal dataset,
      - construct a posterior[j] sample from the proposal dataset,
      - set prior[j+1] = posterior[j].

    In the first iteration, the parameter values given to the generator
    to produce the train/proposal datasets are drawn from the parameters'
    prior distribution. In subsequent iterations, the parameter values
    are drawn from a posterior ABC sample. The posterior is obtained by
    rejection sampling the proposal distribution and weighting the posterior
    by the discriminator predictions, followed by gaussian smoothing.

    :param dinf_model:
        DinfModel object that describes the dinf model.
    :param iterations:
        Number of GAN iterations.
    :param training_replicates:
        Size of the dataset used to train the discriminator.
        This dataset is constructed once each GAN iteration.
    :param test_replicates:
        Size of the dataset used to evaluate the discriminator after
        each training epoch. This dataset is constructed once before the
        GAN iterates, and is reused in each iteration.
    :param proposal_replicates:
        Number of ABC proposals in each iteration.
    :param epochs:
        Number of full passes over the training dataset when training
        the discriminator.
    :param top_n:
        If not None, do ABC rejection sampling in each iteraction by taking
        the ``top_n`` best samples, ranked by discriminator prediction.
        Samples are taken from the test replicates, so ``top_n`` must be
        smaller than ``test_replicates``.
    :param output_folder:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`DinfModel.generator_func` and
        :meth:`DinfModel.target_func`.
    :param seed:
        Seed for the random number generator.
    """
    assert dinf_model.target_func is not None
    if callbacks is None:
        callbacks = {}
    assert all(
        k
        in (
            "test/generator/feature",
            "test/target/feature",
            "iteration",
            "train/generator/feature",
            "train/target/feature",
            "fit/epoch",
            "fit/train_batch",
            "fit/test_batch",
            "proposal/feature",
            "predict/batch",
        )
        for k in callbacks
    )

    if top_n is not None and top_n >= proposal_replicates:
        raise ValueError(f"{top_n=}, but {proposal_replicates=}")

    if output_folder is None:
        output_folder = "."
    store = Store(output_folder, create=True)
    store.assert_complete(["discriminator.nn", "abc.npz"])
    resume = len(store) > 0

    ss = np.random.SeedSequence(seed, spawn_key=spawn_key("abc-gan"))
    ss_loop, ss_thetas, ss_test = ss.spawn(3)
    rng_thetas = np.random.default_rng(ss_thetas)

    parameters = dinf_model.parameters

    if resume:
        discriminator = Discriminator.from_file(
            store[-1] / "discriminator.nn", network=dinf_model.discriminator_network
        )
        thetas, y = _load_results_unstructured(
            store[-1] / "abc.npz", parameters=parameters
        )
        assert len(thetas.shape) == 2
        if top_n is not None:
            thetas, y = parameters.top_n(thetas, probs=y, n=top_n)
        training_thetas = parameters.sample_kde(
            thetas,
            probs=y,
            size=training_replicates // 2,
            rng=rng_thetas,
        )
        proposal_thetas = parameters.sample_kde(
            thetas,
            probs=y,
            size=proposal_replicates,
            rng=rng_thetas,
        )
    else:
        discriminator = Discriminator(network=dinf_model.discriminator_network)
        training_thetas = parameters.sample_prior(
            size=training_replicates // 2, rng=rng_thetas
        )
        proposal_thetas = parameters.sample_prior(
            size=proposal_replicates, rng=rng_thetas
        )

    n_target_calls = 0
    n_generator_calls = 0

    with process_pool(parallelism, dinf_model) as pool:

        val_x, val_y = None, None
        if test_replicates >= 2:
            ss_val, ss_test_thetas = ss_test.spawn(2)
            test_thetas = parameters.sample_prior(
                size=test_replicates // 2, rng=np.random.default_rng(ss_test_thetas)
            )
            val_x, val_y = _get_combined_dataset(
                target=dinf_model.target_func,
                generator=dinf_model.generator_func_v,
                thetas=test_thetas,
                pool=pool,
                ss=ss_val,
                callbacks={
                    "generator/feature": callbacks.get("test/generator/feature"),
                    "target/feature": callbacks.get("test/target/feature"),
                },
            )
            n_target_calls += test_replicates // 2
            n_generator_calls += test_replicates // 2

        if (cb_iter := callbacks.get("iteration")) is not None:
            cb_iter(len(store))

        for _ in range(iterations):

            ss_train, ss_proposals, ss_fit = ss_loop.spawn(3)
            train_x, train_y = _get_combined_dataset(
                target=dinf_model.target_func,
                generator=dinf_model.generator_func_v,
                thetas=training_thetas,
                pool=pool,
                ss=ss_train,
                callbacks={
                    "generator/feature": callbacks.get("train/generator/feature"),
                    "target/feature": callbacks.get("train/target/feature"),
                },
            )
            n_target_calls += training_replicates // 2
            n_generator_calls += training_replicates // 2

            discriminator.fit(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                epochs=epochs,
                ss=ss_fit,
                # Clear the training loss/accuracy metrics from last iteration.
                reset_metrics=True,
                callbacks={
                    "epoch": callbacks.get("fit/epoch"),
                    "train_batch": callbacks.get("fit/train_batch"),
                    "test_batch": callbacks.get("fit/test_batch"),
                },
            )
            store.increment()
            discriminator.to_file(store[-1] / "discriminator.nn")

            proposal_x = _get_generator_dataset(
                generator=dinf_model.generator_func_v,
                thetas=proposal_thetas,
                pool=pool,
                ss=ss_proposals,
                callbacks={
                    "feature": callbacks.get("proposal/feature"),
                },
            )
            n_generator_calls += proposal_replicates

            y = discriminator.predict(
                proposal_x,
                callbacks={"batch": callbacks.get("predict/batch")},
            )
            save_results(
                store[-1] / "abc.npz",
                probs=y,
                thetas=proposal_thetas,
                parameters=parameters,
            )

            # Get the posterior sample for the next iteration.
            thetas = proposal_thetas
            if top_n is not None:
                thetas, y = parameters.top_n(thetas, probs=y, n=top_n)
            training_thetas = parameters.sample_kde(
                thetas,
                probs=y,
                size=training_replicates // 2,
                rng=rng_thetas,
            )
            proposal_thetas = parameters.sample_kde(
                thetas,
                probs=y,
                size=proposal_replicates,
                rng=rng_thetas,
            )

            logger.info("Target called %s times.", n_target_calls)
            logger.info("Generator called %s times.", n_generator_calls)

            if (cb_iter := callbacks.get("iteration")) is not None:
                cb_iter(len(store))


def pretraining_pg_gan(
    *,
    dinf_model,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    pool,
    max_pretraining_iterations: int,
    ss: np.random.SeedSequence,
):
    """
    Pretraining roughly like PG-GAN.

    PG-GAN starts at a point in the parameter space that is favourable
    to the discriminator, and it only trains the discriminator enough
    to identify such a point. PG-GAN does up to 10 iterations, with
    40000/40000 (real/fake) reps each time. Here, we iterate up to
    max_pretraining_iterations times, each with training_replicates reps.
    """

    ss_thetas, ss_train = ss.spawn(2)
    rng_thetas = np.random.default_rng(ss_thetas)

    discriminator = Discriminator(network=dinf_model.discriminator_network)
    acc_best = 0
    theta_best = None

    for k in range(max_pretraining_iterations):
        theta = dinf_model.parameters.sample_prior(size=1, rng=rng_thetas)[0]
        training_thetas = np.tile(theta, (training_replicates // 2, 1))
        test_thetas = np.tile(theta, (test_replicates // 2, 1))

        metrics = _train_discriminator(
            discriminator=discriminator,
            dinf_model=dinf_model,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            pool=pool,
            ss=ss_train,
        )
        # Use the test accuracy, unless there's no test data.
        acc = metrics.get("test_accuracy", metrics["train_accuracy"])
        if acc > acc_best:
            theta_best = theta
        if acc > 0.9:
            break

    theta = theta_best

    num_replicates = training_replicates // 2 + test_replicates // 2
    n_target_calls = k * num_replicates
    n_generator_calls = k * num_replicates

    return discriminator, theta, n_target_calls, n_generator_calls


def pretraining_dinf(
    *,
    dinf_model,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    pool,
    max_pretraining_iterations: int,
    ss: np.random.SeedSequence,
):
    """
    Train the discriminator on data sampled from the prior, until it can
    distinguish randomly chosen points from the target dataset with
    accuracy 0.9, or until max_pretraining_iterations is exhausted.
    In practice, 10,000--20,000 training instances are sufficient.

    After pretraining, we choose the starting point to be the point
    with the highest log probability from a fresh set of candidates
    drawn from the prior.
    """
    ss_thetas, ss_lp, ss_train = ss.spawn(3)
    rng = np.random.default_rng(ss_thetas)

    parameters = dinf_model.parameters
    discriminator = Discriminator(network=dinf_model.discriminator_network)

    for k in range(max_pretraining_iterations):
        training_thetas = parameters.sample_prior(
            size=training_replicates // 2, rng=rng
        )
        test_thetas = parameters.sample_prior(size=test_replicates // 2, rng=rng)

        metrics = _train_discriminator(
            discriminator=discriminator,
            dinf_model=dinf_model,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            pool=pool,
            ss=ss_train,
        )

        # Use the test accuracy, unless there's no test data.
        acc = metrics.get("test_accuracy", metrics["train_accuracy"])
        if acc > 0.9:
            break

    thetas = parameters.sample_prior(size=training_replicates, rng=rng)
    lp = _log_prob(
        thetas,
        discriminator=discriminator,
        generator=dinf_model.generator_func_v,
        parameters=parameters,
        num_replicates=1,
        pool=pool,
        ss=ss_lp,
    )
    best = np.argmax(lp)
    theta = thetas[best]

    num_replicates = training_replicates // 2 + test_replicates // 2
    n_target_calls = k * num_replicates
    n_generator_calls = k * num_replicates + len(thetas)

    return discriminator, theta, n_target_calls, n_generator_calls


def sanneal_proposals_pg_gan(
    *, theta, temperature, rng, num_proposals, parameters, proposal_stddev
):
    """
    Proposals like PG-GAN.

    From Wang et al., pg 8:
        During each main training iteration, we choose 10 independent proposals
        for each parameter, keeping the other parameters fixed. This creates
        10 × P possible parameter sets, where P is the number of parameters
        (P = 6 for the IM model). We select the set that minimizes the gen-
        erator loss, which has the effect of modifying one parameter each
        iteration.
    """
    num_params = len(parameters)
    proposal_thetas = np.tile(theta, (num_params, num_proposals, 1))
    for j, (p, val) in enumerate(zip(parameters.values(), theta)):
        sd = temperature * proposal_stddev * (p.high - p.low)
        new_vals = rng.normal(val, scale=sd, size=num_proposals)
        # Clamp values within the bounds.
        new_vals = np.minimum(np.maximum(new_vals, p.low), p.high)
        proposal_thetas[j, :, j] = new_vals
    proposal_thetas = np.reshape(proposal_thetas, (-1, num_params))
    # Include the original theta as the first entry in the proposals.
    proposal_thetas = np.concatenate(([theta], proposal_thetas))
    return proposal_thetas


def sanneal_proposals_rr(
    *, theta, temperature, rng, iteration, num_proposals, parameters, proposal_stddev
):
    """
    Round-robin proposals. Varies only one param at a time.
    """
    num_params = len(parameters)
    which_param = next(iteration) % num_params
    p = list(parameters.values())[which_param]
    proposal_thetas = np.tile(theta, (num_proposals * num_params, 1))
    assert proposal_thetas.shape == (num_proposals * num_params, num_params)

    sd = temperature * proposal_stddev * (p.high - p.low)
    new_vals = rng.normal(theta[which_param], scale=sd, size=num_proposals * num_params)
    # Clamp values within the bounds.
    new_vals = np.minimum(np.maximum(new_vals, p.low), p.high)
    proposal_thetas[:, which_param] = new_vals
    # Include the original theta as the first entry in the proposals.
    proposal_thetas = np.concatenate(([theta], proposal_thetas))
    return proposal_thetas


def sanneal_proposals_mvn(
    *, theta, temperature, rng, num_proposals, parameters, proposal_stddev
):
    """
    Multivariate-normal proposals. Varies all params each time.
    """
    num_params = len(parameters)
    variances = [
        # Match the move distance with the other proposal methods.
        (temperature * proposal_stddev * (p.high - p.low)) ** 2 / num_params
        for p in parameters.values()
    ]
    cov = variances * np.eye(num_params)
    proposal_thetas = rng.multivariate_normal(
        theta, cov, size=num_proposals * num_params
    )
    # Clamp values within the bounds.
    for j, p in enumerate(parameters.values()):
        proposal_thetas[:, j] = np.minimum(
            np.maximum(proposal_thetas[:, j], p.low), p.high
        )
    # Include the original theta as the first entry in the proposals.
    proposal_thetas = np.concatenate(([theta], proposal_thetas))
    return proposal_thetas


def pg_gan(
    *,
    dinf_model: DinfModel,
    iterations: int,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    Dx_replicates: int,
    num_proposals: int = 10,
    proposal_stddev: float = 1 / 15,  # multiplied by the domain's width
    pretraining_method: str = "dinf",
    proposals_method: str = "rr",
    max_pretraining_iterations: int = 100,
    output_folder: None | str | pathlib.Path = None,
    parallelism: None | int = None,
    seed: None | int = None,
):
    """
    PG-GAN style simulated annealing.

    Wang et al. 2021, https://doi.org/10.1111/1755-0998.13386

    :param dinf_model:
        DinfModel object that describes the dinf model.
    :param iterations:
        Number of GAN iterations.
    :param training_replicates:
        Size of the dataset used to train the discriminator.
        This dataset is constructed once each GAN iteration.
    :param test_replicates:
        Size of the test dataset used to evaluate the discriminator after
        each training epoch.
    :param epochs:
        Number of full passes over the training dataset when training
        the discriminator.
    :param Dx_replicates:
        Number of generator replicates for approximating E[D(x)|θ].
    :param num_proposals:
        Number proposals per parameter in each iteration.
    :param proposal_stddev:
        The standard deviation for the proposal distribution is:
        temperature * proposal_stddev * (param.high - param.low)
        The default is 1 / 15, as used by PG-GAN.
    :param pretraining_method:
        A string indicating which method should be used to pretrain the
        discriminator and choose a starting point in parameter space.

         * "dinf" (*default*):
           Train the discriminator on data sampled from the prior, until it can
           distinguish randomly chosen points from the target dataset with
           accuracy 0.9, or until ``max_pretraining_iterations`` is exhausted.
           In practice, 10,000--20,000 training instances are sufficient.

           After pretraining, we choose the starting point to be the point
           with the highest log probability from a fresh set of candidates
           drawn from the prior.

         * "pg-gan":
           PG-GAN starts at a point in the parameter space that is favourable
           to the discriminator, and it only trains the discriminator enough
           to identify such a point.

           In each iteration we train the discriminator on a single point
           sampled from the prior. If the discriminator can distinguish between
           that point and the target dataset with accuracy 0.9, this point is
           used as the starting point.
           Otherwise, a new point is chosen by sampling from the prior.
           If ``max_pretraining_iterations`` is exhausted without reaching an
           accuracy of 0.9, the point at which the discriminator had the
           highest accuracy is used as the starting point.

    :param proposals_method:
        A string indicating which method should be used to produce the proposal
        distribution. In each case, ``num_proposals`` × P proposals are
        produced, where P is the number of parameters (to match original
        PG-GAN behaviour).

         * "rr" (*default*):
           Round-robin proposals. Proposals are generated for one parameter
           only, keeping the other parameters fixed.  A different parameter is
           chosen in each iteration, cycling through all parameters in order.
           Proposals are drawn from a normal distribution with standard deviation:
           temperature × ``proposal_stddev`` × (param.high - param.low).

         * "mvn":
           Multivariate-normal proposals.
           Proposals are drawn from a multivariate normal distribution with
           covariance matrix defined to have diagonals:
           (temperature × ``proposal_stddev`` × (param.high - param.low))**2 / P,
           where P is the number of parameters and the high/low bounds vary by
           the corresponding parameter. Off-diagonals of the covariance matrix
           are set to zero.

         * "pg-gan":
           Proposals like PG-GAN.
           ``num_proposals`` proposals are generated for each parameter,
           keeping the other parameters fixed, thus producing
           ``num_proposals`` × P total proposals (for P parameters).
           Proposals are drawn from a normal distribution with standard deviation:
           temperature × ``proposal_stddev`` × (param.high - param.low).

    :param max_pretraining_iterations:
        The maximum number of pretraining iterations.
    :param output_folder:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`DinfModel.generator_func` and
        :meth:`DinfModel.target_func`.
    :param seed:
        Seed for the random number generator.
    """
    assert dinf_model.target_func is not None
    num_replicates = training_replicates // 2 + test_replicates // 2

    if output_folder is None:
        output_folder = "."
    store = Store(output_folder, create=True)

    if proposals_method == "pg-gan":
        proposals_func = sanneal_proposals_pg_gan
    elif proposals_method == "rr":
        proposals_func = functools.partial(
            sanneal_proposals_rr, iteration=itertools.count()
        )
    elif proposals_method == "mvn":
        proposals_func = sanneal_proposals_mvn
    else:
        raise ValueError(f"unknown proposals_method {proposals_method}")

    if pretraining_method == "pg-gan":
        pretraining_func = pretraining_pg_gan
    elif pretraining_method == "dinf":
        pretraining_func = pretraining_dinf
    else:
        raise ValueError(f"unknown pretraining_method {pretraining_method}")

    ss = np.random.SeedSequence(seed, spawn_key=spawn_key("pg-gan"))
    ss_accept, ss_pretraining, ss_proposals, ss_loop = ss.spawn(4)
    rng_accept = np.random.default_rng(ss_accept)
    rng_proposals = np.random.default_rng(ss_proposals)

    parameters = dinf_model.parameters
    resume = len(store) > 0
    store.assert_complete(["discriminator.nn", "pg-gan-proposals.npz"])

    if resume:
        discriminator = Discriminator.from_file(
            store[-1] / "discriminator.nn", network=dinf_model.discriminator_network
        )
        proposal_thetas, probs = _load_results_unstructured(
            store[-1] / "pg-gan-proposals.npz", parameters=parameters
        )
        assert len(proposal_thetas.shape) == 2
        lp = np.log(probs)
        # First entry is for the current theta in the simulated annealing chain.
        theta = proposal_thetas[0]
        current_lp = lp[0]
        best = 1 + np.argmax(lp[1:])
        best_lp = lp[best]
        U = rng_accept.uniform()
        if best_lp > current_lp or U < (current_lp / best_lp):
            theta = proposal_thetas[best]

    n_target_calls = 0
    n_generator_calls = 0

    with process_pool(parallelism, dinf_model) as pool:

        if not resume:
            # pre-training
            discriminator, theta, n_target_calls, n_generator_calls = pretraining_func(
                dinf_model=dinf_model,
                training_replicates=training_replicates,
                test_replicates=test_replicates,
                epochs=epochs,
                pool=pool,
                max_pretraining_iterations=max_pretraining_iterations,
                ss=ss_pretraining,
            )

        for i in range(len(store), len(store) + iterations):
            logger.info("PG-GAN simulated annealing iteration %s", i)

            ss_lp, ss_train = ss_loop.spawn(2)
            temperature = max(0.02, 1.0 - i / iterations)

            proposal_thetas = proposals_func(
                theta=theta,
                temperature=temperature,
                rng=rng_proposals,
                num_proposals=num_proposals,
                parameters=parameters,
                proposal_stddev=proposal_stddev,
            )

            lp = _log_prob(
                proposal_thetas,
                discriminator=discriminator,
                generator=dinf_model.generator_func_v,
                parameters=parameters,
                num_replicates=Dx_replicates,
                pool=pool,
                ss=ss_lp,
            )
            n_generator_calls += len(proposal_thetas) * Dx_replicates

            store.increment()
            save_results(
                store[-1] / "pg-gan-proposals.npz",
                thetas=proposal_thetas,
                probs=np.exp(lp),
                parameters=parameters,
            )

            current_lp = lp[0]  # First entry is for the current theta.
            best = 1 + np.argmax(lp[1:])
            best_lp = lp[best]
            accept = False
            U = rng_accept.uniform()
            if best_lp > current_lp or U < (current_lp / best_lp) * temperature:
                # Accept the proposal.
                accept = True
                theta = proposal_thetas[best]
                current_lp = best_lp
                logger.info("Proposal accepted")
            else:
                logger.info("Proposal rejected")

            logger.info("log prob: %s", current_lp)

            for theta_i, (name, param) in zip(theta, parameters.items()):
                if param.truth:
                    logger.info("%s: %s (truth=%s)", name, theta_i, param.truth)
                else:
                    logger.info("%s: %s", name, theta_i)

            if accept:
                # Train.
                # PG-GAN does 5000/5000 (real/fake) reps.
                training_thetas = np.tile(theta, (training_replicates // 2, 1))
                test_thetas = np.tile(theta, (test_replicates // 2, 1))
                _train_discriminator(
                    discriminator=discriminator,
                    dinf_model=dinf_model,
                    training_thetas=training_thetas,
                    test_thetas=test_thetas,
                    epochs=epochs,
                    pool=pool,
                    ss=ss_train,
                    # XXX: Is this helpful?
                    entropy_regularisation=True,
                )
                n_target_calls += num_replicates
                n_generator_calls += num_replicates

            discriminator.to_file(store[-1] / "discriminator.nn")

            logger.info("Target called %s times.", n_target_calls)
            logger.info("Generator called %s times.", n_generator_calls)
