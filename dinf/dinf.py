from __future__ import annotations
import functools
import itertools
import logging
import math
import os
import pathlib
import signal
from typing import cast, Callable, Tuple
import zipfile

import emcee
import jax
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

# We're compatible with the standard lib's multiprocessing,
# but multiprocess uses dill to pickle functions which provides
# greater flexibility to users (i.e. fewer confusing errors).
import multiprocess as multiprocessing

from .discriminator import Discriminator, Surrogate
from .genobuilder import Genobuilder
from .parameters import Parameters
from .store import Store

logger = logging.getLogger(__name__)

_pool = None


def _initializer(filename):
    # Ignore ctrl-c in workers. This is handled in the main process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Ensure symbols from the user's genobuilder are avilable to workers.
    if filename is not None:
        Genobuilder.from_file(filename)


def _process_pool_init(parallelism, genobuilder):
    # Start the process pool before the GPU has been initialised, or using
    # the "spawn" start method, otherwise we get weird GPU resource issues
    # because the subprocesses are holding onto some CUDA thing.
    # We use multiprocessing, because concurrent.futures uses fork() on unix,
    # and the initial processes are forked on demand
    # (https://bugs.python.org/issue39207), which means they can be forked
    # after the GPU has been initialised.
    global _pool
    assert _pool is None
    ctx = multiprocessing.get_context("spawn")
    _pool = ctx.Pool(
        processes=parallelism,
        initializer=_initializer,
        initargs=(genobuilder._filename,),
    )


def cleanup_process_pool_afterwards(func):
    """
    A decorator for functions using the process pool, to do cleanup.
    """

    def wrapper(*args, **kwargs):
        global _pool
        terminate = False
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            terminate = True
            raise
        finally:
            if _pool is not None:
                if terminate:
                    _pool.terminate()
                else:
                    # close() instead of terminate() when reasonable to do so.
                    # https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
                    _pool.close()
                _pool.join()
                _pool = None

    return functools.update_wrapper(wrapper, func)


def _sim_replicates(*, sim_func, args, num_replicates, parallelism):
    if parallelism == 1:
        map_f = map
    else:
        global _pool
        assert _pool is not None
        map_f = _pool.imap

    result = None
    treedef = None
    for j, M in enumerate(map_f(sim_func, args)):
        if result is None:
            treedef = jax.tree_structure(M)
            result = []
            for m in jax.tree_leaves(M):
                result.append(np.empty((num_replicates, *m.shape), dtype=m.dtype))
        for res, m in zip(result, jax.tree_leaves(M)):
            res[j] = m
    return jax.tree_unflatten(treedef, result)


def _generate_data(*, generator, thetas, parallelism, rng):
    """
    Return generator output for randomly drawn parameter values.
    """
    num_replicates = len(thetas)
    seeds = rng.integers(low=1, high=2**31, size=num_replicates)
    data = _sim_replicates(
        sim_func=generator,
        args=zip(seeds, thetas),
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    return data


def _observe_data(*, target, num_replicates, parallelism, rng):
    """
    Return observations from the target dataset.
    """
    seeds = rng.integers(low=1, high=2**31, size=num_replicates)
    data = _sim_replicates(
        sim_func=target,
        args=seeds,
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    return data


def _generate_training_data(*, target, generator, thetas, parallelism, rng):
    num_replicates = len(thetas)
    x_generator = _generate_data(
        generator=generator, thetas=thetas, parallelism=parallelism, rng=rng
    )
    x_target = _observe_data(
        target=target, num_replicates=num_replicates, parallelism=parallelism, rng=rng
    )
    # XXX: Large copy doubles peak memory.
    x = jax.tree_map(lambda *l: np.concatenate(l), x_generator, x_target)
    del x_target
    y = np.concatenate((np.zeros(num_replicates), np.ones(num_replicates)))
    # Note: training data is not shuffled
    return x, y, x_generator


def save_results(
    filename: str | pathlib.Path,
    /,
    *,
    thetas: np.ndarray,
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
    if thetas.shape[-1] != len(parameters):
        raise ValueError(
            f"thetas.shape={thetas.shape}, but got {len(parameters)} parameters"
        )
    if thetas.shape[:-1] != probs.shape:
        raise ValueError(
            f"thetas.shape={thetas.shape}, but got probs.shape={probs.shape}"
        )
    assert "_Pr" not in parameters
    kw = {
        "_Pr": probs,
        **{par_name: thetas[..., j] for j, par_name in enumerate(parameters)},
    }
    np.savez(filename, **kw)


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


def _log_prob(
    thetas: np.ndarray,
    *,
    discriminator: Discriminator,
    generator: Callable,
    parameters: Parameters,
    rng: np.random.Generator,
    num_replicates: int,
    parallelism: int,
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

    seeds = rng.integers(low=1, high=2**31, size=num_replicates * num_in_bounds)
    thetas_reps = np.repeat(thetas[in_bounds], num_replicates, axis=0)
    assert len(seeds) == len(thetas_reps)
    M = _sim_replicates(
        sim_func=generator,
        args=zip(seeds, thetas_reps),
        num_replicates=len(seeds),
        parallelism=parallelism,
    )
    Dreps = discriminator.predict(M).reshape(num_in_bounds, num_replicates)
    D = np.mean(Dreps, axis=1)
    assert len(D) == num_in_bounds
    with np.errstate(divide="ignore"):
        log_D[in_bounds] = np.log(D)

    return log_D


def _log_prob_surrogate(
    thetas: np.ndarray,
    *,
    surrogate: Surrogate,
    parameters: Parameters,
) -> np.ndarray:
    """
    Function to be maximised by alfi mcmc.
    """
    num_walkers, num_params = thetas.shape
    assert num_params == len(parameters)
    log_D = np.full(num_walkers, -np.inf)

    # Identify workers with one or more out-of-bounds parameters.
    in_bounds = parameters.bounds_contain(thetas)
    num_in_bounds = np.sum(in_bounds)
    if num_in_bounds > 0:
        alpha, beta = surrogate.predict(thetas[in_bounds])
        with np.errstate(divide="ignore"):
            log_D[in_bounds] = np.log(alpha / (alpha + beta))
    return log_D


def _run_mcmc_emcee(
    start: np.ndarray,
    parameters: Parameters,
    walkers: int,
    steps: int,
    rng: np.random.Generator,
    log_prob_func,
):
    sampler = emcee.EnsembleSampler(
        walkers,
        len(parameters),
        log_prob_func,
        vectorize=True,
    )

    mt_initial_state = np.random.mtrand.RandomState(rng.integers(2**31)).get_state()
    state = emcee.State(start, random_state=mt_initial_state)
    sampler.run_mcmc(state, nsteps=steps)
    thetas = sampler.get_chain()
    assert thetas.shape == (steps, walkers, len(parameters))

    with np.errstate(over="ignore"):
        probs = np.exp(sampler.get_log_prob())
    assert probs.shape == (steps, walkers)

    # XXX: use logger
    print("MCMC acceptance rate", np.mean(sampler.acceptance_fraction))

    return thetas, probs


def _train_discriminator(
    *,
    discriminator: Discriminator,
    genobuilder: Genobuilder,
    training_thetas: np.ndarray,
    test_thetas: np.ndarray,
    epochs: int,
    parallelism: int,
    rng: np.random.Generator,
    entropy_regularisation: bool = False,
):
    train_x, train_y, train_x_generator = _generate_training_data(
        target=genobuilder.target_func,
        generator=genobuilder.generator_func,
        thetas=training_thetas,
        parallelism=parallelism,
        rng=rng,
    )

    val_x, val_y, val_x_generator = None, None, None
    if test_thetas is not None and len(test_thetas) > 0:
        val_x, val_y, val_x_generator = _generate_training_data(
            target=genobuilder.target_func,
            generator=genobuilder.generator_func,
            thetas=test_thetas,
            parallelism=parallelism,
            rng=rng,
        )

    metrics = discriminator.fit(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        epochs=epochs,
        rng=rng,
        # Clear the training loss/accuracy metrics from last iteration.
        reset_metrics=True,
        # TODO
        # tensorboard_log_dir=working_directory / "tensorboard" / "fit",
        entropy_regularisation=entropy_regularisation,
    )

    return metrics, train_x_generator, val_x_generator


def _train_surrogate(
    *,
    discriminator: Discriminator,
    surrogate: Surrogate,
    training_thetas: np.ndarray,
    test_thetas: np.ndarray,
    train_x_generator,
    test_x_generator,
    epochs: int,
):
    train_y_pred = discriminator.predict(train_x_generator)
    val_x_thetas = None
    val_y_pred = None
    if test_thetas is not None and len(test_thetas) > 0:
        val_x_thetas = test_thetas
        val_y_pred = discriminator.predict(test_x_generator)
    surrogate.fit(
        train_x=training_thetas,
        train_y=train_y_pred,
        val_x=val_x_thetas,
        val_y=val_y_pred,
        epochs=epochs,
    )

    alpha, beta = surrogate.predict(training_thetas)
    return train_y_pred, alpha, beta


@cleanup_process_pool_afterwards
def train(
    *,
    genobuilder: Genobuilder,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    parallelism: None | int = None,
    rng: np.random.Generator,
) -> Discriminator:
    """
    Train a discriminator network.

    :param genobuilder:
        Genobuilder object that describes the GAN.
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
        :meth:`Genobuilder.generator_func` and
        :meth:`Genobuilder.target_func`.
    :param numpy.random.Generator rng:
        Numpy random number generator.
    :return:
        The trained discriminator.
    """
    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, genobuilder)

    discriminator = Discriminator.from_input_shape(
        genobuilder.feature_shape, rng, genobuilder.discriminator_network
    )
    # discriminator.summary()

    training_thetas = genobuilder.parameters.draw_prior(
        training_replicates // 2, rng=rng
    )
    test_thetas = genobuilder.parameters.draw_prior(test_replicates // 2, rng=rng)
    _train_discriminator(
        discriminator=discriminator,
        genobuilder=genobuilder,
        training_thetas=training_thetas,
        test_thetas=test_thetas,
        epochs=epochs,
        parallelism=parallelism,
        rng=rng,
    )
    return discriminator


@cleanup_process_pool_afterwards
def predict(
    *,
    discriminator: Discriminator,
    genobuilder: Genobuilder,
    replicates: int,
    parallelism: None | int = None,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample generator features and make predictions using the discriminator.

    :param genobuilder:
        Genobuilder object that describes the GAN.
    :param replicates:
        Number of features to extract.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`Genobuilder.generator_func` and
        :meth:`Genobuilder.target_func`.
    :param numpy.random.Generator rng:
        Numpy random number generator.
    :return:
        A 2-tuple of (thetas, probs), where ``thetas`` are the drawn parameters
        and ``probs`` are the discriminator predictions.
        theta[j][k] is the j'th draw for the k'th parameter, and
        probs[j] is the discriminator prediction for the j'th draw.
    """
    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, genobuilder)

    thetas = genobuilder.parameters.draw_prior(replicates, rng=rng)
    x = _generate_data(
        generator=genobuilder.generator_func,
        thetas=thetas,
        parallelism=parallelism,
        rng=rng,
    )
    probs = discriminator.predict(x)
    return thetas, probs


@cleanup_process_pool_afterwards
def mcmc_gan(
    *,
    genobuilder: Genobuilder,
    iterations: int,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    walkers: int,
    steps: int,
    Dx_replicates: int,
    working_directory: None | str | pathlib.Path = None,
    parallelism: None | int = None,
    rng: np.random.Generator,
):
    """
    Run the MCMC GAN.

    Each iteration of the GAN can be conceptually divided into parts:

      - construct train/test datasets for the discriminator,
      - train the discriminator for a certain number of epochs,
      - run the MCMC.

    In the first iteration, the parameter values given to the generator
    to produce the test/train datasets are drawn from the parameters' prior
    distribution. In subsequent iterations, the parameter values are drawn
    by sampling without replacement from the previous iteration's MCMC chains.

    :param genobuilder:
        Genobuilder object that describes the GAN.
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
    :param working_directory:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`Genobuilder.generator_func` and
        :meth:`Genobuilder.target_func`.
    :param numpy.random.Generator rng:
        Numpy random number generator.
    """
    num_replicates = math.ceil((training_replicates + test_replicates) / 2)
    if steps * walkers < num_replicates:
        raise ValueError(
            f"Insufficient MCMC samples (steps * walkers = {steps * walkers}) "
            "for training the discriminator: "
            f"(training_replicates + test_replicates) / 2 = {num_replicates}"
        )

    if working_directory is None:
        working_directory = "."
    store = Store(working_directory, create=True)

    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, genobuilder)
    parameters = genobuilder.parameters
    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists() for fn in ("discriminator.pkl", "mcmc.npz")
        ]
        if sum(files_exist) == 1:
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    if resume:
        discriminator = Discriminator.from_file(store[-1] / "discriminator.pkl")
        thetas, _ = _load_results_unstructured(
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

        sampled_thetas = rng.choice(
            thetas.reshape(-1, thetas.shape[-1]), size=num_replicates, replace=False
        )
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]
    else:
        discriminator = Discriminator.from_input_shape(
            genobuilder.feature_shape, rng, genobuilder.discriminator_network
        )
        # Starting point for the mcmc chain.
        start = parameters.draw_prior(walkers, rng=rng)

        training_thetas = parameters.draw_prior(training_replicates // 2, rng=rng)
        test_thetas = parameters.draw_prior(test_replicates // 2, rng=rng)

    # If start values are linearly dependent, emcee complains loudly.
    assert not np.any((start[0] == start[1:]).all(axis=-1))

    n_target_calls = 0
    n_generator_calls = 0

    log_prob_func = functools.partial(
        _log_prob,
        discriminator=discriminator,
        generator=genobuilder.generator_func,
        parameters=parameters,
        parallelism=parallelism,
        num_replicates=Dx_replicates,
        rng=rng,
    )

    for i in range(len(store) + 1, len(store) + 1 + iterations):
        print(f"MCMC GAN iteration {i}")
        store.increment()

        _train_discriminator(
            discriminator=discriminator,
            genobuilder=genobuilder,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            rng=rng,
        )
        discriminator.to_file(store[-1] / "discriminator.pkl")

        thetas, probs = _run_mcmc_emcee(
            start=start,
            parameters=parameters,
            walkers=walkers,
            steps=steps,
            rng=rng,
            log_prob_func=log_prob_func,
        )
        assert thetas.shape == (steps, walkers, len(parameters))
        save_results(
            store[-1] / "mcmc.npz", thetas=thetas, probs=probs, parameters=parameters
        )

        # The chain for next iteration starts at the end of this chain.
        start = thetas[-1]

        sampled_thetas = rng.choice(
            thetas.reshape(-1, thetas.shape[-1]), size=num_replicates, replace=False
        )
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]

        n_target_calls += num_replicates
        n_generator_calls += num_replicates + walkers * steps * Dx_replicates
        print(f"Target called {n_target_calls} times.")
        print(f"Generator called {n_generator_calls} times.")


def _run_abc(
    *,
    discriminator: Discriminator,
    generator: Callable,
    parameters: Parameters,
    thetas: np.ndarray,
    posteriors: int,
    parallelism: int,
    rng: np.random.Generator,
):
    x = _generate_data(
        generator=generator, thetas=thetas, parallelism=parallelism, rng=rng
    )
    y = discriminator.predict(x)
    top = np.argsort(y)[::-1][:posteriors]
    return thetas[top], y[top]


@cleanup_process_pool_afterwards
def abc_gan(
    *,
    genobuilder: Genobuilder,
    iterations: int,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    proposals: int,
    posteriors: int,
    working_directory: None | str | pathlib.Path = None,
    parallelism: None | int = None,
    rng: np.random.Generator,
):
    """
    Run the ABC GAN.

    Each iteration of the GAN can be conceptually divided into:

      - constructing train/test datasets for the discriminator,
      - training the discriminator for a certain number of epochs,
      - running the ABC.

    In the first iteration, the parameter values given to the generator
    to produce the test/train datasets are drawn from the parameters' prior
    distribution. In subsequent iterations, the parameter values are drawn
    by sampling with replacement from the previous iteration's ABC posterior.

    :param genobuilder:
        Genobuilder object that describes the dinf model.
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
    :param proposals:
        Number of ABC sample draws.
    :param posteriors:
        Number of top-ranked ABC sample draws to keep.
    :param working_directory:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`Genobuilder.generator_func` and
        :meth:`Genobuilder.target_func`.
    :param numpy.random.Generator rng:
        Numpy random number generator.
    """
    if posteriors > proposals:
        raise ValueError(
            f"Cannot subsample {posteriors} posteriors from {proposals} proposals"
        )

    num_replicates = math.ceil((training_replicates + test_replicates) / 2)

    if working_directory is None:
        working_directory = "."
    store = Store(working_directory, create=True)

    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, genobuilder)

    parameters = genobuilder.parameters
    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists() for fn in ("discriminator.pkl", "abc.npz")
        ]
        if sum(files_exist) == 1:
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    if resume:
        discriminator = Discriminator.from_file(store[-1] / "discriminator.pkl")
        thetas, _ = _load_results_unstructured(
            store[-1] / "abc.npz", parameters=parameters
        )
        assert len(thetas.shape) == 2
        sampled_thetas = rng.choice(thetas, size=num_replicates, replace=True)
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]
    else:
        discriminator = Discriminator.from_input_shape(
            genobuilder.feature_shape, rng, genobuilder.discriminator_network
        )

        training_thetas = parameters.draw_prior(training_replicates // 2, rng=rng)
        test_thetas = parameters.draw_prior(test_replicates // 2, rng=rng)

    n_target_calls = 0
    n_generator_calls = 0

    for i in range(len(store) + 1, len(store) + 1 + iterations):
        print(f"ABC GAN iteration {i}")
        store.increment()

        _train_discriminator(
            discriminator=discriminator,
            genobuilder=genobuilder,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            rng=rng,
        )
        discriminator.to_file(store[-1] / "discriminator.pkl")

        proposal_thetas = parameters.draw_prior(proposals, rng=rng)
        thetas, probs = _run_abc(
            discriminator=discriminator,
            generator=genobuilder.generator_func,
            parameters=parameters,
            thetas=proposal_thetas,
            posteriors=posteriors,
            parallelism=parallelism,
            rng=rng,
        )
        save_results(
            store[-1] / "abc.npz", probs=probs, thetas=thetas, parameters=parameters
        )

        sampled_thetas = rng.choice(thetas, size=num_replicates, replace=True)
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]

        n_target_calls += num_replicates
        n_generator_calls += num_replicates + proposals
        print(f"Target called {n_target_calls} times.")
        print(f"Generator called {n_generator_calls} times.")


def pretraining_pg_gan(
    *,
    genobuilder,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    parallelism: int,
    max_pretraining_iterations: int,
    rng,
):
    """
    Pretraining roughly like PG-GAN.

    PG-GAN starts at a point in the parameter space that is favourable
    to the discriminator, and it only trains the discriminator enough
    to identify such a point. PG-GAN does up to 10 iterations, with
    40000/40000 (real/fake) reps each time. Here, we iterate up to
    max_pretraining_iterations times, each with training_replicates reps.
    """

    discriminator = Discriminator.from_input_shape(
        genobuilder.feature_shape, rng, genobuilder.discriminator_network
    )
    acc_best = 0
    theta_best = None

    for k in range(max_pretraining_iterations):
        theta = genobuilder.parameters.draw_prior(1, rng=rng)[0]
        training_thetas = np.tile(theta, (training_replicates // 2, 1))
        test_thetas = np.tile(theta, (test_replicates // 2, 1))

        metrics, _, _ = _train_discriminator(
            discriminator=discriminator,
            genobuilder=genobuilder,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            rng=rng,
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
    genobuilder,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    parallelism: int,
    max_pretraining_iterations: int,
    rng,
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

    parameters = genobuilder.parameters
    discriminator = Discriminator.from_input_shape(
        genobuilder.feature_shape, rng, genobuilder.discriminator_network
    )

    for k in range(max_pretraining_iterations):
        training_thetas = parameters.draw_prior(training_replicates // 2, rng=rng)
        test_thetas = parameters.draw_prior(test_replicates // 2, rng=rng)

        metrics, _, _ = _train_discriminator(
            discriminator=discriminator,
            genobuilder=genobuilder,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            rng=rng,
        )

        # Use the test accuracy, unless there's no test data.
        acc = metrics.get("test_accuracy", metrics["train_accuracy"])
        if acc > 0.9:
            break

    thetas = parameters.draw_prior(training_replicates, rng=rng)
    lp = _log_prob(
        thetas,
        discriminator=discriminator,
        generator=genobuilder.generator_func,
        parameters=parameters,
        num_replicates=1,
        parallelism=parallelism,
        rng=rng,
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


@cleanup_process_pool_afterwards
def pg_gan(
    *,
    genobuilder: Genobuilder,
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
    working_directory: None | str | pathlib.Path = None,
    parallelism: None | int = None,
    rng: np.random.Generator,
):
    """
    PG-GAN style simulated annealing.

    Wang et al. 2021, https://doi.org/10.1111/1755-0998.13386

    :param genobuilder:
        Genobuilder object that describes the dinf model.
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
    :param working_directory:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`Genobuilder.generator_func` and
        :meth:`Genobuilder.target_func`.
    :param numpy.random.Generator rng:
        Numpy random number generator.
    """
    num_replicates = training_replicates // 2 + test_replicates // 2

    if working_directory is None:
        working_directory = "."
    store = Store(working_directory, create=True)

    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, genobuilder)

    parameters = genobuilder.parameters
    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists()
            for fn in ("discriminator.pkl", "pg-gan-proposals.npz")
        ]
        if sum(files_exist) == 1:
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    if resume:
        discriminator = Discriminator.from_file(store[-1] / "discriminator.pkl")
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
        U = rng.uniform()
        if best_lp > current_lp or U < (current_lp / best_lp):
            theta = proposal_thetas[best]

        # TODO: do something better here?
        n_target_calls = 0
        n_generator_calls = 0
    else:

        if pretraining_method == "pg-gan":
            pretraining_func = pretraining_pg_gan
        elif pretraining_method == "dinf":
            pretraining_func = pretraining_dinf
        else:
            raise ValueError(f"unknown pretraining_method {pretraining_method}")

        discriminator, theta, n_target_calls, n_generator_calls = pretraining_func(
            genobuilder=genobuilder,
            training_replicates=training_replicates,
            test_replicates=test_replicates,
            epochs=epochs,
            parallelism=parallelism,
            max_pretraining_iterations=max_pretraining_iterations,
            rng=rng,
        )

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

    for i in range(len(store) + 1, len(store) + 1 + iterations):
        print(f"PG-GAN simulated annealing iteration {i}")
        store.increment()

        temperature = max(0.02, 1.0 - i / iterations)

        proposal_thetas = proposals_func(
            theta=theta,
            temperature=temperature,
            rng=rng,
            num_proposals=num_proposals,
            parameters=parameters,
            proposal_stddev=proposal_stddev,
        )

        lp = _log_prob(
            proposal_thetas,
            discriminator=discriminator,
            generator=genobuilder.generator_func,
            parameters=parameters,
            num_replicates=Dx_replicates,
            parallelism=parallelism,
            rng=rng,
        )
        n_generator_calls += len(proposal_thetas) * Dx_replicates

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
        U = rng.uniform()
        if best_lp > current_lp or U < (current_lp / best_lp) * temperature:
            # Accept the proposal.
            accept = True
            theta = proposal_thetas[best]
            current_lp = best_lp
            print("Proposal accepted")
        else:
            print("Proposal rejected")

        print(f"log prob: {current_lp}")

        for theta_i, (name, param) in zip(theta, parameters.items()):
            if param.truth:
                print(f"{name}: {theta_i} (truth={param.truth})")
            else:
                print(f"{name}: {theta_i}")

        if accept:
            # Train.
            # PG-GAN does 5000/5000 (real/fake) reps.
            training_thetas = np.tile(theta, (training_replicates // 2, 1))
            test_thetas = np.tile(theta, (test_replicates // 2, 1))
            _train_discriminator(
                discriminator=discriminator,
                genobuilder=genobuilder,
                training_thetas=training_thetas,
                test_thetas=test_thetas,
                epochs=epochs,
                parallelism=parallelism,
                rng=rng,
                # XXX: Is this helpful?
                entropy_regularisation=True,
            )
            n_target_calls += num_replicates
            n_generator_calls += num_replicates

        discriminator.to_file(store[-1] / "discriminator.pkl")

        print(f"Target called {n_target_calls} times.")
        print(f"Generator called {n_generator_calls} times.")
        print()


@cleanup_process_pool_afterwards
def alfi_mcmc_gan(
    *,
    genobuilder: Genobuilder,
    iterations: int,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    walkers: int,
    steps: int,
    working_directory: None | str | pathlib.Path = None,
    parallelism: None | int = None,
    rng: np.random.Generator,
):
    """
    Run the ALFI MCMC GAN.

    This behaves similarly to :func:`mcmc_gan`, but introduces a surrogate
    neural network that predicts the output of the discriminator from a given
    set of input parameter values. The predictions of the surrogate network
    are used during MCMC sampling, thus bypassing the generator and producing
    deterministic classification of the input parameters.

    Kim et al. 2020, https://arxiv.org/abs/2004.05803v1

    :param genobuilder:
        Genobuilder object that describes the dinf model.
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
    :param working_directory:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`Genobuilder.generator_func` and
        :meth:`Genobuilder.target_func`.
    :param numpy.random.Generator rng:
        Numpy random number generator.
    """
    num_replicates = math.ceil((training_replicates + test_replicates) / 2)
    if steps * walkers < num_replicates:
        raise ValueError(
            f"Insufficient MCMC samples (steps * walkers = {steps * walkers}) "
            "for training the discriminator "
            f"((training_replicates + test_replicates / 2) = {num_replicates})"
        )

    if working_directory is None:
        working_directory = "."
    store = Store(working_directory, create=True)

    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, genobuilder)

    parameters = genobuilder.parameters
    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists()
            for fn in ("discriminator.pkl", "surrogate.pkl", "mcmc.npz")
        ]
        if sum(files_exist) not in (0, len(files_exist)):
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    if resume:
        discriminator = Discriminator.from_file(store[-1] / "discriminator.pkl")
        surrogate = Surrogate.from_file(store[-1] / "surrogate.pkl")
        thetas, _ = _load_results_unstructured(
            store[-1] / "mcmc.npz", parameters=parameters
        )
        assert len(thetas.shape) == 3
        # Discard first half as burn in.
        thetas = thetas[steps:]
        start = thetas[-1]
        if len(start) != walkers:
            # TODO: allow this by sampling start points for the walkers?
            raise ValueError(
                f"request for {walkers} walkers, but resuming from "
                f"{store[-1] / 'mcmc.npz'} which used {len(start)} walkers."
            )

        sampled_thetas = rng.choice(
            thetas.reshape(-1, thetas.shape[-1]), size=num_replicates, replace=False
        )
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]
    else:
        discriminator = Discriminator.from_input_shape(
            genobuilder.feature_shape, rng, genobuilder.discriminator_network
        )
        surrogate = Surrogate.from_input_shape(len(parameters), rng)
        # Starting point for the mcmc chain.
        start = parameters.draw_prior(walkers, rng=rng)

        training_thetas = parameters.draw_prior(training_replicates // 2, rng=rng)
        test_thetas = parameters.draw_prior(test_replicates // 2, rng=rng)

    # If start values are linearly dependent, emcee complains loudly.
    assert not np.any((start[0] == start[1:]).all(axis=-1))

    n_target_calls = 0
    n_generator_calls = 0

    log_prob_func = functools.partial(
        _log_prob_surrogate, surrogate=surrogate, parameters=parameters
    )

    for i in range(len(store) + 1, len(store) + 1 + iterations):
        print(f"ALFI MCMC GAN iteration {i}")
        store.increment()

        _, train_x_generator, test_x_generator = _train_discriminator(
            discriminator=discriminator,
            genobuilder=genobuilder,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            rng=rng,
        )
        n_target_calls += num_replicates
        n_generator_calls += num_replicates
        discriminator.to_file(store[-1] / "discriminator.pkl")

        train_y_pred, alpha, beta = _train_surrogate(
            discriminator=discriminator,
            surrogate=surrogate,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            train_x_generator=train_x_generator,
            test_x_generator=test_x_generator,
        )
        surrogate.to_file(store[-1] / "surrogate.pkl")

        # import pickle
        # with open(store[-1] / "s.pkl", "wb") as f:
        #    pickle.dump((training_thetas, train_y_pred, alpha, beta), f)

        s_pred = alpha / (alpha + beta)
        which = "D"
        for j in (np.argmax(train_y_pred), np.argmax(s_pred)):
            print(
                f"Best {which}: D(θ)={train_y_pred[j]:.3g}; "
                f"S(θ)={s_pred[j]}; "
                f"α={alpha[j]:.3g}, β={beta[j]:.3g}"
            )
            for param_name, value in zip(parameters, training_thetas[j]):
                print(" ", param_name, value)
            which = "S"

        thetas, probs = _run_mcmc_emcee(
            start=start,
            parameters=parameters,
            walkers=walkers,
            steps=2 * steps,
            rng=rng,
            log_prob_func=log_prob_func,
        )
        assert thetas.shape == (2 * steps, walkers, len(parameters))
        save_results(
            store[-1] / "mcmc.npz", thetas=thetas, probs=probs, parameters=parameters
        )

        # Discard first half as burn in.
        thetas = thetas[steps:]

        # The chain for next iteration starts at the end of this chain.
        start = thetas[-1]

        sampled_thetas = rng.choice(
            thetas.reshape(-1, thetas.shape[-1]), size=num_replicates, replace=False
        )
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]

        # n_target_calls += num_replicates
        # n_generator_calls += num_replicates
        print(f"Target called {n_target_calls} times.")
        print(f"Generator called {n_generator_calls} times.")
