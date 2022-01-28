from __future__ import annotations
import copy
import logging
import math
import multiprocessing
import os
import pathlib
import signal
import pickle
from typing import cast, Callable
import warnings

import arviz as az
import emcee
import jax
import numpy as np

from .discriminator import Discriminator, Surrogate
from .genobuilder import Genobuilder
from .parameters import Parameters
from .store import Store
from .mcmc import rw_mcmc

logger = logging.getLogger(__name__)

_pool = None


def _initializer(filename):
    # Ignore ctrl-c in workers. This is handled in the main process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Ensure symbols from the user's genobuilder are avilable to workers.
    if filename is not None:
        Genobuilder._from_file(filename)


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
            func(*args, **kwargs)
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

    return wrapper


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


def _generate_data(*, generator, parameters, num_replicates, parallelism, rng):
    """
    Return generator output for randomly drawn parameter values.
    """
    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates)
    params = parameters.draw(num_replicates=num_replicates, rng=rng)
    data = _sim_replicates(
        sim_func=generator,
        args=zip(seeds, params),
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    return params, data


def _observe_data(*, target, num_replicates, parallelism, rng):
    """
    Return observations from the target dataset.
    """
    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates)
    data = _sim_replicates(
        sim_func=target,
        args=seeds,
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    return data


def _generate_training_data(
    *, target, generator, parameters, num_replicates, parallelism, rng
):
    nreps_generator = num_replicates // 2
    nreps_target = num_replicates - num_replicates // 2
    _, x_generator = _generate_data(
        generator=generator,
        parameters=parameters,
        num_replicates=nreps_generator,
        parallelism=parallelism,
        rng=rng,
    )
    x_target = _observe_data(
        target=target,
        num_replicates=nreps_target,
        parallelism=parallelism,
        rng=rng,
    )
    # XXX: Large copy doubles peak memory.
    # Preallocate somehow?
    x = jax.tree_map(lambda *l: np.concatenate(l), x_generator, x_target)
    del x_generator
    del x_target
    y = np.concatenate((np.zeros(nreps_generator), np.ones(nreps_target)))
    # shuffle
    indexes = rng.permutation(num_replicates)
    # XXX: Large copy doubles peak memory.
    # Do this in-place?
    # https://stackoverflow.com/a/60902210/9500949
    x = jax.tree_map(lambda l: l[indexes], x)
    y = y[indexes]
    return x, y


def _mcmc_log_prob(
    theta: np.ndarray,
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
    num_walkers, num_params = theta.shape
    assert num_params == len(parameters)
    log_D = np.full(num_walkers, -np.inf)

    # Identify workers with one or more out-of-bounds parameters.
    in_bounds = parameters.bounds_contain(theta)
    num_in_bounds = np.sum(in_bounds)
    if num_in_bounds == 0:
        return log_D

    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates * num_in_bounds)
    params = np.repeat(theta[in_bounds], num_replicates, axis=0)
    assert len(seeds) == len(params)
    M = _sim_replicates(
        sim_func=generator,
        args=zip(seeds, params),
        num_replicates=len(seeds),
        parallelism=parallelism,
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
    genobuilder: Genobuilder,
    walkers: int,
    steps: int,
    Dx_replicates: int,
    parallelism: int,
    rng: np.random.Generator,
):
    sampler = emcee.EnsembleSampler(
        walkers,
        len(genobuilder.parameters),
        _mcmc_log_prob,
        vectorize=True,
        # kwargs passed to _mcmc_log_prob
        kwargs=dict(
            discriminator=discriminator,
            generator=genobuilder.generator_func,
            parameters=genobuilder.parameters,
            parallelism=parallelism,
            num_replicates=Dx_replicates,
            rng=rng,
        ),
    )

    mt_initial_state = np.random.mtrand.RandomState(rng.integers(2 ** 31)).get_state()
    state = emcee.State(start, random_state=mt_initial_state)
    sampler.run_mcmc(state, nsteps=steps)
    chain = sampler.get_chain()

    datadict = {
        "posterior": {
            p: chain[..., j].swapaxes(0, 1)
            for j, p in enumerate(genobuilder.parameters)
        },
        "sample_stats": {
            "lp": sampler.get_log_prob().swapaxes(0, 1),
            "acceptance_rate": np.mean(sampler.acceptance_fraction),
        },
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="More chains.*than draws",
            module="arviz",
        )
        dataset = az.from_dict(**datadict)

    return dataset


def _train_discriminator(
    *,
    discriminator: Discriminator,
    genobuilder: Genobuilder,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    parallelism: int,
    rng: np.random.Generator,
):
    train_x, train_y = _generate_training_data(
        target=genobuilder.target_func,
        generator=genobuilder.generator_func,
        parameters=genobuilder.parameters,
        num_replicates=training_replicates,
        parallelism=parallelism,
        rng=rng,
    )
    val_x, val_y = None, None
    if test_replicates > 0:
        val_x, val_y = _generate_training_data(
            target=genobuilder.target_func,
            generator=genobuilder.generator_func,
            parameters=genobuilder.parameters,
            num_replicates=test_replicates,
            parallelism=parallelism,
            rng=rng,
        )

    discriminator.fit(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        epochs=epochs,
        # Clear the training loss/accuracy metrics from last iteration.
        reset_metrics=True,
        # TODO
        # tensorboard_log_dir=working_directory / "tensorboard" / "fit",
    )


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
    by sampling with replacement from the previous iteration's MCMC chains.

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
    # We modify the parameters, so take a copy.
    genobuilder = copy.deepcopy(genobuilder)

    if working_directory is None:
        working_directory = "."
    store = Store(working_directory, create=True)

    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, genobuilder)

    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists() for fn in ("discriminator.pkl", "mcmc.ncf")
        ]
        if sum(files_exist) == 1:
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    if resume:
        discriminator = Discriminator.from_file(store[-1] / "discriminator.pkl")
        dataset = az.from_netcdf(store[-1] / "mcmc.ncf")
        chain = np.array(dataset.posterior.to_array()).swapaxes(0, 2)
        start = chain[-1]
        if len(start) != walkers:
            # TODO: allow this by sampling start points for the walkers?
            raise ValueError(
                f"request for {walkers} walkers, but resuming from "
                f"{store[-1] / 'mcmc.ncf'} which used {len(start)} walkers."
            )
        posterior_sample = chain.reshape(-1, chain.shape[-1])
        genobuilder.parameters = genobuilder.parameters.with_posterior(posterior_sample)
    else:
        discriminator = Discriminator.from_input_shape(genobuilder.feature_shape, rng)
        # Starting point for the mcmc chain.
        start = genobuilder.parameters.draw_prior(num_replicates=walkers, rng=rng)

    # If start values are linearly dependent, emcee complains loudly.
    assert not np.any((start[0] == start[1:]).all(axis=-1))

    n_observed_calls = 0
    n_generator_calls = 0

    for i in range(iterations):
        print(f"MCMC GAN iteration {i}")
        store.increment()

        _train_discriminator(
            discriminator=discriminator,
            genobuilder=genobuilder,
            training_replicates=training_replicates,
            test_replicates=test_replicates,
            epochs=epochs,
            parallelism=parallelism,
            rng=rng,
        )
        discriminator.to_file(store[-1] / "discriminator.pkl")

        dataset = _run_mcmc_emcee(
            start=start,
            discriminator=discriminator,
            genobuilder=genobuilder,
            walkers=walkers,
            steps=steps,
            Dx_replicates=Dx_replicates,
            parallelism=parallelism,
            rng=rng,
        )
        az.to_netcdf(dataset, store[-1] / "mcmc.ncf")

        chain = np.array(dataset.posterior.to_array()).swapaxes(0, 2)

        # The chain for next iteration starts at the end of this chain.
        start = chain[-1]

        # Update the parameters to draw from the posterior sample
        # (the merged chains from the mcmc).
        genobuilder.parameters = genobuilder.parameters.with_posterior(
            chain.reshape(-1, chain.shape[-1])
            # np.array(dataset.posterior.to_array()).swapaxes(0, 2)
        )

        n_observed_calls += (training_replicates + test_replicates) // 2
        n_generator_calls += (
            training_replicates + test_replicates
        ) // 2 + walkers * steps * Dx_replicates
        print(f"Observed data extracted {n_observed_calls} times.")
        print(f"Generator called {n_generator_calls} times.")


def _run_abc(
    *,
    discriminator: Discriminator,
    genobuilder: Genobuilder,
    proposals: int,
    posteriors: int,
    parallelism: int,
    rng: np.random.Generator,
):
    if posteriors > proposals:
        raise ValueError(
            f"Cannot subsample {posteriors} posteriors from {proposals} proposals"
        )

    params, x = _generate_data(
        generator=genobuilder.generator_func,
        parameters=genobuilder.parameters,
        num_replicates=proposals,
        parallelism=parallelism,
        rng=rng,
    )
    y = discriminator.predict(x)
    top = np.argsort(y)[::-1][:posteriors]
    top_params = params[top]
    with np.errstate(divide="ignore"):
        log_top_y = np.log(y[top])

    dataset = az.from_dict(
        posterior={p: top_params[..., j] for j, p in enumerate(genobuilder.parameters)},
        sample_stats={"lp": log_top_y},
    )
    return dataset


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
      - trainin the discriminator for a certain number of epochs,
      - running the ABC.

    In the first iteration, the parameter values given to the generator
    to produce the test/train datasets are drawn from the parameters' prior
    distribution. In subsequent iterations, the parameter values are drawn
    by sampling with replacement from the previous iteration's ABC posterior.

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
    # We modify the parameters, so take a copy.
    genobuilder = copy.deepcopy(genobuilder)

    if working_directory is None:
        working_directory = "."
    store = Store(working_directory, create=True)

    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, genobuilder)

    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists() for fn in ("discriminator.pkl", "abc.ncf")
        ]
        if sum(files_exist) == 1:
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    if resume:
        discriminator = Discriminator.from_file(store[-1] / "discriminator.pkl")
        dataset = az.from_netcdf(store[-1] / "abc.ncf")
        genobuilder.parameters = genobuilder.parameters.with_posterior(
            np.array(dataset.posterior.to_array()).swapaxes(0, 2).squeeze()
        )
    else:
        discriminator = Discriminator.from_input_shape(genobuilder.feature_shape, rng)

    n_observed_calls = 0
    n_generator_calls = 0

    for i in range(iterations):
        print(f"ABC GAN iteration {i}")
        store.increment()

        _train_discriminator(
            discriminator=discriminator,
            genobuilder=genobuilder,
            training_replicates=training_replicates,
            test_replicates=test_replicates,
            epochs=epochs,
            parallelism=parallelism,
            rng=rng,
        )
        discriminator.to_file(store[-1] / "discriminator.pkl")

        dataset = _run_abc(
            discriminator=discriminator,
            genobuilder=genobuilder,
            proposals=proposals,
            posteriors=posteriors,
            parallelism=parallelism,
            rng=rng,
        )
        az.to_netcdf(dataset, store[-1] / "abc.ncf")

        # Update to draw from the posterior sample in the next iteration.
        genobuilder.parameters = genobuilder.parameters.with_posterior(
            np.array(dataset.posterior.to_array()).swapaxes(0, 2).squeeze()
        )

        n_observed_calls += (training_replicates + test_replicates) // 2
        n_generator_calls += (training_replicates + test_replicates) // 2 + proposals
        print(f"Observed data extracted {n_observed_calls} times.")
        print(f"Generator called {n_generator_calls} times.")


def _mcmc_log_prob_alfi(
    theta: np.ndarray,
    *,
    surrogate: Surrogate,
    parameters: Parameters,
) -> np.ndarray:
    """
    Function to be maximised by mcmc. Vectorised version.
    """
    num_walkers, num_params = theta.shape
    assert num_params == len(parameters)
    log_D = np.full(num_walkers, -np.inf)

    # Identify workers with one or more out-of-bounds parameters.
    in_bounds = parameters.bounds_contain(theta)
    num_in_bounds = np.sum(in_bounds)
    if num_in_bounds > 0:
        alpha, beta = surrogate.predict(theta[in_bounds])
        with np.errstate(divide="ignore"):
            log_D[in_bounds] = np.log(alpha / (alpha + beta))
    return log_D


def _run_mcmc_emcee_alfi(
    start: np.ndarray,
    surrogate: Surrogate,
    parameters: Parameters,
    walkers: int,
    steps: int,
    rng: np.random.Generator,
):
    sampler = emcee.EnsembleSampler(
        walkers,
        len(parameters),
        _mcmc_log_prob_alfi,
        vectorize=True,
        #moves=emcee.moves.GaussianMove(10.0),
        # kwargs passed to _mcmc_log_prob_alfi
        kwargs=dict(surrogate=surrogate, parameters=parameters),
    )

    mt_initial_state = np.random.mtrand.RandomState(rng.integers(2 ** 31)).get_state()
    state = emcee.State(start, random_state=mt_initial_state)
    sampler.run_mcmc(state, nsteps=steps)
    chain = sampler.get_chain()

    datadict = {
        "posterior": {
            p: chain[..., j].swapaxes(0, 1)
            for j, p in enumerate(parameters)
        },
        "sample_stats": {
            "lp": sampler.get_log_prob().swapaxes(0, 1),
            "acceptance_rate": np.mean(sampler.acceptance_fraction),
        },
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="More chains.*than draws",
            module="arviz",
        )
        dataset = az.from_dict(**datadict)

    # XXX: use logger
    print("MCMC acceptance rate", np.mean(sampler.acceptance_fraction))

    return dataset


def _generate_data_alfi(*, generator, thetas, parallelism, rng):
    """
    Return generator output for randomly drawn parameter values.
    """
    num_replicates = len(thetas)
    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates)
    data = _sim_replicates(
        sim_func=generator,
        args=zip(seeds, thetas),
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    return data


def _generate_training_data_alfi(*, target, generator, thetas, parallelism, rng):
    x_generator = _generate_data_alfi(
        generator=generator,
        thetas=thetas,
        parallelism=parallelism,
        rng=rng,
    )
    x_target = _observe_data(
        target=target,
        num_replicates=len(thetas),
        parallelism=parallelism,
        rng=rng,
    )
    # XXX: Large copy doubles peak memory.
    # Preallocate somehow?
    x = jax.tree_map(lambda *l: np.concatenate(l), x_generator, x_target)
    # del x_generator
    del x_target
    y = np.concatenate((np.zeros(len(thetas)), np.ones(len(thetas))))
    # shuffle
    indexes = rng.permutation(len(y))
    # XXX: Large copy doubles peak memory.
    # Do this in-place?
    # https://stackoverflow.com/a/60902210/9500949
    x = jax.tree_map(lambda l: l[indexes], x)
    y = y[indexes]
    return x, y, x_generator


def _train_alfi(
    *,
    discriminator: Discriminator,
    surrogate: Surrogate,
    genobuilder: Genobuilder,
    train_thetas: np.ndarray,
    test_thetas: np.ndarray,
    epochs: int,
    parallelism: int,
    rng: np.random.Generator,
):
    train_x, train_y, train_x_generator = _generate_training_data_alfi(
        target=genobuilder.target_func,
        generator=genobuilder.generator_func,
        thetas=train_thetas,
        parallelism=parallelism,
        rng=rng,
    )
    val_x, val_y = None, None
    if len(test_thetas) > 0:
        val_x, val_y, val_x_generator = _generate_training_data_alfi(
            target=genobuilder.target_func,
            generator=genobuilder.generator_func,
            thetas=test_thetas,
            parallelism=parallelism,
            rng=rng,
        )

    discriminator.fit(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        epochs=epochs,
        # Clear the training loss/accuracy metrics from last iteration.
        reset_metrics=True,
    )

    train_y_pred = discriminator.predict(train_x_generator)
    val_x_thetas = None
    val_y_pred = None
    if len(test_thetas) > 0:
        val_x_thetas = test_thetas
        val_y_pred = discriminator.predict(val_x_generator)
    surrogate.fit(
        train_x=train_thetas,
        train_y=train_y_pred,
        val_x=val_x_thetas,
        val_y=val_y_pred,
        epochs=epochs,
    )

    alpha, beta = surrogate.predict(train_thetas)

    return train_y_pred, alpha, beta


@cleanup_process_pool_afterwards
def mcmc_gan_alfi(
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
    Kim et al. 2020, https://arxiv.org/abs/2004.05803v1

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

    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists()
            for fn in ("discriminator.pkl", "surrogate.pkl", "mcmc.ncf")
        ]
        if sum(files_exist) not in (0, len(files_exist)):
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    if resume:
        discriminator = Discriminator.from_file(store[-1] / "discriminator.pkl")
        surrogate = Surrogate.from_file(store[-1] / "surrogate.pkl")
        dataset = az.from_netcdf(store[-1] / "mcmc.ncf")
        # Discard first half as burn in.
        dataset = dataset.isel(draw=slice(steps, None))
        chain = np.array(dataset.posterior.to_array()).swapaxes(0, 2)
        start = chain[-1]
        if len(start) != walkers:
            # TODO: allow this by sampling start points for the walkers?
            raise ValueError(
                f"request for {walkers} walkers, but resuming from "
                f"{store[-1] / 'mcmc.ncf'} which used {len(start)} walkers."
            )

        thetas = chain.reshape(-1, chain.shape[-1])
        sampled_thetas = rng.choice(thetas, size=num_replicates, replace=False)
        train_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]
    else:
        discriminator = Discriminator.from_input_shape(genobuilder.feature_shape, rng)
        surrogate = Surrogate.from_input_shape(len(genobuilder.parameters), rng)
        # Starting point for the mcmc chain.
        start = genobuilder.parameters.draw_prior(num_replicates=walkers, rng=rng)

        train_thetas = genobuilder.parameters.draw_prior(
            num_replicates=training_replicates // 2, rng=rng
        )
        test_thetas = genobuilder.parameters.draw_prior(
            num_replicates=test_replicates // 2, rng=rng
        )

    # If start values are linearly dependent, emcee complains loudly.
    assert not np.any((start[0] == start[1:]).all(axis=-1))

    n_observed_calls = 0
    n_generator_calls = 0

    for i in range(len(store) + 1, len(store) + 1 + iterations):
        print(f"MCMC GAN ALFI iteration {i}")
        store.increment()

        train_y_pred, alpha, beta = _train_alfi(
            discriminator=discriminator,
            surrogate=surrogate,
            genobuilder=genobuilder,
            train_thetas=train_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            rng=rng,
        )
        discriminator.to_file(store[-1] / "discriminator.pkl")
        surrogate.to_file(store[-1] / "surrogate.pkl")

        with open(store[-1] / "s.pkl", "wb") as f:
            pickle.dump((train_thetas, train_y_pred, alpha, beta), f)

        s_pred = alpha / (alpha + beta)
        which = "D"
        for j in (np.argmax(train_y_pred), np.argmax(s_pred)):
            print(
                f"Best {which}: D(θ)={train_y_pred[j]:.3g}; "
                f"S(θ)={s_pred[j]}; "
                f"α={alpha[j]:.3g}, β={beta[j]:.3g}"
            )
            for param_name, value in zip(genobuilder.parameters, train_thetas[j]):
                print(" ", param_name, value)
            which = "S"

        dataset = _run_mcmc_emcee_alfi(
            start=start,
            surrogate=surrogate,
            parameters=genobuilder.parameters,
            walkers=walkers,
            steps=2 * steps,
            rng=rng,
        )
        """
        dataset = rw_mcmc(
            start=start[0],
            surrogate=surrogate,
            parameters=genobuilder.parameters,
            #walkers=walkers,
            steps=2 * steps,
            rng=rng,
        )
        """
        az.to_netcdf(dataset, store[-1] / "mcmc.ncf")

        # Discard first half as burn in.
        dataset = dataset.isel(draw=slice(steps, None))

        chain = np.array(dataset.posterior.to_array()).swapaxes(0, 2)
        # The chain for next iteration starts at the end of this chain.
        start = chain[-1]

        thetas = chain.reshape(-1, chain.shape[-1])

        # Discard half of the dataset with the lowest log prob.
        #lp = np.array(dataset.sample_stats.lp).swapaxes(0, 1).reshape(-1)
        #idx = np.argsort(lp)[len(lp) // 2:]
        #thetas = thetas[idx]


        sampled_thetas = rng.choice(thetas, size=num_replicates, replace=False)
        train_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]

        n_observed_calls += num_replicates
        n_generator_calls += num_replicates
        print(f"Observed data extracted {n_observed_calls} times.")
        print(f"Generator called {n_generator_calls} times.")
