from __future__ import annotations
import copy
import logging
import multiprocessing
import os
import pathlib
import signal
from typing import cast, Callable
import warnings

import arviz as az
import emcee
import jax
import numpy as np

from .discriminator import Discriminator
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
    seeds = rng.integers(low=1, high=2**31, size=num_replicates)
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
    seeds = rng.integers(low=1, high=2**31, size=num_replicates)
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

    seeds = rng.integers(low=1, high=2**31, size=num_replicates * num_in_bounds)
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

    mt_initial_state = np.random.mtrand.RandomState(rng.integers(2**31)).get_state()
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
        rng,
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
