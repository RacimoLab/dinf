from __future__ import annotations
import logging
import multiprocessing
import os
import pathlib
from typing import cast, Callable
import warnings

import numpy as np
import zeus
import arviz as az

import dinf

logger = logging.getLogger(__name__)

_pool = None


def _process_pool_init(parallelism):
    # Start the process pool before the GPU has been initialised, otherwise
    # we get weird GPU resource issues because the subprocesses are holding
    # onto some CUDA thing.
    # We use multiprocessing, because concurrent.futures spawns the initial
    # processes on demand (https://bugs.python.org/issue39207), which means
    # they can be spawned after the GPU has been initialised.
    global _pool
    _pool = multiprocessing.Pool(processes=parallelism)


def _sim_replicates(*, sim_func, args, num_replicates, parallelism):
    if parallelism == 1:
        map_f = map
    else:
        global _pool
        if _pool is None:
            _process_pool_init(parallelism)
        map_f = _pool.imap

    result = None
    for j, m in enumerate(map_f(sim_func, args)):
        if result is None:
            result = np.empty((num_replicates, *m.shape), dtype=m.dtype)
        result[j] = m
    return result


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


def _observe_data(*, empirical, num_replicates, parallelism, rng):
    """
    Return observations from the empirical dataset.
    """
    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates)
    data = _sim_replicates(
        sim_func=empirical,
        args=seeds,
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    return data


def _generate_training_data(
    *, empirical, generator, parameters, num_replicates, parallelism, rng
):
    nreps_generator = num_replicates // 2
    nreps_empirical = num_replicates - num_replicates // 2
    _, x_generator = _generate_data(
        generator=generator,
        parameters=parameters,
        num_replicates=nreps_generator,
        parallelism=parallelism,
        rng=rng,
    )
    x_empirical = _observe_data(
        empirical=empirical,
        num_replicates=nreps_empirical,
        parallelism=parallelism,
        rng=rng,
    )
    x = np.concatenate((x_generator, x_empirical))
    y = np.concatenate((np.zeros(nreps_generator), np.ones(nreps_empirical)))
    # shuffle
    indexes = rng.permutation(len(x))
    x = x[indexes]
    y = y[indexes]
    return x, y


def _mcmc_log_prob(
    theta: np.ndarray,
    *,
    discriminator: dinf.Discriminator,
    generator: Callable,
    parameters: dinf.Parameters,
    rng: np.random.Generator,
    num_replicates: int,
    parallelism: int,
) -> float:
    """
    Function to be maximised by mcmc. For testing the vector version (below).
    """
    assert len(theta) == len(parameters)
    if not parameters.bounds_contain(theta):
        # param out of bounds
        return -np.inf

    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates)
    params = np.tile(theta, (num_replicates, 1))
    M = _sim_replicates(
        sim_func=generator,
        args=zip(seeds, params),
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    D = np.mean(discriminator.predict(M))
    with np.errstate(divide="ignore"):
        return np.log(D)


def _mcmc_log_prob_vector(
    theta: np.ndarray,
    *,
    discriminator: dinf.Discriminator,
    generator: Callable,
    parameters: dinf.Parameters,
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


def _chain_to_netcdf(chain, parameters, filename):
    # Zeus chain has shape (steps, walkers, params)
    # arviz InferenceData needs walkers first
    chain = chain.swapaxes(0, 1)

    datadict = {p: chain[..., j] for j, p in enumerate(parameters)}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="More chains.*than draws",
            module="arviz",
        )
        dataset = az.convert_to_inference_data(datadict)

    az.to_netcdf(dataset, filename)

def _chain_from_netcdf(filename):
    dataset = az.from_netcdf(filename)
    chain = np.array(dataset.posterior.to_array())
    # Chain has shape (params, walkers, steps)
    chain = chain.swapaxes(0, 2)
    # now has shape (steps, walkers, params)
    return chain

def _run_mcmc(
    start: np.ndarray,
    discriminator: dinf.Discriminator,
    genobuilder: dinf.Genobuilder,
    walkers: int,
    steps: int,
    Dx_replicates: int,
    parallelism: int,
    rng: np.random.Generator,
):
    sampler = zeus.EnsembleSampler(
        walkers,
        len(genobuilder.parameters),
        _mcmc_log_prob_vector,
        verbose=False,
        vectorize=True,
        # kwargs passed to _mcmc_log_prob_vector
        kwargs=dict(
            discriminator=discriminator,
            generator=genobuilder.generator_func,
            parameters=genobuilder.parameters,
            parallelism=parallelism,
            num_replicates=Dx_replicates,
            rng=rng,
        ),
    )

    sampler.run_mcmc(start, nsteps=steps)
    chain = sampler.get_chain()
    # This is an overestimate, because the generator won't get called when
    # the proposed parameters are out of bounds.
    num_generator_calls = sampler.ncall * Dx_replicates
    return chain, num_generator_calls


def _train_discriminator(
    *,
    discriminator: dinf.Discriminator,
    genobuilder: dinf.Genobuilder,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    parallelism: int,
    rng: np.random.Generator,
):
    train_x, train_y = _generate_training_data(
        empirical=genobuilder.empirical_func,
        generator=genobuilder.generator_func,
        parameters=genobuilder.parameters,
        num_replicates=training_replicates,
        parallelism=parallelism,
        rng=rng,
    )
    if test_replicates > 0:
        val_x, val_y = _generate_training_data(
            empirical=genobuilder.empirical_func,
            generator=genobuilder.generator_func,
            parameters=genobuilder.parameters,
            num_replicates=test_replicates,
            parallelism=parallelism,
            rng=rng,
        )
    else:
        val_x = np.empty((0, *train_x.shape[1:]), dtype=train_x.dtype)
        val_y = np.empty((0, *train_y.shape[1:]), dtype=train_y.dtype)

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


def mcmc_gan(
    *,
    genobuilder: dinf.Genobuilder,
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
        Size of the test dataset used to evalutate the discriminator after
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
        :meth:`Genobuilder.empirical_func`.
    :param rng:
        Numpy random number generator.
    """

    if working_directory is None:
        working_directory = "."
    store = dinf.Store(working_directory)

    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism)

    resume = False
    if len(store) > 0:
        files_exist = [(store[-1] / fn).exists() for fn in ("discriminator.pkl", "mcmc.ncf")]
        if files_exist == 1:
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    if resume:
        discriminator = dinf.Discriminator.from_file(store[-1] / "discriminator.pkl")
        chain = _chain_from_netcdf(store[-1] / "mcmc.ncf")
        start = chain[-1]
        if len(start) != walkers:
            # TODO: allow this by sampling start points for the walkers?
            raise ValueError(
                f"request for {walkers} walkers, but resuming from "
                f"{store[-1] / 'mcmc.ncf'} which used {len(start)} walkers."
            )
        posterior_sample = chain.reshape(-1, chain.shape[-1])
        genobuilder.parameters.update_posterior(posterior_sample)
    else:
        discriminator = dinf.Discriminator.from_input_shape(genobuilder.feature_shape, rng)
        # Starting point for the mcmc chain.
        start = genobuilder.parameters.draw(num_replicates=walkers, rng=rng)

    n_observed_calls = 0
    n_generator_calls = 0

    for i in range(iterations):
        print(f"GAN iteration {i}")
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
        discriminator.to_file(store[-1] / f"discriminator.pkl")

        chain, mcmc_generator_calls = _run_mcmc(
            start=start,
            discriminator=discriminator,
            genobuilder=genobuilder,
            walkers=walkers,
            steps=steps,
            Dx_replicates=Dx_replicates,
            parallelism=parallelism,
            rng=rng,
        )
        _chain_to_netcdf(chain, genobuilder.parameters, store[-1] / f"mcmc.ncf")

        # The chain for next iteration starts at the end of this chain.
        start = chain[-1]

        # Update the parameters to draw from the posterior sample
        # (the merged chains from the mcmc).
        posterior_sample = chain.reshape(-1, chain.shape[-1])
        genobuilder.parameters.update_posterior(posterior_sample)

        n_observed_calls += training_replicates + test_replicates
        n_generator_calls += (
            training_replicates + test_replicates + mcmc_generator_calls
        )
        print(f"Observed data extracted {n_observed_calls} times.")
        print(f"Generator called {n_generator_calls} times.")
