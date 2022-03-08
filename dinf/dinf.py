from __future__ import annotations
import functools
import itertools
import logging
import math
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
        generator=generator,
        thetas=thetas,
        parallelism=parallelism,
        rng=rng,
    )
    x_target = _observe_data(
        target=target,
        num_replicates=num_replicates,
        parallelism=parallelism,
        rng=rng,
    )
    # XXX: Large copy doubles peak memory.
    # Preallocate somehow?
    x = jax.tree_map(lambda *l: np.concatenate(l), x_generator, x_target)
    del x_generator
    del x_target
    y = np.concatenate((np.zeros(num_replicates), np.ones(num_replicates)))
    # shuffle
    indexes = rng.permutation(len(y))
    # XXX: Large copy doubles peak memory.
    # Do this in-place?
    # https://stackoverflow.com/a/60902210/9500949
    x = jax.tree_map(lambda l: l[indexes], x)
    y = y[indexes]
    return x, y


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


def _run_mcmc_emcee(
    start: np.ndarray,
    discriminator: Discriminator,
    generator: Callable,
    parameters: Parameters,
    walkers: int,
    steps: int,
    Dx_replicates: int,
    parallelism: int,
    rng: np.random.Generator,
):
    sampler = emcee.EnsembleSampler(
        walkers,
        len(parameters),
        _log_prob,
        vectorize=True,
        # kwargs passed to _log_prob
        kwargs=dict(
            discriminator=discriminator,
            generator=generator,
            parameters=parameters,
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
            p: chain[..., j].swapaxes(0, 1) for j, p in enumerate(parameters)
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
    training_thetas: np.ndarray,
    test_thetas: np.ndarray,
    epochs: int,
    parallelism: int,
    rng: np.random.Generator,
):
    train_x, train_y = _generate_training_data(
        target=genobuilder.target_func,
        generator=genobuilder.generator_func,
        thetas=training_thetas,
        parallelism=parallelism,
        rng=rng,
    )
    val_x, val_y = None, None
    if test_thetas is not None and len(test_thetas) > 0:
        val_x, val_y = _generate_training_data(
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
        # Clear the training loss/accuracy metrics from last iteration.
        reset_metrics=True,
        # TODO
        # tensorboard_log_dir=working_directory / "tensorboard" / "fit",
    )
    return metrics


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

        thetas = chain.reshape(-1, chain.shape[-1])
        sampled_thetas = rng.choice(thetas, size=num_replicates, replace=False)
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]
    else:
        discriminator = Discriminator.from_input_shape(genobuilder.feature_shape, rng)
        # Starting point for the mcmc chain.
        start = genobuilder.parameters.draw_prior(num_replicates=walkers, rng=rng)

        training_thetas = genobuilder.parameters.draw_prior(
            num_replicates=training_replicates // 2, rng=rng
        )
        test_thetas = genobuilder.parameters.draw_prior(
            num_replicates=test_replicates // 2, rng=rng
        )

    # If start values are linearly dependent, emcee complains loudly.
    assert not np.any((start[0] == start[1:]).all(axis=-1))

    n_target_calls = 0
    n_generator_calls = 0

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

        dataset = _run_mcmc_emcee(
            start=start,
            discriminator=discriminator,
            generator=genobuilder.generator_func,
            parameters=genobuilder.parameters,
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

        thetas = chain.reshape(-1, chain.shape[-1])
        sampled_thetas = rng.choice(thetas, size=num_replicates, replace=False)
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
        generator=generator,
        thetas=thetas,
        parallelism=parallelism,
        rng=rng,
    )
    y = discriminator.predict(x)
    top = np.argsort(y)[::-1][:posteriors]
    top_params = thetas[top]
    with np.errstate(divide="ignore"):
        log_top_y = np.log(y[top])

    dataset = az.from_dict(
        posterior={p: top_params[..., j] for j, p in enumerate(parameters)},
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
      - training the discriminator for a certain number of epochs,
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

        chain = np.array(dataset.posterior.to_array()).swapaxes(0, 2)
        thetas = chain.reshape(-1, chain.shape[-1])
        sampled_thetas = rng.choice(thetas, size=num_replicates, replace=True)
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]
    else:
        discriminator = Discriminator.from_input_shape(genobuilder.feature_shape, rng)

        training_thetas = genobuilder.parameters.draw_prior(
            num_replicates=training_replicates // 2, rng=rng
        )
        test_thetas = genobuilder.parameters.draw_prior(
            num_replicates=test_replicates // 2, rng=rng
        )

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

        proposal_thetas = genobuilder.parameters.draw_prior(
            num_replicates=proposals, rng=rng
        )
        dataset = _run_abc(
            discriminator=discriminator,
            generator=genobuilder.generator_func,
            parameters=genobuilder.parameters,
            thetas=proposal_thetas,
            posteriors=posteriors,
            parallelism=parallelism,
            rng=rng,
        )
        az.to_netcdf(dataset, store[-1] / "abc.ncf")

        chain = np.array(dataset.posterior.to_array()).swapaxes(0, 2)
        thetas = chain.reshape(-1, chain.shape[-1])
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

    discriminator = Discriminator.from_input_shape(genobuilder.feature_shape, rng)
    acc_best = 0
    theta_best = None

    for k in range(max_pretraining_iterations):
        theta = genobuilder.parameters.draw_prior(num_replicates=1, rng=rng)[0]
        training_thetas = np.tile(theta, (training_replicates // 2, 1))
        test_thetas = np.tile(theta, (test_replicates // 2, 1))

        metrics = _train_discriminator(
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

    discriminator = Discriminator.from_input_shape(genobuilder.feature_shape, rng)

    for k in range(max_pretraining_iterations):
        training_thetas = genobuilder.parameters.draw_prior(
            num_replicates=training_replicates // 2, rng=rng
        )
        test_thetas = genobuilder.parameters.draw_prior(
            num_replicates=test_replicates // 2, rng=rng
        )

        metrics = _train_discriminator(
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

    thetas = genobuilder.parameters.draw_prior(
        num_replicates=training_replicates, rng=rng
    )
    lp = _log_prob(
        thetas,
        discriminator=discriminator,
        generator=genobuilder.generator_func,
        parameters=genobuilder.parameters,
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
    """
    num_replicates = training_replicates // 2 + test_replicates // 2

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
            for fn in ("discriminator.pkl", "pg-gan-proposals.ncf")
        ]
        if sum(files_exist) == 1:
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    if resume:
        discriminator = Discriminator.from_file(store[-1] / "discriminator.pkl")
        dataset = az.from_netcdf(store[-1] / "pg-gan-proposals.ncf")
        proposal_thetas = np.array(dataset.posterior.isel(draw=-1).to_array()).swapaxes(
            0, 1
        )
        lp = np.array(dataset.sample_stats.lp.isel(draw=-1))
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
            parameters=genobuilder.parameters,
            proposal_stddev=proposal_stddev,
        )

        lp = _log_prob(
            proposal_thetas,
            discriminator=discriminator,
            generator=genobuilder.generator_func,
            parameters=genobuilder.parameters,
            num_replicates=Dx_replicates,
            parallelism=parallelism,
            rng=rng,
        )
        n_generator_calls += len(proposal_thetas) * Dx_replicates

        datadict = {
            "posterior": {
                p: np.expand_dims(proposal_thetas[..., j], -1)
                for j, p in enumerate(genobuilder.parameters)
            },
            "sample_stats": {
                "lp": np.expand_dims(lp, -1),
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
        az.to_netcdf(dataset, store[-1] / "pg-gan-proposals.ncf")

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

        for theta_i, (name, param) in zip(theta, genobuilder.parameters.items()):
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
            )
            n_target_calls += num_replicates
            n_generator_calls += num_replicates

        discriminator.to_file(store[-1] / "discriminator.pkl")

        print(f"Target called {n_target_calls} times.")
        print(f"Generator called {n_generator_calls} times.")
        print()
