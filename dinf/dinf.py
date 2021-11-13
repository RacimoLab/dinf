import functools
import logging
import multiprocessing
import pickle
import warnings

import numpy as np
import zeus
import arviz as az

from . import discriminator

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
            _process_pool_init()
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
    _, random_data = _generate_data(
        generator=generator,
        parameters=parameters,
        num_replicates=num_replicates,
        parallelism=parallelism,
        rng=rng,
    )
    fixed_data = _observe_data(
        empirical=empirical,
        num_replicates=num_replicates,
        parallelism=parallelism,
        rng=rng,
    )
    x = np.concatenate((random_data, fixed_data))
    y = np.concatenate((np.zeros(num_replicates), np.ones(num_replicates)))
    # shuffle
    indexes = rng.permutation(len(x))
    x = x[indexes]
    y = y[indexes]
    return x, y


def _mcmc_log_prob(
    theta, *, discr, generator, parameters, rng, num_replicates, parallelism
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
        generator=generator,
        args=zip(seeds, params),
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    D = np.mean(discr.predict(M))
    with np.errstate(divide="ignore"):
        return np.log(D)


def _mcmc_log_prob_vector(
    theta, *, discr, generator, parameters, rng, num_replicates, parallelism
) -> np.ndarray:
    """
    Function to be maximised by mcmc. Vectorised version.
    """
    num_walkers, num_params = theta.shape
    assert num_params == len(parameters)
    log_D = np.full(num_walkers, -np.inf)

    # Identify workers with one or more out-of-bounds parameters.
    in_bounds = parameters.bounds_contain_vec(theta)
    num_in_bounds = np.sum(in_bounds)
    if num_in_bounds == 0:
        return log_D

    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates * num_in_bounds)
    params = np.repeat(theta[in_bounds], num_replicates, axis=0)
    assert len(seeds) == len(params)
    M = _sim_replicates(
        generator=generator,
        args=zip(seeds, params),
        num_replicates=len(seeds),
        parallelism=parallelism,
    )
    Dreps = discr.predict(M).reshape(num_in_bounds, num_replicates)
    D = np.mean(Dreps, axis=1)
    assert len(D) == num_in_bounds
    with np.errstate(divide="ignore"):
        log_D[in_bounds] = np.log(D)

    return log_D


def _arviz_dataset_from_zeus_chain(chain, parameters):
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
        return az.convert_to_inference_data(datadict)


def _sim_shim(args, *, func, keys):
    """
    Wrapper that takes an argument list, and calls func with keyword args.
    """
    seed, *func_args = args
    kwargs = dict(zip(keys, func_args))
    return func(seed=seed, **kwargs)


def mcmc_gan(
    *,
    empirical_func,
    generator_func,
    parameters,
    feature_shape,
    walkers,
    steps_per_iteration,
    parallelism,
    working_directory,
    num_Dx_replicates,
    gan_iterations,
    training_epochs,
    num_training_replicates,
    num_validation_replicates,
    rng,
):
    _process_pool_init(parallelism)

    # discr = discriminator.Discriminator.from_file(discriminator_filename)
    discr = discriminator.Discriminator.from_input_shape(feature_shape, rng)

    generator = functools.partial(
        _sim_shim, func=generator_func, keys=tuple(parameters)
    )
    zeus_kwargs = dict(
        discr=discr,
        generator=generator,
        parameters=parameters,
        parallelism=parallelism,
        num_replicates=num_Dx_replicates,
        rng=rng,
    )
    num_params = len(parameters)

    n_observed_calls = 0
    n_generator_calls = 0

    # Starting point for the mcmc chain.
    start = parameters.draw(num_replicates=walkers, rng=rng)

    for i in range(gan_iterations):
        print(f"GAN iteration {i}")

        train_x, train_y = _generate_training_data(
            empirical=empirical_func,
            generator=generator,
            parameters=parameters,
            num_replicates=num_training_replicates,
            parallelism=parallelism,
            rng=rng,
        )
        val_x, val_y = _generate_training_data(
            empirical=empirical_func,
            generator=generator,
            parameters=parameters,
            num_replicates=num_validation_replicates,
            parallelism=parallelism,
            rng=rng,
        )

        discr.fit(
            rng,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            epochs=training_epochs,
            tensorboard_log_dir=working_directory / "tensorboard" / "fit",
            # Clear the training loss/accuracy metrics from last iteration.
            reset_metrics=True,
        )
        discr.to_file(working_directory / f"discriminator_{i}.pkl")

        sampler = zeus.EnsembleSampler(
            walkers,
            num_params,
            _mcmc_log_prob_vector,
            kwargs=zeus_kwargs,
            verbose=False,
            vectorize=True,
        )

        sampler.run_mcmc(start, nsteps=steps_per_iteration)
        # The chain for next iteration starts at the end of this chain.
        start = sampler.get_last_sample()

        chain = sampler.get_chain()
        dataset = _arviz_dataset_from_zeus_chain(chain, parameters)
        az.to_netcdf(dataset, working_directory / f"mcmc_samples_{i}.ncf")

        # Update the parameters to draw from the posterior sample
        # (the merged chains from the mcmc).
        posterior_sample = chain.reshape(-1, chain.shape[-1])
        parameters.update_posterior(posterior_sample)

        # training
        n_observed_calls += num_training_replicates + num_validation_replicates
        n_generator_calls += num_training_replicates + num_validation_replicates
        # mcmc sampler
        n_generator_calls += sampler.ncall * num_Dx_replicates

        print(f"Observed data extracted {n_observed_calls} times.")
        print(f"Generator called {n_generator_calls} times.")


def save_genobuilder(
    filename, *, empirical_func, generator_func, parameters, feature_shape
) -> None:
    data = dict(
        empirical_func=empirical_func,
        generator_func=generator_func,
        parameters=parameters,
        feature_shape=feature_shape,
    )
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_genobuilder(filename) -> tuple:
    with open(filename, "rb") as f:
        data = pickle.load(f)
    keys = ("empirical_func", "generator_func", "parameters", "feature_shape")
    values = tuple(data.get(k) for k in keys)
    for k, v in zip(keys, values):
        if v is None:
            raise ValueError(f"{k} not found in genobuilder {filename}")
    return values
