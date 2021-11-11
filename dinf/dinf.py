import functools
import logging
import multiprocessing
import warnings

import numpy as np
import gradient_free_optimizers
import zeus
import arviz as az

from . import cache, discriminator

logger = logging.getLogger(__name__)

_pool = None


def _process_pool_init(parallelism):
    # Start the process pool before the GPU has been initialised, otherwise
    # we get weird GPU resource issues because the subprocesses are holding
    # onto some CUDA thing.
    # We use multiprocessing, because concurrent.futures spawns the initial
    # processes on demand (https://bugs.python.org/issue39207), which means they
    # can be spawned after the GPU has been initialised.
    global _pool
    _pool = multiprocessing.Pool(processes=parallelism)


def _sim_replicates(*, generator, args, num_replicates, parallelism):
    if parallelism == 1:
        map_f = map
    else:
        global _pool
        if _pool is None:
            _process_pool_init()
        map_f = _pool.imap

    result = None
    for j, m in enumerate(map_f(generator.sim, args)):
        if result is None:
            result = np.zeros((num_replicates, *m.shape), dtype=m.dtype)
        result[j] = m
    # Expand dimension by 1 (add channel dim).
    result = np.expand_dims(result, axis=-1)
    return result


def _generate_data(*, generator, num_replicates, parallelism, rng, random):
    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates)
    params = generator.draw_params(
        num_replicates=num_replicates, random=random, rng=rng
    )
    data = _sim_replicates(
        generator=generator,
        args=zip(seeds, params),
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    return params, data


def _generate_training_data(*, generator, num_replicates, parallelism, rng):
    (_, random_data), (_, fixed_data) = (
        _generate_data(
            generator=generator,
            num_replicates=num_replicates,
            rng=rng,
            parallelism=parallelism,
            random=random,
        )
        for random in [True, False]
    )
    x = np.concatenate((random_data, fixed_data))
    y = np.concatenate((np.zeros(num_replicates), np.ones(num_replicates)))
    # shuffle
    indexes = rng.permutation(len(x))
    x = x[indexes]
    y = y[indexes]
    return x, y


def train(
    *,
    generator,
    discriminator_filename,
    num_training_replicates,
    num_validation_replicates,
    parallelism,
    training_epochs,
    working_directory,
    rng,
):

    train_cache = cache.Cache(
        path="train-cache.zarr",
        keys=("train/data", "train/labels", "val/data", "val/labels"),
    )
    if train_cache.exists():
        logger.info("loading training data from {train_zarr_cache}")
        train_x, train_y, val_x, val_y = train_cache.load()
    else:
        logger.info("generating training data")
        train_x, train_y = _generate_training_data(
            generator=generator,
            num_replicates=num_training_replicates,
            parallelism=parallelism,
            rng=rng,
        )
        val_x, val_y = _generate_training_data(
            generator=generator,
            num_replicates=num_validation_replicates,
            parallelism=parallelism,
            rng=rng,
        )
        logger.info("saving training data to {train_zarr_cache}")
        train_cache.save((train_x, train_y, val_x, val_y))

    discr = discriminator.Discriminator.from_input_shape(train_x.shape[1:], rng)
    discr.summary()
    discr.fit(
        rng,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        epochs=training_epochs,
        tensorboard_log_dir=working_directory / "tensorboard" / "fit",
    )
    discr.to_file(discriminator_filename)


def abc(
    *,
    generator,
    discriminator_filename,
    num_replicates,
    parallelism,
    working_directory,
    rng,
):
    abc_cache = cache.Cache(
        path=working_directory / "abc-cache.zarr",
        keys=("abc/params", "abc/data"),
    )
    if abc_cache.exists():
        params, data = abc_cache.load()
    else:
        params, data = _generate_data(
            generator=generator,
            num_replicates=num_replicates,
            rng=rng,
            parallelism=parallelism,
            random=True,
        )
        abc_cache.save((params, data))

    d = discriminator.Discriminator.from_file(discriminator_filename)
    predictions = d.predict(data)
    datadict = {p.name: params[:, j] for j, p in enumerate(generator.params)}
    datadict["D"] = predictions
    dataset = az.convert_to_inference_data(datadict)
    az.to_netcdf(dataset, working_directory / "abc.ncf")


def _opt_func(opt_params, *, discr, generator, rng, num_replicates, parallelism):
    """
    Function to be maximised by gfo.
    """
    param_values = tuple(opt_params.values())
    if not all(
        p.bounds[0] <= x <= p.bounds[1] for x, p in zip(param_values, generator.params)
    ):
        # param out of bounds
        return -np.inf

    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates)
    params = np.tile(param_values, (num_replicates, 1))
    M = _sim_replicates(
        generator=generator,
        args=zip(seeds, params),
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    D = np.mean(discr.predict(M))
    return D


def opt(
    *,
    generator,
    discriminator_filename,
    iterations,
    parallelism,
    working_directory,
    num_Dx_replicates,
    rng,
):
    _process_pool_init(parallelism)

    discr = discriminator.Discriminator.from_file(discriminator_filename)
    search_space = {p.name: np.arange(*p.bounds) for p in generator.params}
    opt = gradient_free_optimizers.SimulatedAnnealingOptimizer(search_space)
    # opt = gradient_free_optimizers.RandomAnnealingOptimizer(search_space)
    f = functools.partial(
        _opt_func,
        discr=discr,
        generator=generator,
        parallelism=parallelism,
        num_replicates=num_Dx_replicates,
        rng=rng,
    )
    f = functools.update_wrapper(f, _opt_func)
    opt.search(f, n_iter=iterations)  # iterations)
    # opt.results is a pandas dataframe

    datadict = {p.name: opt.results[p.name] for j, p in enumerate(generator.params)}
    dataset = az.convert_to_inference_data(datadict)
    az.to_netcdf(dataset, working_directory / "opt.ncf")


def _mcmc_log_prob(mcmc_params, *, discr, generator, rng, num_replicates, parallelism):
    """
    Function to be maximised by mcmc. For testing the vector version (below).
    """
    assert len(mcmc_params) == len(generator.params)
    if not all(
        p.bounds[0] <= x <= p.bounds[1] for x, p in zip(mcmc_params, generator.params)
    ):
        # param out of bounds
        return -np.inf

    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates)
    params = np.tile(mcmc_params, (num_replicates, 1))
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
    mcmc_params, *, discr, generator, rng, num_replicates, parallelism
):
    """
    Function to be maximised by mcmc. Vectorised version.
    """
    num_walkers, num_params = mcmc_params.shape
    assert num_params == len(generator.params)
    log_D = np.full(num_walkers, -np.inf)

    # Identify workers with one or more out-of-bounds parameters.
    lo, hi = zip(*[p.bounds for p in generator.params])
    in_bounds = np.all(np.logical_and(lo <= mcmc_params, mcmc_params <= hi), axis=1)
    num_in_bounds = np.sum(in_bounds)
    if num_in_bounds == 0:
        return log_D

    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates * num_in_bounds)
    params = np.repeat(mcmc_params[in_bounds], num_replicates, axis=0)
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


def mcmc(
    *,
    generator,
    discriminator_filename,
    walkers,
    steps,
    parallelism,
    working_directory,
    num_Dx_replicates,
    rng,
):
    _process_pool_init(parallelism)

    discr = discriminator.Discriminator.from_file(discriminator_filename)
    kwargs = dict(
        discr=discr,
        generator=generator,
        parallelism=parallelism,
        num_replicates=num_Dx_replicates,
        rng=rng,
    )
    ndim = len(generator.params)
    start = generator.draw_params(num_replicates=walkers, random=True, rng=rng)
    sampler = zeus.EnsembleSampler(
        walkers,
        ndim,
        _mcmc_log_prob_vector,
        kwargs=kwargs,
        verbose=False,
        vectorize=True,
    )
    sampler.run_mcmc(start, nsteps=steps)
    chain = sampler.get_chain()
    D = np.exp(sampler.get_log_prob())
    # shape is (steps, walkers, params), but arviz needs walkers first
    chain = chain.swapaxes(0, 1)
    D = D.swapaxes(0, 1)

    datadict = {p.name: chain[..., j] for j, p in enumerate(generator.params)}
    dataset = az.convert_to_inference_data(datadict)
    az.to_netcdf(dataset, working_directory / "mcmc.ncf")


def mcmc_gan(
    *,
    generator,
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

    input_shape = generator.feature_extractor.shape + tuple((1,))
    # discr = discriminator.Discriminator.from_file(discriminator_filename)
    discr = discriminator.Discriminator.from_input_shape(input_shape, rng)
    kwargs = dict(
        discr=discr,
        generator=generator,
        parallelism=parallelism,
        num_replicates=num_Dx_replicates,
        rng=rng,
    )
    num_params = len(generator.params)

    n_observed_calls = 0
    n_generator_calls = 0

    # Starting point for the mcmc chain is drawn from the prior.
    start = generator.draw_params(num_replicates=walkers, random=True, rng=rng)

    for i in range(gan_iterations):
        print(f"GAN iteration {i}")

        train_x, train_y = _generate_training_data(
            generator=generator,
            num_replicates=num_training_replicates,
            parallelism=parallelism,
            rng=rng,
        )
        val_x, val_y = _generate_training_data(
            generator=generator,
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
            # clear the training loss/accuracy metrics from last iteraction
            reset_metrics=True,
        )
        discr.to_file(working_directory / f"discriminator_{i}.pkl")

        sampler = zeus.EnsembleSampler(
            walkers,
            num_params,
            _mcmc_log_prob_vector,
            kwargs=kwargs,
            verbose=False,
            vectorize=True,
        )

        sampler.run_mcmc(start, nsteps=steps_per_iteration)
        # The chain for next iteration starts at the end of this chain.
        start = sampler.get_last_sample()

        chain = sampler.get_chain()
        assert chain.shape == (steps_per_iteration, walkers, num_params)
        # arviz InferenceData needs walkers first
        chain = chain.swapaxes(0, 1)

        datadict = {p.name: chain[..., j] for j, p in enumerate(generator.params)}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="More chains.*than draws",
                module="arviz",
            )
            dataset = az.convert_to_inference_data(datadict)
        az.to_netcdf(dataset, working_directory / f"mcmc_samples_{i}.ncf")

        # Update the generator to draw from the posterior sample
        # (the merged chains from the mcmc).
        posterior_sample = chain.reshape(-1, chain.shape[-1])
        generator.update_posterior(posterior_sample)

        # training
        n_observed_calls += num_training_replicates + num_validation_replicates
        n_generator_calls += num_training_replicates + num_validation_replicates
        # mcmc sampler
        n_generator_calls += sampler.ncall * num_Dx_replicates

        print(f"Observed data extracted {n_observed_calls} times.")
        print(f"Generator called {n_generator_calls} times.")
