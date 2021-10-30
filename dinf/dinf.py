import concurrent.futures
import itertools
import logging
import functools

import numpy as np
import gradient_free_optimizers
import zeus
import arviz as az

from . import cache, discriminator

logger = logging.getLogger(__name__)

_ex = None


def _sim_replicates(*, generator, args, num_replicates, parallelism):
    def chunkify(iterable, chunk_size):
        # Stolen from https://bugs.python.org/issue34168
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, chunk_size))
            if chunk:
                yield chunk
            else:
                return

    if parallelism == 1:
        map_f = map
    else:
        global _ex
        if _ex is None:
            _ex = concurrent.futures.ProcessPoolExecutor(max_workers=parallelism)
            # _ex = concurrent.futures.ThreadPoolExecutor(max_workers=parallelism)
        map_f = _ex.map

    result = None
    j = 0
    for chunk_args in chunkify(args, chunk_size=1000):
        for m in map_f(generator.sim, chunk_args):
            if result is None:
                result = np.zeros((num_replicates, *m.shape), dtype=m.dtype)
            result[j] = m
            j += 1
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


def _train_test_split(x, y, validation_ratio, rng):
    # shuffle
    indexes = rng.permutation(len(x))
    x = x[indexes]
    y = y[indexes]
    # split
    n_train = int(len(x) * (1 - validation_ratio))
    train_x = x[:n_train]
    train_y = y[:n_train]
    val_x = x[n_train:]
    val_y = y[n_train:]
    return train_x, train_y, val_x, val_y


def _generate_training_data(
    *, generator, num_replicates, validation_ratio, parallelism, rng
):
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
    return _train_test_split(x, y, validation_ratio, rng)


def train(
    *,
    generator,
    discriminator_filename,
    num_replicates,
    validation_ratio,
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
        train_x, train_y, val_x, val_y = _generate_training_data(
            generator=generator,
            num_replicates=num_replicates,
            parallelism=parallelism,
            validation_ratio=validation_ratio,
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


def _opt_func(gfo_params, *, discr, generator, rng, num_replicates, parallelism):
    """
    Function to be maximised by gfo.
    """
    param_values = tuple(gfo_params.values())
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
    az.to_netcdf(dataset, working_directory / "gfo.ncf")


def _mcmc_log_prob(mcmc_params, *, discr, generator, rng, num_replicates, parallelism):
    """
    Function to be maximised by zeus mcmc.
    """
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
    Function to be maximised by zeus mcmc. Vectorised version.
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
        _mcmc_log_prob,
        # _mcmc_log_prob_vector,
        kwargs=kwargs,
        verbose=False,
        # vectorize=True,
    )
    sampler.run_mcmc(start, nsteps=steps)
    chain = sampler.get_chain()
    D = np.exp(sampler.get_log_prob())
    # shape is (steps, walkers, params), but arviz needs walkers first
    chain = chain.swapaxes(0, 1)
    D = D.swapaxes(0, 1)

    datadict = {p.name: chain[..., j] for j, p in enumerate(generator.params)}
    datadict["D"] = D
    # dataset = az.convert_to_inference_data(datadict)
    # az.to_netcdf(dataset, working_directory / "mcmc.ncf")
