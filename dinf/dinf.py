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

    global _ex
    if _ex is None:
        _ex = concurrent.futures.ProcessPoolExecutor(max_workers=parallelism)
        # _ex = concurrent.futures.ThreadPoolExecutor(max_workers=parallelism)
    ex = _ex

    result = None
    j = 0
    for chunk_args in chunkify(args, chunk_size=1000):
        for m in ex.map(generator.sim, chunk_args):
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

    nn = discriminator.build(train_x.shape[1:])
    nn.summary()
    discriminator.fit(
        nn,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        epochs=training_epochs,
    )
    discriminator.save(nn, discriminator_filename)


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

    nn = discriminator.load(discriminator_filename)
    predictions = discriminator.predict(nn, data)
    datadict = {p.name: params[:, j] for j, p in enumerate(generator.params)}
    datadict["D"] = predictions
    dataset = az.convert_to_inference_data(datadict)
    az.to_netcdf(dataset, working_directory / "abc.ncf")


def _opt_func(gfo_params, *, nn, generator, rng, num_replicates, parallelism):
    """
    Function to be maximised by gfo.
    """
    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates)
    params = np.tile(tuple(gfo_params.values()), (num_replicates, 1))
    M = _sim_replicates(
        generator=generator,
        args=zip(seeds, params),
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    D = np.mean(discriminator.predict(nn, M))
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
    nn = discriminator.load(discriminator_filename)
    search_space = {p.name: np.arange(*p.bounds) for p in generator.params}
    opt = gradient_free_optimizers.SimulatedAnnealingOptimizer(search_space)
    # opt = gradient_free_optimizers.RandomAnnealingOptimizer(search_space)
    f = functools.partial(
        _opt_func,
        nn=nn,
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


def _mcmc_log_prob(mcmc_params, *, nn, generator, rng, num_replicates, parallelism):
    """
    Function to be maximised by zeus mcmc.
    """
    if not all(
        p.bounds[0] <= x <= p.bounds[1] for x, p in zip(mcmc_params, generator.params)
    ):
        # param out of bounds
        return -np.inf

    #    logging.basicConfig(level="WARNING")
    seeds = rng.integers(low=1, high=2 ** 31, size=num_replicates)
    params = np.tile(mcmc_params, (num_replicates, 1))
    M = _sim_replicates(
        generator=generator,
        args=zip(seeds, params),
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    D = np.mean(discriminator.predict(nn, M))
    with np.errstate(divide="ignore"):
        return np.log(D)


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
    nn = discriminator.load(discriminator_filename)
    f = functools.partial(
        _mcmc_log_prob,
        nn=nn,
        generator=generator,
        parallelism=parallelism,
        num_replicates=num_Dx_replicates,
        rng=rng,
    )
    ndim = len(generator.params)
    start = generator.draw_params(num_replicates=walkers, random=True, rng=rng)
    sampler = zeus.EnsembleSampler(walkers, ndim, f, verbose=False)
    sampler.run_mcmc(start, nsteps=steps)
    chain = sampler.get_chain()
    # shape is (steps, walkers, params), but arviz needs walkers first
    chain = chain.swapaxes(0, 1)

    datadict = {p.name: chain[..., j] for j, p in enumerate(generator.params)}
    dataset = az.convert_to_inference_data(datadict)
    az.to_netcdf(dataset, working_directory / "mcmc.ncf")
