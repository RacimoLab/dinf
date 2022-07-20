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
from .dinf_model import DinfModel
from .parameters import Parameters
from .store import Store

logger = logging.getLogger(__name__)

_pool = None


def _initializer(filename):
    # Ignore ctrl-c in workers. This is handled in the main process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Ensure symbols from the user's dinf_model are avilable to workers.
    if filename is not None:
        DinfModel.from_file(filename)


def _process_pool_init(parallelism, dinf_model):
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
        initargs=(dinf_model.filename,),
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


def _generate_training_data(*, target, generator, thetas, parallelism, ss):
    ss_generator, ss_target = ss.spawn(("generator", "target"))
    num_replicates = len(thetas)
    x_generator = _generate_data(
        generator=generator,
        thetas=thetas,
        parallelism=parallelism,
        rng=np.random.default_rng(ss_generator),
    )
    x_target = _observe_data(
        target=target,
        num_replicates=num_replicates,
        parallelism=parallelism,
        rng=np.random.default_rng(ss_target),
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
    # We open the file ourselves to stop numpy from adding a .npz extension,
    # e.g. if the filename is /dev/null or a fifo.
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


class NamedSeedSequence(np.random.SeedSequence):
    """
    Extends numpy SeedSequence to support string-valued spawn_key.
    """

    def __init__(self, entropy=None, *, spawn_key=(), **kwargs):
        if isinstance(spawn_key, str):
            spawn_key = tuple(map(ord, spawn_key))
        if isinstance(entropy, np.random.SeedSequence):
            if entropy.spawn_key != ():
                spawn_key = entropy.spawn_key + (ord(":"),) + spawn_key
            entropy = entropy.entropy
        super().__init__(entropy, spawn_key=spawn_key, **kwargs)

    def spawn(self, n: int | Tuple[str, ...]):
        if isinstance(n, int):
            return super().spawn(n)
        else:
            return [type(self)(self, spawn_key=nj) for nj in n]


def _train_discriminator(
    *,
    discriminator: Discriminator,
    dinf_model: DinfModel,
    training_thetas: np.ndarray,
    test_thetas: np.ndarray,
    epochs: int,
    parallelism: int,
    ss: NamedSeedSequence,
    entropy_regularisation: bool = False,
):
    ss_train, ss_val, ss_fit = ss.spawn(
        ("features:train", "features:val", "discriminator:fit")
    )
    train_x, train_y, train_x_generator = _generate_training_data(
        target=dinf_model.target_func,
        generator=dinf_model.generator_func_v,
        thetas=training_thetas,
        parallelism=parallelism,
        ss=ss_train,
    )

    val_x, val_y, val_x_generator = None, None, None
    if test_thetas is not None and len(test_thetas) > 0:
        val_x, val_y, val_x_generator = _generate_training_data(
            target=dinf_model.target_func,
            generator=dinf_model.generator_func_v,
            thetas=test_thetas,
            parallelism=parallelism,
            ss=ss_val,
        )

    metrics = discriminator.fit(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        epochs=epochs,
        rng=np.random.default_rng(ss_fit),
        # Clear the training loss/accuracy metrics from last iteration.
        reset_metrics=True,
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
    dinf_model: DinfModel,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    parallelism: None | int = None,
    seed: None | int = None,
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
    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, dinf_model)

    ss = NamedSeedSequence(seed)
    ss_train, ss_test, ss_discr_init = ss.spawn(
        ("thetas:train", "thetas:test", "discriminator:init")
    )

    training_thetas = dinf_model.parameters.draw_prior(
        training_replicates // 2, rng=np.random.default_rng(ss_train)
    )
    test_thetas = dinf_model.parameters.draw_prior(
        test_replicates // 2, rng=np.random.default_rng(ss_test)
    )

    discriminator = Discriminator(
        dinf_model.feature_shape, network=dinf_model.discriminator_network
    ).init(np.random.default_rng(ss_discr_init))
    # discriminator.summary()

    _train_discriminator(
        discriminator=discriminator,
        dinf_model=dinf_model,
        training_thetas=training_thetas,
        test_thetas=test_thetas,
        epochs=epochs,
        parallelism=parallelism,
        ss=ss,
    )
    return discriminator


@cleanup_process_pool_afterwards
def predict(
    *,
    discriminator: Discriminator,
    dinf_model: DinfModel,
    replicates: int,
    sample_target: bool = False,
    parallelism: None | int = None,
    seed: int | None = None,
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
    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, dinf_model)
    ss = NamedSeedSequence(seed)

    if sample_target:
        (ss_target,) = ss.spawn(("features:predict:target",))
        x = _observe_data(
            target=dinf_model.target_func,
            num_replicates=replicates,
            parallelism=parallelism,
            rng=np.random.default_rng(ss_target),
        )
        thetas = None
    else:
        ss_thetas, ss_generator = ss.spawn(
            ("thetas:predict", "features:predict:generator")
        )
        thetas = dinf_model.parameters.draw_prior(
            replicates, rng=np.random.default_rng(ss_thetas)
        )
        x = _generate_data(
            generator=dinf_model.generator_func_v,
            thetas=thetas,
            parallelism=parallelism,
            rng=np.random.default_rng(ss_generator),
        )
    probs = discriminator.predict(x)
    return thetas, probs


@cleanup_process_pool_afterwards
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
    working_directory: None | str | pathlib.Path = None,
    parallelism: None | int = None,
    seed: None | int = None,
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
    :param working_directory:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`DinfModel.generator_func` and
        :meth:`DinfModel.target_func`.
    :param seed:
        Seed for the random number generator.
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

    ss = NamedSeedSequence(seed)
    ss_loop, ss_mcmc, ss_thetas, ss_discr_init = ss.spawn(
        ("mcmc-gan:loop", "mcmc-gan:mcmc", "abc-gan:thetas", "discriminator:init")
    )
    rng_thetas = np.random.default_rng(ss_thetas)
    rng_mcmc = np.random.default_rng(ss_mcmc)

    _process_pool_init(parallelism, dinf_model)
    parameters = dinf_model.parameters
    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists() for fn in ("discriminator.nn", "mcmc.npz")
        ]
        if sum(files_exist) == 1:
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    discriminator = Discriminator(
        dinf_model.feature_shape, network=dinf_model.discriminator_network
    )
    if resume:
        discriminator = discriminator.from_file(store[-1] / "discriminator.nn")
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

        sampled_thetas = rng_thetas.choice(
            thetas.reshape(-1, thetas.shape[-1]), size=num_replicates, replace=False
        )
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]
    else:
        discriminator = discriminator.init(np.random.default_rng(ss_discr_init))
        # Starting point for the mcmc chain.
        start = parameters.draw_prior(walkers, rng=rng_thetas)

        training_thetas = parameters.draw_prior(
            training_replicates // 2, rng=rng_thetas
        )
        test_thetas = parameters.draw_prior(test_replicates // 2, rng=rng_thetas)

    # If start values are linearly dependent, emcee complains loudly.
    assert not np.any((start[0] == start[1:]).all(axis=-1))

    n_target_calls = 0
    n_generator_calls = 0

    log_prob_func = functools.partial(
        _log_prob,
        discriminator=discriminator,
        generator=dinf_model.generator_func_v,
        parameters=parameters,
        parallelism=parallelism,
        num_replicates=Dx_replicates,
        rng=rng_thetas,
    )

    for i in range(len(store) + 1, len(store) + 1 + iterations):
        print(f"MCMC GAN iteration {i}")
        store.increment()

        (ss_loop,) = ss_loop.spawn(1)
        _train_discriminator(
            discriminator=discriminator,
            dinf_model=dinf_model,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            ss=ss_loop,
        )
        discriminator.to_file(store[-1] / "discriminator.nn")

        thetas, probs = _run_mcmc_emcee(
            start=start,
            parameters=parameters,
            walkers=walkers,
            steps=steps,
            rng=rng_mcmc,
            log_prob_func=log_prob_func,
        )
        assert thetas.shape == (steps, walkers, len(parameters))
        save_results(
            store[-1] / "mcmc.npz", thetas=thetas, probs=probs, parameters=parameters
        )

        # The chain for next iteration starts at the end of this chain.
        start = thetas[-1]

        sampled_thetas = rng_thetas.choice(
            thetas.reshape(-1, thetas.shape[-1]), size=num_replicates, replace=False
        )
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]

        n_target_calls += num_replicates
        n_generator_calls += num_replicates + walkers * steps * Dx_replicates
        print(f"Target called {n_target_calls} times.")
        print(f"Generator called {n_generator_calls} times.")


def _sample_smooth(*, thetas, probs, size: int, rng):
    """
    Sample from a smoothed set of weighted observations.
    """
    # Weighted sampling of points from the thetas.
    sample = rng.choice(thetas, size=size, replace=True, p=probs / np.sum(probs))
    # Calculate bandwidth.
    _, d = thetas.shape
    neff = np.sum(probs) ** 2 / np.sum(probs**2)
    bw_scott = neff ** (-1.0 / (d + 4))  # bandwidth multiplier
    cov = bw_scott * np.cov(thetas, rowvar=False, aweights=probs)
    assert not np.any(np.isnan(cov))
    assert not np.any(np.isinf(cov))
    # Jitter the sample with an MVN.
    sample += rng.multivariate_normal(np.zeros(d), cov, size=size)
    return sample


def sample_smooth(
    *,
    thetas: np.ndarray,
    probs: np.ndarray,
    size: int,
    rng: np.random.Generator,
    parameters: Parameters | None = None,
    mode: str | None = None,
) -> np.ndarray:
    """
    Sample from a smoothed set of weighted observations.

    Samples are drawn from ``thetas``, weighted by their probability.
    New points are drawn within a neighbourhood of the sampled thetas
    using a mulivariate normal whose covariance is calculated from the
    thetas. This is equivalent to sampling from a Gaussian KDE, but
    avoids doing an explicit density estimation.
    Scott's rule of thumb is used for bandwidth selection.

    :param thetas:
        Parameter values to sample from.
    :param probs:
        Discriminator predictions corresponding to the ``thetas``.
    :param size:
        Number of samples to draw.
    :param numpy.random.Generator rng:
        Numpy random generator.
    :param parameters:
        The parameters against which the values' bounds will be checked.
        See the ``mode`` argument.
    :param mode:
        The mode determines how to deal with values that are out of the
        parameter bounds. If mode is not None, then ``parameters`` must
        also be specified. The options are:

         * ``None`` (*default*): the returned values are not modified
           after sampling and may be out of bounds.
         * "transform": thetas are transformed before sampling, and
           the sampled values are inverse-transformed before being
           returned.
           See :meth:`Parameters.transform` and :meth:`Parameters.itransform`.
         * "truncate": sampled values are truncated at the parameter limits.
           See :meth:`Parameters.truncate`.
         * "reflect": sample values that are out of bounds are reflected
           inside the parameter limits by the same magnitude that they were
           out of bounds. Values that are too far out of bounds to be
           reflected are truncated at the parameter limits.
           See :meth:`Parameters.reflect`.

    :return:
        The sampled values.
    """
    if mode is not None:
        if mode not in ("transform", "truncate", "reflect"):
            raise ValueError(f"Unknown sampling mode '{mode}'")
        if parameters is None:
            raise ValueError("Must pass 'parameters' when 'mode' is not None")
    if mode == "transform":
        assert parameters is not None
        thetas = parameters.transform(thetas)
    X = _sample_smooth(thetas=thetas, probs=probs, size=size, rng=rng)
    if mode == "transform":
        assert parameters is not None
        X = parameters.itransform(X)
    elif mode == "reflect":
        assert parameters is not None
        X = parameters.reflect(X)
    elif mode == "truncate":
        assert parameters is not None
        parameters.truncate(X)
    return X


def filter_top_n(
    thetas: np.ndarray,
    probs: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    assert n >= 1
    k = len(probs) - n
    assert k >= 1
    idx = np.argpartition(probs, k)[k:]
    return thetas[idx], probs[idx]


@cleanup_process_pool_afterwards
def abc_gan(
    *,
    dinf_model: DinfModel,
    iterations: int,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    top_n: int | None = None,
    working_directory: None | str | pathlib.Path = None,
    parallelism: None | int = None,
    seed: None | int = None,
):
    """
    Run the ABC GAN.

    Conceptually, the GAN takes the following steps for iteration j:

      - sample train/test datasets from the prior[j] distribution,
      - train the discriminator,
      - make predictions with the discriminator,
      - construct a posterior[j] sample from the test dataset,
      - set prior[j+1] = posterior[j].

    In the first iteration, the parameter values given to the generator
    to produce the test/train datasets are drawn from the parameters'
    prior distribution. In subsequent iterations, the parameter values
    are drawn from a posterior ABC sample. The posterior is obtained by
    weighting the prior distribution by the discriminator predictions,
    followed by gaussian smoothing.

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
    :param top_n:
        If not None, do ABC rejection sampling in each iteraction by taking
        the ``top_n`` best samples, ranked by discriminator prediction.
        Samples are taken from the test replicates, so ``top_n`` must be
        smaller than ``test_replicates``.
    :param working_directory:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`DinfModel.generator_func` and
        :meth:`DinfModel.target_func`.
    :param seed:
        Seed for the random number generator.
    """

    if test_replicates < 2:
        raise ValueError(
            "Must have test_replicates>=2. Test replicates are (also) used "
            "for sampling the parameter values for the next GAN iteration."
        )

    num_replicates = math.ceil((training_replicates + test_replicates) / 2)

    if top_n is not None and top_n >= test_replicates // 2:
        raise ValueError(f"{top_n=}, but {test_replicates=}")

    if working_directory is None:
        working_directory = "."
    store = Store(working_directory, create=True)

    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    ss = NamedSeedSequence(seed)
    ss_loop, ss_thetas, ss_discr_init = ss.spawn(
        ("abc-gan:loop", "abc-gan:thetas", "discriminator:init")
    )
    rng_thetas = np.random.default_rng(ss_thetas)

    _process_pool_init(parallelism, dinf_model)

    parameters = dinf_model.parameters
    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists() for fn in ("discriminator.nn", "abc.npz")
        ]
        if sum(files_exist) == 1:
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    # Use mode="reflect", because "transform" seems to produce ABC-GAN
    # degenerate states due to density accumulation at the bounds.
    # This effect was even more pronouned than for "truncate".
    sampling_mode = "reflect"

    discriminator = Discriminator(
        dinf_model.feature_shape, network=dinf_model.discriminator_network
    )
    if resume:
        discriminator = discriminator.from_file(store[-1] / "discriminator.nn")
        thetas, y = _load_results_unstructured(
            store[-1] / "abc.npz", parameters=parameters
        )
        assert len(thetas.shape) == 2
        if top_n is not None:
            thetas, y = filter_top_n(thetas, y, top_n)
        sampled_thetas = sample_smooth(
            thetas=thetas,
            probs=y,
            size=num_replicates,
            rng=rng_thetas,
            parameters=parameters,
            mode=sampling_mode,
        )
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]
    else:
        discriminator = discriminator.init(np.random.default_rng(ss_discr_init))
        training_thetas = parameters.draw_prior(
            training_replicates // 2, rng=rng_thetas
        )
        test_thetas = parameters.draw_prior(test_replicates // 2, rng=rng_thetas)

    n_target_calls = 0
    n_generator_calls = 0

    for i in range(len(store) + 1, len(store) + 1 + iterations):
        print(f"ABC GAN iteration {i}")
        store.increment()

        (ss_loop,) = ss_loop.spawn(1)
        _, _, test_x_generator = _train_discriminator(
            discriminator=discriminator,
            dinf_model=dinf_model,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            ss=ss_loop,
        )
        discriminator.to_file(store[-1] / "discriminator.nn")

        y = discriminator.predict(test_x_generator)
        save_results(
            store[-1] / "abc.npz",
            probs=y,
            thetas=test_thetas,
            parameters=parameters,
        )

        # Get the posterior sample for the next iteration.
        thetas = test_thetas
        if top_n is not None:
            thetas, y = filter_top_n(thetas, y, top_n)
        sampled_thetas = sample_smooth(
            thetas=thetas,
            probs=y,
            size=num_replicates,
            rng=rng_thetas,
            parameters=parameters,
            mode=sampling_mode,
        )
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]

        n_target_calls += num_replicates
        n_generator_calls += num_replicates
        print(f"Target called {n_target_calls} times.")
        print(f"Generator called {n_generator_calls} times.")


def pretraining_pg_gan(
    *,
    dinf_model,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    parallelism: int,
    max_pretraining_iterations: int,
    ss: NamedSeedSequence,
):
    """
    Pretraining roughly like PG-GAN.

    PG-GAN starts at a point in the parameter space that is favourable
    to the discriminator, and it only trains the discriminator enough
    to identify such a point. PG-GAN does up to 10 iterations, with
    40000/40000 (real/fake) reps each time. Here, we iterate up to
    max_pretraining_iterations times, each with training_replicates reps.
    """

    ss_discr_init, ss_thetas = ss.spawn(("discriminator:init", "thetas"))
    rng_thetas = np.random.default_rng(ss_thetas)

    discriminator = Discriminator(
        dinf_model.feature_shape, network=dinf_model.discriminator_network
    ).init(np.random.default_rng(ss_discr_init))
    acc_best = 0
    theta_best = None

    for k in range(max_pretraining_iterations):
        theta = dinf_model.parameters.draw_prior(1, rng=rng_thetas)[0]
        training_thetas = np.tile(theta, (training_replicates // 2, 1))
        test_thetas = np.tile(theta, (test_replicates // 2, 1))

        metrics, _, _ = _train_discriminator(
            discriminator=discriminator,
            dinf_model=dinf_model,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            ss=ss,
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
    parallelism: int,
    max_pretraining_iterations: int,
    ss: NamedSeedSequence,
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
    ss_discr_init, ss_thetas = ss.spawn(("discriminator:init", "thetas"))
    rng = np.random.default_rng(ss_thetas)

    parameters = dinf_model.parameters
    discriminator = Discriminator(
        dinf_model.feature_shape, network=dinf_model.discriminator_network
    ).init(np.random.default_rng(ss_discr_init))

    for k in range(max_pretraining_iterations):
        training_thetas = parameters.draw_prior(training_replicates // 2, rng=rng)
        test_thetas = parameters.draw_prior(test_replicates // 2, rng=rng)

        metrics, _, _ = _train_discriminator(
            discriminator=discriminator,
            dinf_model=dinf_model,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            ss=ss,
        )

        # Use the test accuracy, unless there's no test data.
        acc = metrics.get("test_accuracy", metrics["train_accuracy"])
        if acc > 0.9:
            break

    thetas = parameters.draw_prior(training_replicates, rng=rng)
    lp = _log_prob(
        thetas,
        discriminator=discriminator,
        generator=dinf_model.generator_func_v,
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
    working_directory: None | str | pathlib.Path = None,
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
    :param working_directory:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`DinfModel.generator_func` and
        :meth:`DinfModel.target_func`.
    :param seed:
        Seed for the random number generator.
    """
    num_replicates = training_replicates // 2 + test_replicates // 2

    if working_directory is None:
        working_directory = "."
    store = Store(working_directory, create=True)

    if parallelism is None:
        parallelism = cast(int, os.cpu_count())

    _process_pool_init(parallelism, dinf_model)

    ss = NamedSeedSequence(seed)
    ss_accept, ss_pretraining, ss_proposals, ss_loop = ss.spawn(
        ("pg-gan:acceptance", "pg-gan:pretraining", "pg-gan:proposals", "pg-gan:loop")
    )
    rng_accept = np.random.default_rng(ss_accept)
    rng_proposals = np.random.default_rng(ss_proposals)

    parameters = dinf_model.parameters
    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists()
            for fn in ("discriminator.nn", "pg-gan-proposals.npz")
        ]
        if sum(files_exist) == 1:
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    if resume:
        discriminator = Discriminator(
            dinf_model.feature_shape, network=dinf_model.discriminator_network
        ).from_file(store[-1] / "discriminator.nn")
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
            dinf_model=dinf_model,
            training_replicates=training_replicates,
            test_replicates=test_replicates,
            epochs=epochs,
            parallelism=parallelism,
            max_pretraining_iterations=max_pretraining_iterations,
            ss=ss_pretraining,
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
            parallelism=parallelism,
            rng=rng_proposals,
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
        U = rng_accept.uniform()
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
            (ss_loop,) = ss_loop.spawn(1)
            _train_discriminator(
                discriminator=discriminator,
                dinf_model=dinf_model,
                training_thetas=training_thetas,
                test_thetas=test_thetas,
                epochs=epochs,
                parallelism=parallelism,
                ss=ss_loop,
                # XXX: Is this helpful?
                entropy_regularisation=True,
            )
            n_target_calls += num_replicates
            n_generator_calls += num_replicates

        discriminator.to_file(store[-1] / "discriminator.nn")

        print(f"Target called {n_target_calls} times.")
        print(f"Generator called {n_generator_calls} times.")
        print()


@cleanup_process_pool_afterwards
def alfi_mcmc_gan(
    *,
    dinf_model: DinfModel,
    iterations: int,
    training_replicates: int,
    test_replicates: int,
    epochs: int,
    walkers: int,
    steps: int,
    working_directory: None | str | pathlib.Path = None,
    parallelism: None | int = None,
    seed: None | int = None,
):
    """
    Run the ALFI MCMC GAN.

    This behaves similarly to :func:`mcmc_gan`, but introduces a surrogate
    neural network that predicts the output of the discriminator from a given
    set of input parameter values. The predictions of the surrogate network
    are used during MCMC sampling, thus bypassing the generator and producing
    deterministic classification of the input parameters.

    Kim et al. 2020, https://arxiv.org/abs/2004.05803v1

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
    :param walkers:
        Number of independent MCMC chains.
    :param steps:
        The chain length for each MCMC walker.
    :param working_directory:
        Folder to output results. If not specified, the current
        directory will be used.
    :param parallelism:
        Number of processes to use for parallelising calls to the
        :meth:`DinfModel.generator_func` and
        :meth:`DinfModel.target_func`.
    :param seed:
        Seed for the random number generator.
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

    _process_pool_init(parallelism, dinf_model)

    ss = NamedSeedSequence(seed)
    ss_loop, ss_mcmc, ss_thetas, ss_discr_init, ss_surrogate_init = ss.spawn(
        (
            "mcmc-gan:loop",
            "mcmc-gan:mcmc",
            "abc-gan:thetas",
            "discriminator:init",
            "surrogate:init",
        )
    )
    rng_thetas = np.random.default_rng(ss_thetas)
    rng_mcmc = np.random.default_rng(ss_mcmc)

    parameters = dinf_model.parameters
    resume = False
    if len(store) > 0:
        files_exist = [
            (store[-1] / fn).exists()
            for fn in ("discriminator.nn", "surrogate.nn", "mcmc.npz")
        ]
        if sum(files_exist) not in (0, len(files_exist)):
            raise RuntimeError(f"{store[-1]} is incomplete. Delete and try again?")
        resume = all(files_exist)

    discriminator = Discriminator(
        dinf_model.feature_shape, network=dinf_model.discriminator_network
    )
    surrogate = Surrogate(len(parameters))
    if resume:
        discriminator = discriminator.from_file(store[-1] / "discriminator.nn")
        surrogate = surrogate.from_file(store[-1] / "surrogate.nn")
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

        sampled_thetas = rng_thetas.choice(
            thetas.reshape(-1, thetas.shape[-1]), size=num_replicates, replace=False
        )
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]
    else:
        discriminator = discriminator.init(np.random.default_rng(ss_discr_init))
        surrogate = surrogate.init(np.random.default_rng(ss_surrogate_init))
        # Starting point for the mcmc chain.
        start = parameters.draw_prior(walkers, rng=rng_thetas)

        training_thetas = parameters.draw_prior(
            training_replicates // 2, rng=rng_thetas
        )
        test_thetas = parameters.draw_prior(test_replicates // 2, rng=rng_thetas)

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

        (ss_loop,) = ss_loop.spawn(1)
        _, train_x_generator, test_x_generator = _train_discriminator(
            discriminator=discriminator,
            dinf_model=dinf_model,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            parallelism=parallelism,
            ss=ss_loop,
        )
        n_target_calls += num_replicates
        n_generator_calls += num_replicates
        discriminator.to_file(store[-1] / "discriminator.nn")

        train_y_pred, alpha, beta = _train_surrogate(
            discriminator=discriminator,
            surrogate=surrogate,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=epochs,
            train_x_generator=train_x_generator,
            test_x_generator=test_x_generator,
        )
        surrogate.to_file(store[-1] / "surrogate.nn")

        # import pickle
        # with open(store[-1] / "s.nn", "wb") as f:
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
            rng=rng_mcmc,
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

        sampled_thetas = rng_thetas.choice(
            thetas.reshape(-1, thetas.shape[-1]), size=num_replicates, replace=False
        )
        training_thetas = sampled_thetas[: training_replicates // 2]
        test_thetas = sampled_thetas[training_replicates // 2 :]

        # n_target_calls += num_replicates
        # n_generator_calls += num_replicates
        print(f"Target called {n_target_calls} times.")
        print(f"Generator called {n_generator_calls} times.")
