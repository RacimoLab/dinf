from __future__ import annotations
import copy
import functools
import itertools
import os
import pathlib
from typing import Callable

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import pytest

import dinf
import examples.bottleneck.model  # type: ignore[import]


def get_dinf_model():
    return copy.deepcopy(examples.bottleneck.model.dinf_model)


def check_discriminator(filename: str | pathlib.Path, dinf_model: dinf.DinfModel):
    dinf.Discriminator(
        dinf_model.feature_shape, network=dinf_model.discriminator_network
    ).from_file(filename)


def check_npz(
    filename: str | pathlib.Path,
    *,
    chains: int,
    draws: int,
    parameters: dinf.Parameters,
):
    data = dinf.load_results(filename, parameters=parameters)
    if chains == 1:
        assert data.shape == (draws,)
    else:
        assert data.shape == (draws, chains)
    return data


@pytest.mark.parametrize("top_n", [None, 2])
@pytest.mark.usefixtures("tmp_path")
def test_abc_gan(tmp_path, top_n):
    dinf_model = get_dinf_model()
    working_directory = tmp_path / "work_dir"
    dinf.dinf.abc_gan(
        dinf_model=dinf_model,
        iterations=2,
        training_replicates=8,
        test_replicates=8,
        top_n=top_n,
        epochs=1,
        working_directory=working_directory,
        parallelism=2,
        seed=1,
    )
    assert working_directory.exists()
    for i in range(2):
        check_discriminator(working_directory / f"{i}" / "discriminator.nn", dinf_model)
        check_npz(
            working_directory / f"{i}" / "abc.npz",
            chains=1,
            draws=4,
            parameters=dinf_model.parameters,
        )

    # resume
    os.chdir(working_directory)
    dinf.dinf.abc_gan(
        dinf_model=dinf_model,
        iterations=1,
        training_replicates=8,
        test_replicates=8,
        top_n=top_n,
        epochs=1,
        seed=2,
    )
    for i in range(3):
        check_discriminator(working_directory / f"{i}" / "discriminator.nn", dinf_model)
        check_npz(
            working_directory / f"{i}" / "abc.npz",
            chains=1,
            draws=4,
            parameters=dinf_model.parameters,
        )

    with pytest.raises(ValueError, match="Must have test_replicates"):
        dinf.dinf.abc_gan(
            dinf_model=dinf_model,
            iterations=2,
            training_replicates=8,
            test_replicates=0,
            epochs=1,
            working_directory=working_directory,
            parallelism=2,
            seed=1,
        )

    with pytest.raises(ValueError, match="top_n"):
        dinf.dinf.abc_gan(
            dinf_model=dinf_model,
            iterations=2,
            training_replicates=8,
            test_replicates=8,
            top_n=4,
            epochs=1,
            working_directory=working_directory,
            parallelism=2,
            seed=1,
        )

    backup = working_directory / "bak"
    for file in [
        working_directory / f"{i}" / "discriminator.nn",
        working_directory / f"{i}" / "abc.npz",
    ]:
        file.rename(backup)
        with pytest.raises(RuntimeError, match="incomplete"):
            dinf.dinf.abc_gan(
                dinf_model=dinf_model,
                iterations=2,
                training_replicates=8,
                test_replicates=8,
                epochs=1,
                working_directory=working_directory,
                parallelism=2,
                seed=1,
            )
        backup.rename(file)


@pytest.mark.usefixtures("tmp_path")
def test_mcmc_gan(tmp_path):
    dinf_model = get_dinf_model()
    working_directory = tmp_path / "workdir"
    dinf.mcmc_gan(
        dinf_model=dinf_model,
        iterations=2,
        training_replicates=10,
        test_replicates=0,
        epochs=1,
        walkers=6,
        steps=1,
        Dx_replicates=2,
        working_directory=working_directory,
        parallelism=2,
        seed=1,
    )
    assert working_directory.exists()
    for i in range(2):
        check_discriminator(working_directory / f"{i}" / "discriminator.nn", dinf_model)
        check_npz(
            working_directory / f"{i}" / "mcmc.npz",
            chains=6,
            draws=1,
            parameters=dinf_model.parameters,
        )

    # resume
    os.chdir(working_directory)
    dinf.mcmc_gan(
        dinf_model=dinf_model,
        iterations=1,
        training_replicates=10,
        test_replicates=2,
        epochs=1,
        walkers=6,
        steps=1,
        Dx_replicates=2,
        seed=2,
    )
    for i in range(3):
        check_discriminator(working_directory / f"{i}" / "discriminator.nn", dinf_model)
        check_npz(
            working_directory / f"{i}" / "mcmc.npz",
            chains=6,
            draws=1,
            parameters=dinf_model.parameters,
        )

    with pytest.raises(ValueError, match="resuming from .* which used .* walkers"):
        dinf.mcmc_gan(
            dinf_model=dinf_model,
            iterations=2,
            training_replicates=10,
            test_replicates=0,
            epochs=1,
            walkers=8,
            steps=1,
            Dx_replicates=2,
            working_directory=working_directory,
            parallelism=2,
            seed=1,
        )

    with pytest.raises(ValueError, match="Insufficient MCMC samples"):
        dinf.mcmc_gan(
            dinf_model=dinf_model,
            iterations=2,
            training_replicates=100,
            test_replicates=0,
            epochs=1,
            walkers=6,
            steps=1,
            Dx_replicates=2,
            working_directory=working_directory,
            parallelism=2,
            seed=1,
        )

    backup = working_directory / "bak"
    for file in [
        working_directory / f"{i}" / "discriminator.nn",
        working_directory / f"{i}" / "mcmc.npz",
    ]:
        file.rename(backup)
        with pytest.raises(RuntimeError, match="incomplete"):
            dinf.mcmc_gan(
                dinf_model=dinf_model,
                iterations=2,
                training_replicates=10,
                test_replicates=0,
                epochs=1,
                walkers=6,
                steps=1,
                Dx_replicates=2,
                working_directory=working_directory,
                parallelism=2,
                seed=1,
            )
        backup.rename(file)


@pytest.mark.usefixtures("tmp_path")
def test_alfi_mcmc_gan(tmp_path):
    dinf_model = get_dinf_model()
    working_directory = tmp_path / "workdir"
    steps = 1
    dinf.alfi_mcmc_gan(
        dinf_model=dinf_model,
        iterations=2,
        training_replicates=10,
        test_replicates=0,
        epochs=1,
        walkers=6,
        steps=steps,
        working_directory=working_directory,
        parallelism=2,
        seed=1,
    )
    assert working_directory.exists()
    for i in range(2):
        check_discriminator(working_directory / f"{i}" / "discriminator.nn", dinf_model)
        check_npz(
            working_directory / f"{i}" / "mcmc.npz",
            chains=6,
            draws=2 * steps,
            parameters=dinf_model.parameters,
        )

    # resume
    os.chdir(working_directory)
    dinf.alfi_mcmc_gan(
        dinf_model=dinf_model,
        iterations=1,
        training_replicates=4,
        test_replicates=4,
        epochs=1,
        walkers=6,
        steps=steps,
        seed=2,
    )
    for i in range(3):
        check_discriminator(working_directory / f"{i}" / "discriminator.nn", dinf_model)
        check_npz(
            working_directory / f"{i}" / "mcmc.npz",
            chains=6,
            draws=2 * steps,
            parameters=dinf_model.parameters,
        )

    with pytest.raises(ValueError, match="resuming from .* which used .* walkers"):
        dinf.alfi_mcmc_gan(
            dinf_model=dinf_model,
            iterations=2,
            training_replicates=10,
            test_replicates=0,
            epochs=1,
            walkers=8,
            steps=1,
            working_directory=working_directory,
            parallelism=2,
            seed=1,
        )

    with pytest.raises(ValueError, match="Insufficient MCMC samples"):
        dinf.alfi_mcmc_gan(
            dinf_model=dinf_model,
            iterations=2,
            training_replicates=100,
            test_replicates=0,
            epochs=1,
            walkers=6,
            steps=1,
            working_directory=working_directory,
            parallelism=2,
            seed=1,
        )

    backup = working_directory / "bak"
    for file in [
        working_directory / f"{i}" / "discriminator.nn",
        working_directory / f"{i}" / "surrogate.nn",
        working_directory / f"{i}" / "mcmc.npz",
    ]:
        file.rename(backup)
        with pytest.raises(RuntimeError, match="incomplete"):
            dinf.alfi_mcmc_gan(
                dinf_model=dinf_model,
                iterations=2,
                training_replicates=10,
                test_replicates=0,
                epochs=1,
                walkers=6,
                steps=1,
                working_directory=working_directory,
                parallelism=2,
                seed=1,
            )
        backup.rename(file)


@pytest.mark.usefixtures("tmp_path")
def test_pg_gan(tmp_path):
    dinf_model = get_dinf_model()
    num_params = len(dinf_model.parameters)
    num_proposals = 2
    working_directory = tmp_path / "workdir"
    dinf.pg_gan(
        dinf_model=dinf_model,
        iterations=2,
        training_replicates=10,
        test_replicates=0,
        epochs=1,
        Dx_replicates=2,
        num_proposals=num_proposals,
        max_pretraining_iterations=1,
        working_directory=working_directory,
        parallelism=2,
        seed=1,
    )
    assert working_directory.exists()
    for i in range(2):
        check_discriminator(working_directory / f"{i}" / "discriminator.nn", dinf_model)
        check_npz(
            working_directory / f"{i}" / "pg-gan-proposals.npz",
            chains=1,
            draws=num_params * num_proposals + 1,
            parameters=dinf_model.parameters,
        )

    # resume
    os.chdir(working_directory)
    dinf.pg_gan(
        dinf_model=dinf_model,
        iterations=1,
        training_replicates=10,
        test_replicates=2,
        epochs=1,
        Dx_replicates=2,
        num_proposals=num_proposals,
        seed=2,
    )
    for i in range(3):
        check_discriminator(working_directory / f"{i}" / "discriminator.nn", dinf_model)
        check_npz(
            working_directory / f"{i}" / "pg-gan-proposals.npz",
            chains=1,
            draws=num_params * num_proposals + 1,
            parameters=dinf_model.parameters,
        )

    backup = working_directory / "bak"
    for file in [
        working_directory / f"{i}" / "discriminator.nn",
        working_directory / f"{i}" / "pg-gan-proposals.npz",
    ]:
        file.rename(backup)
        with pytest.raises(RuntimeError, match="incomplete"):
            dinf.pg_gan(
                dinf_model=dinf_model,
                iterations=2,
                training_replicates=10,
                test_replicates=0,
                epochs=1,
                Dx_replicates=2,
                num_proposals=num_proposals,
                working_directory=working_directory,
                parallelism=2,
                seed=3,
            )
        backup.rename(file)

    # TODO: check the right pretraining function is called.
    for pretraining_method in ("pg-gan", "dinf"):
        dinf.pg_gan(
            dinf_model=dinf_model,
            iterations=0,
            training_replicates=10,
            test_replicates=0,
            epochs=1,
            Dx_replicates=2,
            num_proposals=num_proposals,
            pretraining_method=pretraining_method,
            max_pretraining_iterations=1,
            working_directory=tmp_path / pretraining_method,
            parallelism=2,
            seed=4,
        )
    with pytest.raises(ValueError, match="pretraining_method"):
        dinf.pg_gan(
            dinf_model=dinf_model,
            iterations=0,
            training_replicates=10,
            test_replicates=0,
            epochs=1,
            Dx_replicates=2,
            num_proposals=num_proposals,
            pretraining_method="not-a-valid-method",
            max_pretraining_iterations=1,
            working_directory=tmp_path / "somewhere-new",
            parallelism=2,
            seed=1,
        )

    # TODO: check the right proposals function is called.
    for proposals_method in ("pg-gan", "rr", "mvn"):
        dinf.pg_gan(
            dinf_model=dinf_model,
            iterations=1,
            training_replicates=10,
            test_replicates=0,
            epochs=1,
            Dx_replicates=2,
            num_proposals=num_proposals,
            proposals_method=proposals_method,
            working_directory=working_directory,
            parallelism=2,
            seed=1,
        )
    with pytest.raises(ValueError, match="proposals_method"):
        dinf.pg_gan(
            dinf_model=dinf_model,
            iterations=1,
            training_replicates=10,
            test_replicates=0,
            epochs=1,
            Dx_replicates=2,
            num_proposals=num_proposals,
            proposals_method="not-a-valid-method",
            working_directory=working_directory,
            parallelism=2,
            seed=1,
        )


def elementwise_allclose(array):
    """
    Return True if elements of the array are all close, False otherwise.
    """
    for i in range(len(array) - 1):
        if not np.allclose(array[i], array[i + 1 :]):
            return False
    else:
        return True


def check_proposals(proposal_thetas, theta, dinf_model, num_proposals):
    num_params = len(dinf_model.parameters)
    assert proposal_thetas.shape == (num_proposals * num_params + 1, num_params)
    np.testing.assert_array_equal(theta, proposal_thetas[0])
    for i, p in enumerate(dinf_model.parameters.values()):
        # All proposals should be within the bounds.
        assert np.all(p.bounds_contain(proposal_thetas[:, i]))


def test_sanneal_proposals_pg_gan():
    rng = np.random.default_rng(4321)
    dinf_model = get_dinf_model()
    num_proposals = 10

    for i in range(100):
        theta = dinf_model.parameters.draw_prior(1, rng=rng)[0]
        proposal_thetas = dinf.dinf.sanneal_proposals_pg_gan(
            theta=theta,
            temperature=1,
            num_proposals=num_proposals,
            parameters=dinf_model.parameters,
            proposal_stddev=1 / 15,
            rng=rng,
        )
        check_proposals(proposal_thetas, theta, dinf_model, num_proposals)
        for i, p in enumerate(dinf_model.parameters.values()):
            assert not elementwise_allclose(proposal_thetas[:, i])


def test_sanneal_proposals_rr():
    rng = np.random.default_rng(4321)
    dinf_model = get_dinf_model()
    num_proposals = 10
    num_params = len(dinf_model.parameters)

    iteration = itertools.count()
    for i in range(100):
        theta = dinf_model.parameters.draw_prior(1, rng=rng)[0]
        proposal_thetas = dinf.dinf.sanneal_proposals_rr(
            theta=theta,
            temperature=1,
            num_proposals=num_proposals,
            parameters=dinf_model.parameters,
            proposal_stddev=1 / 15,
            rng=rng,
            iteration=iteration,
        )
        check_proposals(proposal_thetas, theta, dinf_model, num_proposals)

        for j, p in enumerate(dinf_model.parameters.values()):
            # Check that all proposals are for only one parameter.
            if j == i % num_params:
                assert not elementwise_allclose(proposal_thetas[:, j]), (i, j)
            else:
                assert elementwise_allclose(proposal_thetas[:, j]), (i, j)


def test_sanneal_proposals_mvn():
    rng = np.random.default_rng(4321)
    dinf_model = get_dinf_model()
    num_proposals = 10

    for i in range(100):
        theta = dinf_model.parameters.draw_prior(1, rng=rng)[0]
        proposal_thetas = dinf.dinf.sanneal_proposals_mvn(
            theta=theta,
            temperature=1,
            num_proposals=num_proposals,
            parameters=dinf_model.parameters,
            proposal_stddev=1 / 15,
            rng=rng,
        )
        check_proposals(proposal_thetas, theta, dinf_model, num_proposals)
        for i, p in enumerate(dinf_model.parameters.values()):
            assert not elementwise_allclose(proposal_thetas[:, i])


def log_prob(
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
    Non-vector version of dinf.dinf._log_prob()
    (the function to be maximised by the mcmc).
    """
    assert len(theta) == len(parameters)
    if not parameters.bounds_contain(theta):
        # param out of bounds
        return -np.inf

    seeds = rng.integers(low=1, high=2**31, size=num_replicates)
    params = np.tile(theta, (num_replicates, 1))
    M = dinf.dinf._sim_replicates(
        sim_func=generator,
        args=zip(seeds, params),
        num_replicates=num_replicates,
        parallelism=parallelism,
    )
    D = np.mean(discriminator.predict(M))
    with np.errstate(divide="ignore"):
        return np.log(D)


class TestLogProb:
    @classmethod
    def setup_class(cls):
        cls.dinf_model = get_dinf_model()
        rng = np.random.default_rng(111)
        cls.discriminator = dinf.Discriminator(cls.dinf_model.feature_shape).init(rng)
        training_thetas = cls.dinf_model.parameters.draw_prior(100, rng=rng)
        test_thetas = cls.dinf_model.parameters.draw_prior(0, rng=rng)
        dinf.dinf._train_discriminator(
            discriminator=cls.discriminator,
            dinf_model=cls.dinf_model,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=1,
            parallelism=1,
            ss=dinf.dinf.NamedSeedSequence(1),
        )

    def test_log_prob(self):
        # non-vector version
        log_prob_1 = functools.partial(
            log_prob,
            discriminator=self.discriminator,
            generator=self.dinf_model.generator_func_v,
            parameters=self.dinf_model.parameters,
            rng=np.random.default_rng(1),
            num_replicates=2,
            parallelism=1,
        )
        # vector version
        log_prob_n = functools.partial(
            dinf.dinf._log_prob,
            discriminator=self.discriminator,
            generator=self.dinf_model.generator_func_v,
            parameters=self.dinf_model.parameters,
            rng=np.random.default_rng(1),
            num_replicates=2,
            parallelism=1,
        )

        parameters = tuple(self.dinf_model.parameters.values())
        true_params = [p.truth for p in parameters]
        thetas = np.array(
            [
                true_params,
                # out of bounds
                [parameters[0].low - 1] + true_params[1:],
                [parameters[0].high + 1] + true_params[1:],
                true_params[:-1] + [parameters[-1].low - 1],
                true_params[:-1] + [parameters[-1].high + 1],
                [p.low - 1 for p in parameters],
                [p.high + 1 for p in parameters],
            ]
        )

        log_D = log_prob_n(thetas)
        assert len(log_D) == len(thetas)
        assert np.exp(log_D[0]) > 0
        assert all(np.isinf(log_D[1:]))

        for j in range(len(thetas)):
            log_D_1 = log_prob_1(thetas[j])
            assert np.isclose(log_D_1, log_D[j])

        # Random thetas.
        thetas = self.dinf_model.parameters.draw_prior(
            20, rng=np.random.default_rng(123)
        )
        log_D = log_prob_n(thetas)
        assert len(log_D) == len(thetas)
        assert all(np.exp(log_D) > 0)

        for j in range(len(thetas)):
            log_D_1 = log_prob_1(thetas[j])
            assert np.isclose(log_D_1, log_D[j])

        # All out of bounds.
        thetas = np.array([[p.low - 1 for p in parameters] for _ in range(20)])
        log_D = log_prob_n(thetas)
        assert all(np.isinf(log_D))
        for j in range(len(thetas)):
            log_D_1 = log_prob_1(thetas[j])
            assert np.isinf(log_D_1)


@pytest.mark.usefixtures("tmp_path")
def test_save_load_results(tmp_path):
    size = 100
    params = dinf.Parameters(
        p0=dinf.Param(low=0, high=10), p1=dinf.Param(low=10, high=20)
    )
    thetas = params.draw_prior(size, rng=np.random.default_rng(123))
    y = np.zeros(size)
    file = tmp_path / "a.npz"
    dinf.save_results(file, thetas=thetas, probs=y, parameters=params)
    assert file.exists()

    data = dinf.load_results(file, parameters=params)
    names = list(data.dtype.names)
    assert names[0] == "_Pr"
    np.testing.assert_array_equal(y, data["_Pr"])

    thetas2 = structured_to_unstructured(data[names[1:]])
    np.testing.assert_array_equal(thetas, thetas2)


@pytest.mark.usefixtures("tmp_path")
def test_save_load_results_3d(tmp_path):
    size = 100
    params = dinf.Parameters(
        p0=dinf.Param(low=0, high=10), p1=dinf.Param(low=10, high=20)
    )
    thetas = params.draw_prior(size, rng=np.random.default_rng(123))
    y = np.zeros(size)

    # Reshape to 3 dims.
    thetas = thetas.reshape((-1, 5, len(params)))
    y = y.reshape((-1, 5))

    file = tmp_path / "a.npz"
    dinf.save_results(file, thetas=thetas, probs=y, parameters=params)
    assert file.exists()

    data = dinf.load_results(file, parameters=params)
    names = list(data.dtype.names)
    assert names[0] == "_Pr"
    np.testing.assert_array_equal(y, data["_Pr"])

    thetas2 = structured_to_unstructured(data[names[1:]])
    np.testing.assert_array_equal(thetas, thetas2)


@pytest.mark.usefixtures("tmp_path")
def test_save_results_bad_shape(tmp_path):
    size = 100
    params = dinf.Parameters(
        p0=dinf.Param(low=0, high=10), p1=dinf.Param(low=10, high=20)
    )
    thetas = params.draw_prior(size, rng=np.random.default_rng(123))
    y = np.zeros(size)
    file = tmp_path / "a.npz"
    with pytest.raises(ValueError, match="thetas.shape.*parameters"):
        dinf.save_results(file, thetas=thetas, probs=y, parameters=[])
    with pytest.raises(ValueError, match="thetas.shape.*probs.shape"):
        dinf.save_results(file, thetas=thetas, probs=y[:50], parameters=params)


@pytest.mark.usefixtures("tmp_path")
def test_load_results_bad_shape(tmp_path):
    size = 100
    params = dinf.Parameters(
        p0=dinf.Param(low=0, high=10), p1=dinf.Param(low=10, high=20)
    )
    thetas = params.draw_prior(size, rng=np.random.default_rng(123))
    y = np.zeros(size)
    file = tmp_path / "a.npz"
    dinf.save_results(file, thetas=thetas, probs=y, parameters=params)
    assert file.exists()

    with pytest.raises(ValueError, match="expected arrays.*my_param"):
        dinf.load_results(file, parameters=["my_param"])

    file2 = tmp_path / "b.npz"
    np.savez(file2, p0=thetas[..., 0], p1=thetas[..., 1])
    with pytest.raises(ValueError, match="expected array '_Pr'"):
        dinf.load_results(file2)
