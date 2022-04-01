from __future__ import annotations
import copy
import functools
import itertools
import os
import pathlib
from typing import Callable, Iterable

import arviz as az
import numpy as np
import pytest


import dinf
import examples.bottleneck.model  # type: ignore[import]


def get_genobuilder():
    return copy.deepcopy(examples.bottleneck.model.genobuilder)


def check_discriminator(filename: str | pathlib.Path):
    dinf.Discriminator.from_file(filename)


def check_ncf(
    filename: str | pathlib.Path,
    *,
    chains: int,
    draws: int,
    var_names: Iterable[str],
    check_acceptance_rate: bool,
):
    ds = az.from_netcdf(filename)
    assert len(ds.posterior.chain) == chains
    assert len(ds.posterior.draw) == draws
    np.testing.assert_array_equal(list(ds.posterior.data_vars.keys()), list(var_names))
    assert len(ds.sample_stats.lp.chain) == chains
    assert len(ds.sample_stats.lp.draw) == draws
    if check_acceptance_rate:
        assert "acceptance_rate" in ds.sample_stats


@pytest.mark.usefixtures("tmp_path")
def test_abc_gan(tmp_path):
    rng = np.random.default_rng(123)
    genobuilder = get_genobuilder()
    working_directory = tmp_path / "work_dir"
    dinf.dinf.abc_gan(
        genobuilder=genobuilder,
        iterations=2,
        training_replicates=16,
        test_replicates=0,
        epochs=1,
        proposals=20,
        posteriors=7,
        working_directory=working_directory,
        parallelism=2,
        rng=rng,
    )
    assert working_directory.exists()
    for i in range(2):
        check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
        check_ncf(
            working_directory / f"{i}" / "abc.ncf",
            chains=1,
            draws=7,
            var_names=genobuilder.parameters,
            check_acceptance_rate=False,
        )

    # resume
    os.chdir(working_directory)
    dinf.dinf.abc_gan(
        genobuilder=genobuilder,
        iterations=1,
        training_replicates=16,
        test_replicates=2,
        epochs=1,
        proposals=20,
        posteriors=7,
        rng=rng,
    )
    for i in range(3):
        check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
        check_ncf(
            working_directory / f"{i}" / "abc.ncf",
            chains=1,
            draws=7,
            var_names=genobuilder.parameters,
            check_acceptance_rate=False,
        )

    with pytest.raises(
        ValueError, match="Cannot subsample .* posteriors from .* proposals"
    ):
        dinf.dinf.abc_gan(
            genobuilder=genobuilder,
            iterations=2,
            training_replicates=16,
            test_replicates=0,
            epochs=1,
            proposals=20,
            posteriors=70,
            working_directory=working_directory,
            parallelism=2,
            rng=rng,
        )

    backup = working_directory / "bak"
    for file in [
        working_directory / f"{i}" / "discriminator.pkl",
        working_directory / f"{i}" / "abc.ncf",
    ]:
        file.rename(backup)
        with pytest.raises(RuntimeError, match="incomplete"):
            dinf.dinf.abc_gan(
                genobuilder=genobuilder,
                iterations=2,
                training_replicates=16,
                test_replicates=0,
                epochs=1,
                proposals=20,
                posteriors=7,
                working_directory=working_directory,
                parallelism=2,
                rng=rng,
            )
        backup.rename(file)


@pytest.mark.usefixtures("tmp_path")
def test_mcmc_gan(tmp_path):
    rng = np.random.default_rng(1234)
    genobuilder = get_genobuilder()
    working_directory = tmp_path / "workdir"
    dinf.mcmc_gan(
        genobuilder=genobuilder,
        iterations=2,
        training_replicates=10,
        test_replicates=0,
        epochs=1,
        walkers=6,
        steps=1,
        Dx_replicates=2,
        working_directory=working_directory,
        parallelism=2,
        rng=rng,
    )
    assert working_directory.exists()
    for i in range(2):
        check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
        check_ncf(
            working_directory / f"{i}" / "mcmc.ncf",
            chains=6,
            draws=1,
            var_names=genobuilder.parameters,
            check_acceptance_rate=True,
        )

    # resume
    os.chdir(working_directory)
    dinf.mcmc_gan(
        genobuilder=genobuilder,
        iterations=1,
        training_replicates=10,
        test_replicates=2,
        epochs=1,
        walkers=6,
        steps=1,
        Dx_replicates=2,
        rng=rng,
    )
    for i in range(3):
        check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
        check_ncf(
            working_directory / f"{i}" / "mcmc.ncf",
            chains=6,
            draws=1,
            var_names=genobuilder.parameters,
            check_acceptance_rate=True,
        )

    with pytest.raises(ValueError, match="resuming from .* which used .* walkers"):
        dinf.mcmc_gan(
            genobuilder=genobuilder,
            iterations=2,
            training_replicates=10,
            test_replicates=0,
            epochs=1,
            walkers=8,
            steps=1,
            Dx_replicates=2,
            working_directory=working_directory,
            parallelism=2,
            rng=rng,
        )

    with pytest.raises(ValueError, match="Insufficient MCMC samples"):
        dinf.mcmc_gan(
            genobuilder=genobuilder,
            iterations=2,
            training_replicates=100,
            test_replicates=0,
            epochs=1,
            walkers=6,
            steps=1,
            Dx_replicates=2,
            working_directory=working_directory,
            parallelism=2,
            rng=rng,
        )

    backup = working_directory / "bak"
    for file in [
        working_directory / f"{i}" / "discriminator.pkl",
        working_directory / f"{i}" / "mcmc.ncf",
    ]:
        file.rename(backup)
        with pytest.raises(RuntimeError, match="incomplete"):
            dinf.mcmc_gan(
                genobuilder=genobuilder,
                iterations=2,
                training_replicates=10,
                test_replicates=0,
                epochs=1,
                walkers=6,
                steps=1,
                Dx_replicates=2,
                working_directory=working_directory,
                parallelism=2,
                rng=rng,
            )
        backup.rename(file)


@pytest.mark.usefixtures("tmp_path")
def test_alfi_mcmc_gan(tmp_path):
    rng = np.random.default_rng(1234)
    genobuilder = get_genobuilder()
    working_directory = tmp_path / "workdir"
    steps = 1
    dinf.alfi_mcmc_gan(
        genobuilder=genobuilder,
        iterations=2,
        training_replicates=10,
        test_replicates=0,
        epochs=1,
        walkers=6,
        steps=steps,
        working_directory=working_directory,
        parallelism=2,
        rng=rng,
    )
    assert working_directory.exists()
    for i in range(2):
        check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
        check_ncf(
            working_directory / f"{i}" / "mcmc.ncf",
            chains=6,
            draws=2 * steps,
            var_names=genobuilder.parameters,
            check_acceptance_rate=True,
        )

    # resume
    os.chdir(working_directory)
    dinf.alfi_mcmc_gan(
        genobuilder=genobuilder,
        iterations=1,
        training_replicates=4,
        test_replicates=4,
        epochs=1,
        walkers=6,
        steps=steps,
        rng=rng,
    )
    for i in range(3):
        check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
        check_ncf(
            working_directory / f"{i}" / "mcmc.ncf",
            chains=6,
            draws=2 * steps,
            var_names=genobuilder.parameters,
            check_acceptance_rate=True,
        )

    with pytest.raises(ValueError, match="resuming from .* which used .* walkers"):
        dinf.alfi_mcmc_gan(
            genobuilder=genobuilder,
            iterations=2,
            training_replicates=10,
            test_replicates=0,
            epochs=1,
            walkers=8,
            steps=1,
            working_directory=working_directory,
            parallelism=2,
            rng=rng,
        )

    with pytest.raises(ValueError, match="Insufficient MCMC samples"):
        dinf.alfi_mcmc_gan(
            genobuilder=genobuilder,
            iterations=2,
            training_replicates=100,
            test_replicates=0,
            epochs=1,
            walkers=6,
            steps=1,
            working_directory=working_directory,
            parallelism=2,
            rng=rng,
        )

    backup = working_directory / "bak"
    for file in [
        working_directory / f"{i}" / "discriminator.pkl",
        working_directory / f"{i}" / "surrogate.pkl",
        working_directory / f"{i}" / "mcmc.ncf",
    ]:
        file.rename(backup)
        with pytest.raises(RuntimeError, match="incomplete"):
            dinf.alfi_mcmc_gan(
                genobuilder=genobuilder,
                iterations=2,
                training_replicates=10,
                test_replicates=0,
                epochs=1,
                walkers=6,
                steps=1,
                working_directory=working_directory,
                parallelism=2,
                rng=rng,
            )
        backup.rename(file)


@pytest.mark.usefixtures("tmp_path")
def test_pg_gan(tmp_path):
    rng = np.random.default_rng(1234)
    genobuilder = get_genobuilder()
    num_params = len(genobuilder.parameters)
    num_proposals = 2
    working_directory = tmp_path / "workdir"
    dinf.pg_gan(
        genobuilder=genobuilder,
        iterations=2,
        training_replicates=10,
        test_replicates=0,
        epochs=1,
        Dx_replicates=2,
        num_proposals=num_proposals,
        max_pretraining_iterations=1,
        working_directory=working_directory,
        parallelism=2,
        rng=rng,
    )
    assert working_directory.exists()
    for i in range(2):
        check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
        check_ncf(
            working_directory / f"{i}" / "pg-gan-proposals.ncf",
            chains=num_params * num_proposals + 1,
            draws=1,
            var_names=genobuilder.parameters,
            check_acceptance_rate=False,
        )

    # resume
    os.chdir(working_directory)
    dinf.pg_gan(
        genobuilder=genobuilder,
        iterations=1,
        training_replicates=10,
        test_replicates=2,
        epochs=1,
        Dx_replicates=2,
        num_proposals=num_proposals,
        rng=rng,
    )
    for i in range(3):
        check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
        check_ncf(
            working_directory / f"{i}" / "pg-gan-proposals.ncf",
            chains=num_params * num_proposals + 1,
            draws=1,
            var_names=genobuilder.parameters,
            check_acceptance_rate=False,
        )

    backup = working_directory / "bak"
    for file in [
        working_directory / f"{i}" / "discriminator.pkl",
        working_directory / f"{i}" / "pg-gan-proposals.ncf",
    ]:
        file.rename(backup)
        with pytest.raises(RuntimeError, match="incomplete"):
            dinf.pg_gan(
                genobuilder=genobuilder,
                iterations=2,
                training_replicates=10,
                test_replicates=0,
                epochs=1,
                Dx_replicates=2,
                num_proposals=num_proposals,
                working_directory=working_directory,
                parallelism=2,
                rng=rng,
            )
        backup.rename(file)

    # TODO: check the right pretraining function is called.
    for pretraining_method in ("pg-gan", "dinf"):
        dinf.pg_gan(
            genobuilder=genobuilder,
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
            rng=rng,
        )
    with pytest.raises(ValueError, match="pretraining_method"):
        dinf.pg_gan(
            genobuilder=genobuilder,
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
            rng=rng,
        )

    # TODO: check the right proposals function is called.
    for proposals_method in ("pg-gan", "rr", "mvn"):
        dinf.pg_gan(
            genobuilder=genobuilder,
            iterations=1,
            training_replicates=10,
            test_replicates=0,
            epochs=1,
            Dx_replicates=2,
            num_proposals=num_proposals,
            proposals_method=proposals_method,
            working_directory=working_directory,
            parallelism=2,
            rng=rng,
        )
    with pytest.raises(ValueError, match="proposals_method"):
        dinf.pg_gan(
            genobuilder=genobuilder,
            iterations=1,
            training_replicates=10,
            test_replicates=0,
            epochs=1,
            Dx_replicates=2,
            num_proposals=num_proposals,
            proposals_method="not-a-valid-method",
            working_directory=working_directory,
            parallelism=2,
            rng=rng,
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


def check_proposals(proposal_thetas, theta, genobuilder, num_proposals):
    num_params = len(genobuilder.parameters)
    assert proposal_thetas.shape == (num_proposals * num_params + 1, num_params)
    np.testing.assert_array_equal(theta, proposal_thetas[0])
    for i, p in enumerate(genobuilder.parameters.values()):
        # All proposals should be within the bounds.
        assert np.all(p.bounds_contain(proposal_thetas[:, i]))


def test_sanneal_proposals_pg_gan():
    rng = np.random.default_rng(4321)
    genobuilder = get_genobuilder()
    num_proposals = 10

    for i in range(100):
        theta = genobuilder.parameters.draw_prior(num_replicates=1, rng=rng)[0]
        proposal_thetas = dinf.dinf.sanneal_proposals_pg_gan(
            theta=theta,
            temperature=1,
            num_proposals=num_proposals,
            parameters=genobuilder.parameters,
            proposal_stddev=1 / 15,
            rng=rng,
        )
        check_proposals(proposal_thetas, theta, genobuilder, num_proposals)
        for i, p in enumerate(genobuilder.parameters.values()):
            assert not elementwise_allclose(proposal_thetas[:, i])


def test_sanneal_proposals_rr():
    rng = np.random.default_rng(4321)
    genobuilder = get_genobuilder()
    num_proposals = 10
    num_params = len(genobuilder.parameters)

    iteration = itertools.count()
    for i in range(100):
        theta = genobuilder.parameters.draw_prior(num_replicates=1, rng=rng)[0]
        proposal_thetas = dinf.dinf.sanneal_proposals_rr(
            theta=theta,
            temperature=1,
            num_proposals=num_proposals,
            parameters=genobuilder.parameters,
            proposal_stddev=1 / 15,
            rng=rng,
            iteration=iteration,
        )
        check_proposals(proposal_thetas, theta, genobuilder, num_proposals)

        for j, p in enumerate(genobuilder.parameters.values()):
            # Check that all proposals are for only one parameter.
            if j == i % num_params:
                assert not elementwise_allclose(proposal_thetas[:, j]), (i, j)
            else:
                assert elementwise_allclose(proposal_thetas[:, j]), (i, j)


def test_sanneal_proposals_mvn():
    rng = np.random.default_rng(4321)
    genobuilder = get_genobuilder()
    num_proposals = 10

    for i in range(100):
        theta = genobuilder.parameters.draw_prior(num_replicates=1, rng=rng)[0]
        proposal_thetas = dinf.dinf.sanneal_proposals_mvn(
            theta=theta,
            temperature=1,
            num_proposals=num_proposals,
            parameters=genobuilder.parameters,
            proposal_stddev=1 / 15,
            rng=rng,
        )
        check_proposals(proposal_thetas, theta, genobuilder, num_proposals)
        for i, p in enumerate(genobuilder.parameters.values()):
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
        cls.genobuilder = get_genobuilder()
        rng = np.random.default_rng(111)
        cls.discriminator = dinf.Discriminator.from_input_shape(
            cls.genobuilder.feature_shape, rng
        )
        training_thetas = cls.genobuilder.parameters.draw_prior(
            num_replicates=100, rng=rng
        )
        test_thetas = cls.genobuilder.parameters.draw_prior(num_replicates=0, rng=rng)
        dinf.dinf._train_discriminator(
            discriminator=cls.discriminator,
            genobuilder=cls.genobuilder,
            training_thetas=training_thetas,
            test_thetas=test_thetas,
            epochs=1,
            parallelism=1,
            rng=rng,
        )

    def test_log_prob(self):
        # non-vector version
        log_prob_1 = functools.partial(
            log_prob,
            discriminator=self.discriminator,
            generator=self.genobuilder.generator_func,
            parameters=self.genobuilder.parameters,
            rng=np.random.default_rng(1),
            num_replicates=2,
            parallelism=1,
        )
        # vector version
        log_prob_n = functools.partial(
            dinf.dinf._log_prob,
            discriminator=self.discriminator,
            generator=self.genobuilder.generator_func,
            parameters=self.genobuilder.parameters,
            rng=np.random.default_rng(1),
            num_replicates=2,
            parallelism=1,
        )

        parameters = tuple(self.genobuilder.parameters.values())
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
        thetas = self.genobuilder.parameters.draw_prior(
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
