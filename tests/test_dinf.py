from __future__ import annotations
import functools
import pathlib
from typing import Iterable

import arviz as az
import numpy as np
import pytest


import dinf
import examples.bottleneck.model  # type: ignore[import]


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
    genobuilder = examples.bottleneck.model.genobuilder
    working_directory = tmp_path / "work_dir"
    dinf.abc_gan(
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
    dinf.abc_gan(
        genobuilder=genobuilder,
        iterations=1,
        training_replicates=16,
        test_replicates=0,
        epochs=1,
        proposals=20,
        posteriors=7,
        working_directory=working_directory,
        parallelism=2,
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


@pytest.mark.usefixtures("tmp_path")
def test_mcmc_gan(tmp_path):
    rng = np.random.default_rng(1234)
    genobuilder = examples.bottleneck.model.genobuilder
    working_directory = tmp_path / "workdir"
    dinf.mcmc_gan(
        genobuilder=genobuilder,
        iterations=2,
        training_replicates=16,
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
    dinf.mcmc_gan(
        genobuilder=genobuilder,
        iterations=1,
        training_replicates=16,
        test_replicates=0,
        epochs=1,
        walkers=6,
        steps=1,
        Dx_replicates=2,
        working_directory=working_directory,
        parallelism=2,
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


class TestLogProb:
    @classmethod
    def setup_class(cls):
        cls.genobuilder = examples.bottleneck.model.genobuilder
        rng = np.random.default_rng(111)
        cls.discriminator = dinf.Discriminator.from_input_shape(
            cls.genobuilder.feature_shape, rng
        )
        dinf.dinf._train_discriminator(
            discriminator=cls.discriminator,
            genobuilder=cls.genobuilder,
            training_replicates=100,
            test_replicates=0,
            epochs=1,
            parallelism=2,  # XXX: should't hardcode this
            rng=rng,
        )

    def test_log_prob(self):
        rng = np.random.default_rng(1)
        log_prob = functools.partial(
            dinf.dinf._mcmc_log_prob,
            discriminator=self.discriminator,
            generator=self.genobuilder.generator_func,
            parameters=self.genobuilder.parameters,
            rng=rng,
            num_replicates=2,
            parallelism=1,
        )

        parameters = tuple(self.genobuilder.parameters.values())
        true_params = [p.truth for p in parameters]
        D = np.exp(log_prob(true_params))
        # The model should learn *something* about the true data even with
        # negligible training.
        assert D > 0

        for theta in [
            [parameters[0].low - 1] + true_params[1:],
            [parameters[0].high + 1] + true_params[1:],
            true_params[:-1] + [parameters[-1].low - 1],
            true_params[:-1] + [parameters[-1].high + 1],
            [p.low - 1 for p in parameters],
            [p.high + 1 for p in parameters],
        ]:
            log_D = log_prob(theta)
            assert log_D < 0
            assert np.isinf(log_D)

    def test_log_prob_vector(self):
        rng = np.random.default_rng(1)
        log_prob = functools.partial(
            dinf.dinf._mcmc_log_prob_vector,
            discriminator=self.discriminator,
            generator=self.genobuilder.generator_func,
            parameters=self.genobuilder.parameters,
            rng=rng,
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

        log_D = log_prob(thetas)
        assert len(log_D) == len(thetas)
        assert np.exp(log_D[0]) > 0
        assert all(np.isinf(log_D[1:]))
