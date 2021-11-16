import functools
import pathlib
import tempfile

import numpy as np

import dinf
import tests


def test_mcmc_gan():
    rng = np.random.default_rng(123)
    genobuilder = tests.get_genobuilder()
    with tempfile.TemporaryDirectory() as tmpdir:
        working_directory = pathlib.Path(tmpdir) / "workdir"
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
            assert (working_directory / f"{i}" / "discriminator.pkl").exists()
            assert (working_directory / f"{i}" / "mcmc.ncf").exists()

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
            assert (working_directory / f"{i}" / "discriminator.pkl").exists()
            assert (working_directory / f"{i}" / "mcmc.ncf").exists()


class TestLogProb:
    @classmethod
    def setup_class(cls):
        cls.genobuilder = tests.get_genobuilder()
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
