import functools

import numpy as np

from dinf import dinf, discriminator, feature_extractor, models


class TestLogProb:
    @classmethod
    def setup_class(cls):
        num_samples = 128
        bh_matrix = feature_extractor.BinnedHaplotypeMatrix(
            num_samples=num_samples,
            fixed_dimension=128,
            maf_thresh=0.05,
        )
        cls.generator = models.Bottleneck(
            num_samples=num_samples,
            sequence_length=1_000_000,
            feature_extractor=bh_matrix,
        )

        rng = np.random.default_rng(111)
        train_x, train_y, val_x, val_y = dinf._generate_training_data(
            generator=cls.generator,
            num_replicates=100,
            validation_ratio=0.1,
            parallelism=1,
            rng=rng,
        )
        cls.discriminator = discriminator.Discriminator.from_input_shape(
            train_x.shape[1:], rng
        )
        cls.discriminator.fit(
            rng, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y
        )

    def test_log_prob(self):
        rng = np.random.default_rng(1)
        log_prob = functools.partial(
            dinf._mcmc_log_prob,
            discr=self.discriminator,
            generator=self.generator,
            rng=rng,
            num_replicates=2,
            parallelism=1,
        )

        true_params = [p.value for p in self.generator.params]
        D = np.exp(log_prob(true_params))
        # The model should learn *something* about the true data even with
        # negligible training.
        assert D > 0

        for params in [
            [self.generator.params[0].bounds[0] - 1] + true_params[1:],
            [self.generator.params[0].bounds[1] + 1] + true_params[1:],
            true_params[:-1] + [self.generator.params[-1].bounds[0] - 1],
            true_params[:-1] + [self.generator.params[-1].bounds[1] + 1],
            [p.bounds[0] - 1 for p in self.generator.params],
            [p.bounds[1] + 1 for p in self.generator.params],
        ]:
            log_D = log_prob(params)
            assert log_D < 0
            assert np.isinf(log_D)

    def test_log_prob_vector(self):
        rng = np.random.default_rng(1)
        log_prob = functools.partial(
            dinf._mcmc_log_prob_vector,
            discr=self.discriminator,
            generator=self.generator,
            rng=rng,
            num_replicates=2,
            parallelism=1,
        )

        true_params = [p.value for p in self.generator.params]
        params = np.array(
            [
                true_params,
                # out of bounds
                [self.generator.params[0].bounds[0] - 1] + true_params[1:],
                [self.generator.params[0].bounds[1] + 1] + true_params[1:],
                true_params[:-1] + [self.generator.params[-1].bounds[0] - 1],
                true_params[:-1] + [self.generator.params[-1].bounds[1] + 1],
                [p.bounds[0] - 1 for p in self.generator.params],
                [p.bounds[1] + 1 for p in self.generator.params],
            ]
        )

        log_D = log_prob(params)
        assert len(log_D) == len(params)
        assert np.exp(log_D[0]) > 0
        assert all(np.isinf(log_D[1:]))
