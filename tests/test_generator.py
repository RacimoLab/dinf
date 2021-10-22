import numpy as np
import demes
import pytest

from dinf import generator, feature_extractor


class TestGenerator:
    num_samples = 128
    fixed_dimension = 64
    sequence_length = 1_000_000

    @classmethod
    def setup_class(cls):
        class SmallModel(generator.MsprimeHudsonSimulator, generator.Generator):
            params = [generator.Parameter("N", 1000, (10, 30000))]
            mutation_rate = 1.25e-8
            recombination_rate = 1e-9

            def demography(self, N):
                b = demes.Builder()
                b.add_deme("A", epochs=[dict(start_size=N)])
                return b.resolve()

        cls.bh_matrix = feature_extractor.BinnedHaplotypeMatrix(
            num_samples=cls.num_samples,
            fixed_dimension=cls.fixed_dimension,
            maf_thresh=0.05,
        )
        cls.generator = SmallModel(
            num_samples=cls.num_samples,
            sequence_length=cls.sequence_length,
            feature_extractor=cls.bh_matrix,
        )

    @pytest.mark.parametrize("num_replicates", [1, 32])
    def test_draw_params(self, num_replicates):
        rng = np.random.default_rng(1234)
        params = self.generator.draw_params(
            num_replicates=num_replicates, random=False, rng=rng
        )
        assert params.shape == (num_replicates, len(self.generator.params))
        for j, p in enumerate(self.generator.params):
            assert all(params[:j] == p.value)

        params = self.generator.draw_params(
            num_replicates=num_replicates, random=True, rng=rng
        )
        assert params.shape == (num_replicates, len(self.generator.params))
        for j, p in enumerate(self.generator.params):
            assert all(params[:j] >= p.bounds[0])
            assert all(params[:j] <= p.bounds[1])

    def test_sim(self):
        rng = np.random.default_rng(1234)
        seeds = rng.integers(2 ** 63, size=4)
        params = self.generator.draw_params(
            num_replicates=len(seeds), random=True, rng=rng
        )
        for seed, param_args in zip(seeds, params):
            M = self.generator.sim((seed, param_args))
            assert M.shape == self.bh_matrix.shape
