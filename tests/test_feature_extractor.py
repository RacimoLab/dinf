import numpy as np
import msprime
import pytest

from dinf import feature_extractor


def do_sim(
    *, num_samples, sequence_length, recombination_rate=1e-9, mutation_rate=1e-8
):
    """Return a tree sequence with infinite-sites 0/1 mutations."""
    rng = np.random.default_rng(1234)
    seed1, seed2 = rng.integers(low=1, high=2 ** 31, size=2)
    demography = msprime.Demography()
    demography.add_population(name="a", initial_size=10_000)
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(num_samples, ploidy=1)],
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed1,
    )
    ts = msprime.sim_mutations(
        ts,
        rate=mutation_rate,
        random_seed=seed2,
        model=msprime.BinaryMutationModel(),
        discrete_genome=False,
    )
    return ts


class TestBinnedHaplotypeMatrix:
    @pytest.mark.parametrize("num_samples", [8, 16])
    @pytest.mark.parametrize("sequence_length", [100_000, 1_000_000])
    @pytest.mark.parametrize("fixed_dimension", [32, 64])
    def test_feature_shape(self, num_samples, sequence_length, fixed_dimension):
        ts = do_sim(num_samples=num_samples, sequence_length=sequence_length)
        bhm = feature_extractor.BinnedHaplotypeMatrix(
            num_samples=num_samples,
            fixed_dimension=fixed_dimension,
            maf_thresh=0,
        )
        rng = np.random.default_rng(1234)
        M = bhm.from_ts(ts, rng=rng)
        assert M.shape == (num_samples, fixed_dimension)

    def test_maf_thresh(self):
        num_samples = 128
        ts = do_sim(num_samples=num_samples, sequence_length=100_000)
        thresholds = [0, 0.01, 0.05, 0.1, 1]
        rng = np.random.default_rng(1234)
        M_list = []
        for maf_thresh in thresholds:
            bhm = feature_extractor.BinnedHaplotypeMatrix(
                num_samples=num_samples,
                fixed_dimension=64,
                maf_thresh=maf_thresh,
            )
            M = bhm.from_ts(ts, rng=rng)
            M_list.append(M)

        counts = [np.sum(M) for M in M_list]
        assert counts[0] > 0
        assert counts[-1] == 0
        # We should get fewer and fewer counts for increasing maf_thresh.
        assert all(np.diff(counts) <= 0)

    def test_bad_maf_thresh(self):
        for maf_thresh in [-5, 10, np.inf]:
            with pytest.raises(ValueError):
                feature_extractor.BinnedHaplotypeMatrix(
                    num_samples=128,
                    fixed_dimension=128,
                    maf_thresh=maf_thresh,
                )

    def test_bad_num_samples(self):
        for num_samples in [-5, 10.2]:
            with pytest.raises(ValueError):
                feature_extractor.BinnedHaplotypeMatrix(
                    num_samples=num_samples,
                    fixed_dimension=128,
                    maf_thresh=0,
                )

    def test_bad_fixed_dimension(self):
        for fixed_dimension in [-5, 10.2]:
            with pytest.raises(ValueError):
                feature_extractor.BinnedHaplotypeMatrix(
                    num_samples=128,
                    fixed_dimension=fixed_dimension,
                    maf_thresh=0,
                )

    def test_bad_dtype(self):
        for dtype in [float, np.char]:
            with pytest.raises(ValueError):
                feature_extractor.BinnedHaplotypeMatrix(
                    num_samples=128,
                    fixed_dimension=128,
                    maf_thresh=0,
                    dtype=dtype
                )
