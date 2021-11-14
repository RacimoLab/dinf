import numpy as np
import msprime
import pytest

import dinf


def do_sim(
    *,
    num_samples,
    sequence_length,
    recombination_rate=1e-9,
    mutation_rate=1e-8,
    demography=None,
    samples=None,
):
    """Return a tree sequence with infinite-sites 0/1 mutations."""
    rng = np.random.default_rng(1234)
    seed1, seed2 = rng.integers(low=1, high=2 ** 31, size=2)
    if demography is None:
        demography = msprime.Demography()
        demography.add_population(name="a", initial_size=10_000)
    if samples is None:
        samples = [msprime.SampleSet(num_samples, ploidy=1)]
    ts = msprime.sim_ancestry(
        samples=samples,
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
    @pytest.mark.parametrize("num_bins", [32, 64])
    def test_from_ts_feature_shape(self, num_samples, sequence_length, num_bins):
        ts = do_sim(num_samples=num_samples, sequence_length=sequence_length)
        bhm = dinf.BinnedHaplotypeMatrix(
            num_samples=num_samples,
            num_bins=num_bins,
            maf_thresh=0,
        )
        assert bhm.shape == (num_samples, num_bins, 1)
        rng = np.random.default_rng(1234)
        M = bhm.from_ts(ts, rng=rng)
        assert M.shape == (num_samples, num_bins, 1)

    def test_from_ts_num_bins_extrema(self):
        # from_ts() encodes the minor allele as 1, where the minor allele is
        # the allele with frequency < 0.5. When the frequency is exactly 0.5,
        # the minor allele is chosen with a random number. This is awkward
        # for testing. So to get determinstic behaviour here, we make
        # num_samples odd which avoids a frequency of 0.5.
        num_samples = 101
        sequence_length = 100_000
        ts = do_sim(num_samples=num_samples, sequence_length=sequence_length)
        G = ts.genotype_matrix().T
        assert G.shape[0] == num_samples
        # Encode majority alleles as 0.
        invert = np.where(np.sum(G, axis=0) > num_samples // 2)
        G[:, invert] ^= 1

        # 1 bin per haplotype
        bhm = dinf.BinnedHaplotypeMatrix(
            num_samples=num_samples,
            num_bins=1,
            maf_thresh=0,
        )
        rng = np.random.default_rng(1234)
        M = bhm.from_ts(ts, rng=rng)
        assert M.shape == (num_samples, 1, 1)
        np.testing.assert_array_equal(M[..., 0], np.sum(G, axis=1, keepdims=True))

        # Feature matrix is the genotype matrix, including invariant sites.
        bhm = dinf.BinnedHaplotypeMatrix(
            num_samples=num_samples,
            num_bins=sequence_length,
            maf_thresh=0,
        )
        rng = np.random.default_rng(1234)
        M = bhm.from_ts(ts, rng=rng)
        has_variant = np.where(np.sum(M, axis=0) > 0)[0]
        assert len(has_variant) == ts.num_sites
        np.testing.assert_array_equal(M[:, has_variant, 0], G)

    def test_from_ts_maf_thresh(self):
        num_samples = 128
        ts = do_sim(num_samples=num_samples, sequence_length=100_000)
        thresholds = [0, 0.01, 0.05, 0.1, 1]
        rng = np.random.default_rng(1234)
        M_list = []
        for maf_thresh in thresholds:
            bhm = dinf.BinnedHaplotypeMatrix(
                num_samples=num_samples,
                num_bins=64,
                maf_thresh=maf_thresh,
            )
            M = bhm.from_ts(ts, rng=rng)
            M_list.append(M)

        counts = [np.sum(M) for M in M_list]
        assert counts[0] > 0
        assert counts[-1] == 0
        # We should get fewer and fewer counts for increasing maf_thresh.
        assert all(np.diff(counts) <= 0)

    def test_from_ts_mismatched_ts(self):
        bhm = dinf.BinnedHaplotypeMatrix(
            num_samples=64,
            num_bins=1024,
            maf_thresh=0,
        )
        rng = np.random.default_rng(1234)
        ts = do_sim(num_samples=32, sequence_length=100_000)
        with pytest.raises(ValueError, match="Number of samples"):
            bhm.from_ts(ts, rng=rng)
        ts = do_sim(num_samples=64, sequence_length=100)
        with pytest.raises(ValueError, match="Sequence length"):
            bhm.from_ts(ts, rng=rng)

        # multi-population demography not supported
        demography = msprime.Demography()
        demography.add_population(name="a", initial_size=10_000)
        demography.add_population(name="b", initial_size=10_000)
        demography.add_population(name="c", initial_size=10_000)
        demography.add_population_split(time=1000, derived=["b", "c"], ancestral="a")
        ts = do_sim(
            num_samples=64,
            sequence_length=100_000,
            demography=demography,
            samples=[
                msprime.SampleSet(32, ploidy=1, population=pop) for pop in ["b", "c"]
            ],
        )
        with pytest.raises(ValueError, match="Multi-population"):
            bhm.from_ts(ts, rng=rng)

    def test_bad_maf_thresh(self):
        for maf_thresh in [-5, 10, np.inf]:
            with pytest.raises(ValueError):
                dinf.BinnedHaplotypeMatrix(
                    num_samples=128,
                    num_bins=128,
                    maf_thresh=maf_thresh,
                )

    def test_bad_num_samples(self):
        for num_samples in [-5]:
            with pytest.raises(ValueError, match="num_samples"):
                dinf.BinnedHaplotypeMatrix(
                    num_samples=num_samples,
                    num_bins=128,
                    maf_thresh=0,
                )

    def test_bad_num_bins(self):
        for num_bins in [-5]:
            with pytest.raises(ValueError, match="num_bins"):
                dinf.BinnedHaplotypeMatrix(
                    num_samples=128,
                    num_bins=num_bins,
                    maf_thresh=0,
                )

    def test_bad_dtype(self):
        for dtype in [float, np.char]:
            with pytest.raises(ValueError, match="dtype"):
                dinf.BinnedHaplotypeMatrix(
                    num_samples=128, num_bins=128, maf_thresh=0, dtype=dtype
                )
