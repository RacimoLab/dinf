import numpy as np
import msprime
import pytest
import tskit

import dinf


def do_sim(
    *,
    num_individuals,
    ploidy,
    sequence_length,
    recombination_rate=1e-9,
    mutation_rate=1e-8,
    demography=None,
    samples=None,
):
    """Return a tree sequence."""
    rng = np.random.default_rng(1234)
    seed1, seed2 = rng.integers(low=1, high=2 ** 31, size=2)
    if demography is None:
        demography = msprime.Demography()
        demography.add_population(name="a", initial_size=10_000)
    if samples is None:
        samples = num_individuals
    ts = msprime.sim_ancestry(
        samples=samples,
        ploidy=ploidy,
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed1,
    )
    return msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed2)


def _feature_matrix_from_ts(
    ts: tskit.TreeSequence,
    *,
    num_samples: int,
    num_bins: int,
    maf_thresh: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Non-vector implementation of BinnedHaplotypeMatrix.from_ts().

    Assumes the data are phased.
    """
    assert ts.num_samples == num_samples
    assert ts.sequence_length >= num_bins
    assert ts.num_populations == 1

    # We use a minimum threshold of 1 to exclude invariant sites.
    allele_count_threshold = max(1, maf_thresh * num_samples)

    M = np.zeros((num_samples, num_bins, 1), dtype=np.int32)
    randbits = rng.random(ts.num_sites)

    for k, variant in enumerate(ts.variants()):
        genotypes = variant.genotypes
        ignore = np.logical_or(genotypes < 0, genotypes > 1)
        # Filter by MAF
        ac1 = np.sum(genotypes == 1)
        ac0 = np.sum(genotypes == 0)
        if min(ac0, ac1) < allele_count_threshold:
            continue

        # Polarise 0 and 1 in genotype matrix by major allele frequency.
        # If allele counts are the same, randomly choose a major allele.
        if ac1 > ac0 or (ac1 == ac0 and randbits[k] > 0.5):
            genotypes ^= 1

        genotypes[ignore] = 0

        j = int(variant.site.position * num_bins / ts.sequence_length)
        M[:, j, 0] += genotypes

    return M


class TestBinnedHaplotypeMatrix:
    @pytest.mark.parametrize("num_individuals", [8, 16])
    @pytest.mark.parametrize("ploidy", [1, 2, 3])
    @pytest.mark.parametrize("sequence_length", [100_000, 1_000_000])
    @pytest.mark.parametrize("num_bins", [32, 64])
    def test_from_ts_feature_shape(
        self, num_individuals, ploidy, sequence_length, num_bins
    ):
        num_haplotypes = ploidy * num_individuals
        ts = do_sim(
            num_individuals=num_individuals,
            ploidy=ploidy,
            sequence_length=sequence_length,
        )
        bhm = dinf.BinnedHaplotypeMatrix(
            num_individuals=num_individuals,
            num_bins=num_bins,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        assert bhm.shape == (num_haplotypes, num_bins, 1)
        rng = np.random.default_rng(1234)
        M = bhm.from_ts(ts, rng=rng)
        assert M.shape == (num_haplotypes, num_bins, 1)
        # ref implementation
        rng = np.random.default_rng(1234)
        M_ref = _feature_matrix_from_ts(
            ts, num_samples=num_haplotypes, num_bins=num_bins, maf_thresh=0, rng=rng
        )
        np.testing.assert_array_equal(M, M_ref)

    @pytest.mark.parametrize("ploidy", [1, 3])
    def test_from_ts_num_bins_extrema(self, ploidy):
        # from_ts() encodes the minor allele as 1, where the minor allele is
        # the allele with frequency < 0.5. When the frequency is exactly 0.5,
        # the minor allele is chosen with a random number. This is awkward
        # for testing. So to get determinstic behaviour here, we make
        # num_haplotypes odd which avoids a frequency of 0.5.
        num_individuals = 101
        num_haplotypes = ploidy * num_individuals
        sequence_length = 100_000
        ts = do_sim(
            num_individuals=num_individuals,
            ploidy=ploidy,
            sequence_length=sequence_length,
        )
        G = ts.genotype_matrix().T
        assert G.shape[0] == num_haplotypes
        ignore = G > 1  # only consider alleles 0 and 1
        # Encode majority alleles as 0.
        invert = np.where(np.sum(G == 1, axis=0) > num_haplotypes // 2)
        G[:, invert] ^= 1
        G[ignore] = 0

        # 1 bin per haplotype
        bhm = dinf.BinnedHaplotypeMatrix(
            num_individuals=num_individuals,
            num_bins=1,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        rng = np.random.default_rng(1234)
        M = bhm.from_ts(ts, rng=rng)
        assert M.shape == (num_haplotypes, 1, 1)
        np.testing.assert_array_equal(M[..., 0], np.sum(G, axis=1, keepdims=True))
        # ref implementation
        rng = np.random.default_rng(1234)
        M_ref = _feature_matrix_from_ts(
            ts, num_samples=num_haplotypes, num_bins=1, maf_thresh=0, rng=rng
        )
        np.testing.assert_array_equal(M, M_ref)

        # Feature matrix is the genotype matrix, including invariant sites.
        bhm = dinf.BinnedHaplotypeMatrix(
            num_individuals=num_individuals,
            num_bins=sequence_length,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        rng = np.random.default_rng(1234)
        M = bhm.from_ts(ts, rng=rng)
        has_variant = np.where(np.sum(M, axis=0) > 0)[0]
        assert len(has_variant) == ts.num_sites
        np.testing.assert_array_equal(M[:, has_variant, 0], G)
        # ref implementation
        rng = np.random.default_rng(1234)
        M_ref = _feature_matrix_from_ts(
            ts,
            num_samples=num_haplotypes,
            num_bins=sequence_length,
            maf_thresh=0,
            rng=rng,
        )
        np.testing.assert_array_equal(M, M_ref)

    @pytest.mark.parametrize("ploidy", [1, 2, 3])
    def test_from_ts_maf_thresh(self, ploidy):
        num_individuals = 128
        num_haplotypes = ploidy * num_individuals
        ts = do_sim(
            num_individuals=num_individuals, ploidy=ploidy, sequence_length=100_000
        )
        thresholds = [0, 0.01, 0.05, 0.1, 1]
        M_list = []
        for maf_thresh in thresholds:
            rng = np.random.default_rng(1234)
            bhm = dinf.BinnedHaplotypeMatrix(
                num_individuals=num_individuals,
                num_bins=64,
                maf_thresh=maf_thresh,
                ploidy=ploidy,
                phased=True,
            )
            M = bhm.from_ts(ts, rng=rng)
            M_list.append(M)
            # ref implementation
            rng = np.random.default_rng(1234)
            M_ref = _feature_matrix_from_ts(
                ts,
                num_samples=num_haplotypes,
                num_bins=64,
                maf_thresh=maf_thresh,
                rng=rng,
            )
            np.testing.assert_array_equal(M, M_ref)

        counts = [np.sum(M) for M in M_list]
        assert counts[0] > 0
        assert counts[-1] == 0
        # We should get fewer and fewer counts for increasing maf_thresh.
        assert all(np.diff(counts) <= 0)

    @pytest.mark.parametrize("ploidy", [1, 2, 3])
    def test_from_ts_mismatched_ts(self, ploidy):
        bhm = dinf.BinnedHaplotypeMatrix(
            num_individuals=64,
            num_bins=1024,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        rng = np.random.default_rng(1234)
        ts = do_sim(num_individuals=32, ploidy=ploidy, sequence_length=100_000)
        with pytest.raises(ValueError, match="Expected.*haplotypes"):
            bhm.from_ts(ts, rng=rng)
        ts = do_sim(num_individuals=64, ploidy=ploidy, sequence_length=100)
        with pytest.raises(ValueError, match="Sequence length"):
            bhm.from_ts(ts, rng=rng)

        # multi-population demography not supported
        demography = msprime.Demography()
        demography.add_population(name="a", initial_size=10_000)
        demography.add_population(name="b", initial_size=10_000)
        demography.add_population(name="c", initial_size=10_000)
        demography.add_population_split(time=1000, derived=["b", "c"], ancestral="a")
        ts = do_sim(
            num_individuals=32,
            ploidy=ploidy,
            sequence_length=100_000,
            demography=demography,
            samples=[
                msprime.SampleSet(32, ploidy=ploidy, population=pop)
                for pop in ["b", "c"]
            ],
        )
        with pytest.raises(ValueError, match="Multi-population"):
            bhm.from_ts(ts, rng=rng)

    def test_bad_maf_thresh(self):
        for maf_thresh in [-5, 10, np.inf]:
            with pytest.raises(ValueError):
                dinf.BinnedHaplotypeMatrix(
                    num_individuals=128,
                    num_bins=128,
                    maf_thresh=maf_thresh,
                    ploidy=2,
                    phased=True,
                )

    def test_bad_num_individuals(self):
        for num_individuals in [-5]:
            with pytest.raises(ValueError, match="num_individuals"):
                dinf.BinnedHaplotypeMatrix(
                    num_individuals=num_individuals,
                    num_bins=128,
                    maf_thresh=0,
                    ploidy=2,
                    phased=True,
                )

    def test_bad_num_bins(self):
        for num_bins in [-5]:
            with pytest.raises(ValueError, match="num_bins"):
                dinf.BinnedHaplotypeMatrix(
                    num_individuals=128,
                    num_bins=num_bins,
                    maf_thresh=0,
                    ploidy=2,
                    phased=True,
                )

    def test_bad_dtype(self):
        for dtype in [float, np.char]:
            with pytest.raises(ValueError, match="dtype"):
                dinf.BinnedHaplotypeMatrix(
                    num_individuals=128,
                    num_bins=128,
                    maf_thresh=0,
                    ploidy=2,
                    phased=True,
                    dtype=dtype,
                )
