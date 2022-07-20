from typing import Callable

import numpy as np
import msprime
import pytest
import tskit

import jax

import dinf
from .test_vcf import bcftools_index, create_vcf_dataset


def do_sim(
    *,
    num_individuals=None,
    ploidy,
    sequence_length,
    recombination_rate=1e-9,
    mutation_rate=1e-8,
    demography=None,
    samples=None,
):
    """Return a tree sequence."""
    rng = np.random.default_rng(1234)
    seed1, seed2 = rng.integers(low=1, high=2**31, size=2)
    if demography is None:
        demography = msprime.Demography()
        demography.add_population(name="a", initial_size=10_000)
    if samples is None:
        assert num_individuals is not None
        samples = num_individuals
    else:
        assert num_individuals is None
    ts = msprime.sim_ancestry(
        samples=samples,
        ploidy=ploidy,
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed1,
    )
    return msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed2)


class TestHaplotypeMatrix:
    def test_get_fixed_num_snps_example(self):
        G = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=np.int8)
        positions = np.array([5, 10, 50, 500], dtype=np.int16)

        hm = dinf.HaplotypeMatrix(
            num_individuals=2,
            num_loci=2,
            ploidy=1,
            maf_thresh=0,
            phased=True,
        )
        G2, positions2 = hm._get_fixed_num_snps(G, positions=positions, num_snps=2)
        assert G2.dtype == G.dtype
        assert positions2.dtype == positions.dtype
        np.testing.assert_array_equal(G2, [[1, 1], [1, 0]])
        np.testing.assert_array_equal(positions2, [10, 50])

        hm = dinf.HaplotypeMatrix(
            num_individuals=2,
            num_loci=2,
            ploidy=1,
            maf_thresh=0,
            phased=True,
        )
        G2, positions2 = hm._get_fixed_num_snps(G, positions=positions, num_snps=8)
        assert G2.dtype == G.dtype
        assert positions2.dtype == positions.dtype
        np.testing.assert_array_equal(
            G2, [[0, 0], [0, 0], [0, 0], [1, 1], [1, 0], [0, 1], [0, 0], [0, 0]]
        )
        np.testing.assert_array_equal(positions2, [0, 0, 5, 10, 50, 500, 500, 500])

    @pytest.mark.parametrize("phased", [True, False])
    @pytest.mark.parametrize("num_sites", [7, 36, 64])
    @pytest.mark.parametrize("num_haplotypes", [20])
    def test_get_fixed_num_snps_random(self, num_sites, num_haplotypes, phased):
        rng = np.random.default_rng(1234)
        G = rng.integers(low=0, high=2, size=(num_sites, num_haplotypes))
        positions = np.sort(rng.choice(50_000, size=num_sites, replace=False))

        for num_loci in (
            num_sites // 2,
            num_sites - 1,
            num_sites,
            num_sites + 1,
            num_sites * 2,
        ):
            hm = dinf.HaplotypeMatrix(
                num_individuals=num_haplotypes,
                num_loci=num_loci,
                ploidy=1,
                maf_thresh=0,
                phased=phased,
            )
            G2, positions2 = hm._get_fixed_num_snps(
                G, positions=positions, num_snps=num_loci
            )
            assert G2.dtype == G.dtype
            assert G2.shape == (num_loci, num_haplotypes)
            assert positions2.dtype == positions.dtype
            assert positions2.shape == (num_loci,)
            assert np.any(positions2 > 0)

    @pytest.mark.parametrize("phased", [True, False])
    @pytest.mark.parametrize("num_individuals", [8, 16])
    @pytest.mark.parametrize("num_loci", [32, 64])
    @pytest.mark.parametrize("ploidy", [1, 2, 3])
    @pytest.mark.parametrize("sequence_length", [100_000, 1_000_000])
    def test_from_ts(self, num_individuals, num_loci, ploidy, sequence_length, phased):
        num_haplotypes = ploidy * num_individuals
        num_pseudo_haplotypes = num_haplotypes if phased else num_individuals
        ts = do_sim(
            num_individuals=num_individuals,
            ploidy=ploidy,
            sequence_length=sequence_length,
        )
        hm = dinf.HaplotypeMatrix(
            num_individuals=num_individuals,
            num_loci=num_loci,
            maf_thresh=0,
            ploidy=ploidy,
            phased=phased,
        )
        assert hm.shape == (num_pseudo_haplotypes, num_loci, 2)
        M = hm.from_ts(ts)
        assert M.shape == (num_pseudo_haplotypes, num_loci, 2)

        H = M[..., 0]
        if phased:
            # All genotypes should be 0 or 1.
            assert np.all(np.logical_or(H == 0, H == 1))
        else:
            # All genotypes should be less than the allele count.
            assert np.all(H[H != 0] <= ploidy)

        # Allele 1 should be the minor allele.
        maf = np.sum(H, axis=0) / num_haplotypes
        assert np.all(maf <= 0.5)

        P = M[..., 1]
        # The first inter-SNP distance should always be 0.
        assert np.all(P[:, 0] == 0)
        # Pretty unlikely to have all zeros---we can safely assume this
        # would constitute a bug.
        assert not np.all(P[0] == 0)
        # Each row in the positions matrix should be identical.
        for row in P[1:]:
            np.testing.assert_array_equal(P[0], row)

    @pytest.mark.parametrize("phased", [True, False])
    @pytest.mark.parametrize("ploidy", [1, 2, 3])
    def test_from_ts_no_seg_sites(self, ploidy, phased):
        num_loci = 32
        num_individuals = 8
        num_haplotypes = ploidy * num_individuals
        num_pseudo_haplotypes = num_haplotypes if phased else num_individuals
        ts = do_sim(
            num_individuals=num_individuals,
            ploidy=ploidy,
            sequence_length=100_000,
            mutation_rate=0,
        )
        hm = dinf.HaplotypeMatrix(
            num_individuals=num_individuals,
            num_loci=num_loci,
            maf_thresh=0,
            ploidy=ploidy,
            phased=phased,
        )
        assert hm.shape == (num_pseudo_haplotypes, num_loci, 2)
        M = hm.from_ts(ts)
        assert M.shape == (num_pseudo_haplotypes, num_loci, 2)
        assert np.all(M == 0)

    @pytest.mark.parametrize("phased", [True, False])
    @pytest.mark.parametrize("ploidy", [1, 2, 3])
    def test_from_ts_maf_thresh(self, ploidy, phased):
        num_individuals = 128
        num_haplotypes = ploidy * num_individuals
        ts = do_sim(
            num_individuals=num_individuals, ploidy=ploidy, sequence_length=100_000
        )
        thresholds = [0, 0.01, 0.05, 0.1, 1]
        H_list = []
        P_list = []
        for maf_thresh in thresholds:
            hm = dinf.HaplotypeMatrix(
                num_individuals=num_individuals,
                num_loci=64,
                maf_thresh=maf_thresh,
                ploidy=ploidy,
                phased=phased,
            )
            M = hm.from_ts(ts)

            H = M[..., 0]
            if phased:
                # All genotypes should be 0 or 1.
                assert np.all(np.logical_or(H == 0, H == 1))
            else:
                # All genotypes should be less than the allele count.
                assert np.all(H[H != 0] <= ploidy)

            # Allele 1 should be the minor allele.
            maf = np.sum(H, axis=0) / num_haplotypes
            assert np.all(maf <= 0.5)
            # Minor allele frequency should be above the threshold.
            assert np.all(maf[maf > 0] >= maf_thresh)

            P = M[..., 1]
            # The first inter-SNP distance should always be 0.
            assert np.all(P[:, 0] == 0)
            if maf_thresh < 1:
                # Pretty unlikely to have all zeros---we can safely assume this
                # would constitute a bug.
                assert not np.all(P[0] == 0)
            # Each row in the positions matrix should be identical.
            for row in P[1:]:
                np.testing.assert_array_equal(P[0], row)

            H_list.append(H)
            P_list.append(P[0])  # position array for first haplotype

        non_pad = [np.sum(np.nonzero(P)[0]) for P in P_list]
        assert non_pad[0] > 0
        assert non_pad[-1] == 0
        # We should get fewer and fewer non-pad positions for increasing maf_thresh.
        assert all(np.diff(non_pad) <= 0)

        counts = [np.sum(H) for H in H_list]
        assert counts[0] > 0
        assert counts[-1] == 0

    @pytest.mark.parametrize("ploidy", [1, 2, 3])
    def test_from_ts_mismatched_ts(self, ploidy):
        hm = dinf.HaplotypeMatrix(
            num_individuals=64,
            num_loci=1024,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        ts = do_sim(num_individuals=32, ploidy=ploidy, sequence_length=100_000)
        with pytest.raises(ValueError, match="Expected.*haplotypes"):
            hm.from_ts(ts)

    @pytest.mark.parametrize("maf_thresh", [-5, 10, np.inf])
    def test_bad_maf_thresh(self, maf_thresh):
        with pytest.raises(ValueError):
            dinf.HaplotypeMatrix(
                num_individuals=128,
                num_loci=128,
                maf_thresh=maf_thresh,
                ploidy=2,
                phased=True,
            )

    @pytest.mark.parametrize("num_individuals", [0, -5])
    def test_bad_num_individuals(self, num_individuals):
        with pytest.raises(ValueError, match="num_individuals"):
            dinf.HaplotypeMatrix(
                num_individuals=num_individuals,
                num_loci=128,
                maf_thresh=0,
                ploidy=2,
                phased=True,
            )

    @pytest.mark.parametrize("num_loci", [0, -5])
    def test_bad_num_snps(self, num_loci):
        with pytest.raises(ValueError, match="num_loci"):
            dinf.HaplotypeMatrix(
                num_individuals=128,
                num_loci=num_loci,
                maf_thresh=0,
                ploidy=2,
                phased=True,
            )

    @pytest.mark.parametrize("phased", [True, False])
    @pytest.mark.parametrize("num_individuals", [31, 64])
    @pytest.mark.parametrize("maf_thresh", [0, 0.05])
    @pytest.mark.parametrize("ploidy", [1, 3])
    @pytest.mark.usefixtures("tmp_path")
    def test_from_vcf(self, tmp_path, ploidy, maf_thresh, num_individuals, phased):
        num_loci = 8
        sequence_length = 100_000
        hm = dinf.HaplotypeMatrix(
            num_individuals=num_individuals,
            num_loci=num_loci,
            maf_thresh=maf_thresh,
            ploidy=ploidy,
            phased=phased,
        )
        ts = do_sim(
            num_individuals=num_individuals,
            ploidy=ploidy,
            sequence_length=sequence_length,
        )
        Mts = hm.from_ts(ts)
        Hts = Mts[..., 0]
        Pts = Mts[..., 1]

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            ts.write_vcf(f, contig_id="1", position_transform=lambda x: np.floor(x) + 1)
        bcftools_index(vcf_path)
        vb = dinf.BagOfVcf([f"{vcf_path}.gz"])

        _, positions = vb.sample_genotype_matrix(
            sequence_length=sequence_length,
            max_missing_genotypes=0,
            min_seg_sites=1,
            require_phased=phased,
            rng=np.random.default_rng(1234),
        )

        Mvcf = hm.from_vcf(
            vb,
            sequence_length=sequence_length,
            max_missing_genotypes=0,
            min_seg_sites=1,
            rng=np.random.default_rng(1234),
        )
        Hvcf = Mvcf[..., 0]
        Pvcf = Mvcf[..., 1]

        def row_sorted(A):
            """Sort the rows of A."""
            return np.array(sorted(A, key=tuple))

        # Sort haplotypes because from_vcf() shuffles them.
        Hts = row_sorted(Hts)
        Hvcf = row_sorted(Hvcf)

        assert Mts.shape == Mvcf.shape
        np.testing.assert_array_equal(Hts, Hvcf)
        np.testing.assert_array_equal(Pts, Pvcf)

    @pytest.mark.usefixtures("tmp_path")
    def test_from_vcf_insufficient_individuals(self, tmp_path):
        ploidy = 2
        create_vcf_dataset(tmp_path, contig_lengths=[100_000], ploidy=ploidy)
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        num_samples = len(vb["1"].samples)

        hm = dinf.HaplotypeMatrix(
            num_individuals=num_samples + 1,
            num_loci=128,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        with pytest.raises(ValueError, match="at least .* individuals in the vcf"):
            hm.from_vcf(
                vb,
                sequence_length=10_000,
                max_missing_genotypes=0,
                min_seg_sites=20,
                rng=np.random.default_rng(123),
            )

    @pytest.mark.usefixtures("tmp_path")
    def test_from_vcf_mismatched_ploidy(self, tmp_path):
        demography = msprime.Demography()
        demography.add_population(name="a", initial_size=10_000)
        sequence_length = 100_000
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=sequence_length,
            demography=demography,
            samples=[
                msprime.SampleSet(20, ploidy=1),
                msprime.SampleSet(20, ploidy=2),
            ],
        )

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            ts.write_vcf(
                f,
                contig_id="1",
                position_transform=lambda x: np.floor(x) + 1,
            )
        bcftools_index(vcf_path)
        vb = dinf.BagOfVcf([str(vcf_path) + ".gz"])

        hm = dinf.HaplotypeMatrix(
            num_individuals=40,
            num_loci=128,
            maf_thresh=0,
            ploidy=2,
            phased=True,
        )
        with pytest.raises(ValueError, match="Mismatched ploidy"):
            hm.from_vcf(
                vb,
                sequence_length=sequence_length,
                max_missing_genotypes=0,
                min_seg_sites=1,
                rng=np.random.default_rng(123),
            )


def _binned_haplotype_matrix_from_ts(
    ts: tskit.TreeSequence,
    *,
    num_samples: int,
    num_loci: int,
    maf_thresh: int,
) -> np.ndarray:
    """
    Non-vector implementation of BinnedHaplotypeMatrix.from_ts().

    Assumes the data are phased.
    """
    assert ts.num_samples == num_samples
    assert ts.sequence_length >= num_loci
    assert ts.num_populations == 1

    # We use a minimum threshold of 1 to exclude invariant sites.
    allele_count_threshold = max(1, maf_thresh * num_samples)

    M = np.zeros((num_samples, num_loci, 1), dtype=np.int32)

    for k, variant in enumerate(ts.variants()):
        # Copy genotypes as they're read only.
        genotypes = np.copy(variant.genotypes)
        ignore = np.logical_or(genotypes < 0, genotypes > 1)
        # Filter by MAF
        ac1 = np.sum(genotypes == 1)
        ac0 = np.sum(genotypes == 0)
        if min(ac0, ac1) < allele_count_threshold:
            continue

        # Polarise 0 and 1 in genotype matrix by major allele frequency.
        if ac1 > ac0:
            genotypes ^= 1

        genotypes[ignore] = 0

        j = int(variant.site.position * num_loci / ts.sequence_length)
        M[:, j, 0] += genotypes

    return M


class TestBinnedHaplotypeMatrix:
    @pytest.mark.parametrize("num_individuals", [8, 16])
    @pytest.mark.parametrize("ploidy", [1, 2, 3])
    @pytest.mark.parametrize("sequence_length", [100_000, 1_000_000])
    @pytest.mark.parametrize("num_loci", [32, 64])
    def test_from_ts_feature_shape(
        self, num_individuals, ploidy, sequence_length, num_loci
    ):
        num_haplotypes = ploidy * num_individuals
        ts = do_sim(
            num_individuals=num_individuals,
            ploidy=ploidy,
            sequence_length=sequence_length,
        )
        bhm = dinf.BinnedHaplotypeMatrix(
            num_individuals=num_individuals,
            num_loci=num_loci,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        assert bhm.shape == (num_haplotypes, num_loci, 1)
        M = bhm.from_ts(ts)
        assert M.shape == (num_haplotypes, num_loci, 1)
        # ref implementation
        M_ref = _binned_haplotype_matrix_from_ts(
            ts, num_samples=num_haplotypes, num_loci=num_loci, maf_thresh=0
        )
        np.testing.assert_array_equal(M, M_ref)

    @pytest.mark.parametrize("num_individuals", [31, 64])
    @pytest.mark.parametrize("ploidy", [1, 3])
    def test_from_ts_num_loci_extrema(self, ploidy, num_individuals):
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
            num_loci=1,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        M = bhm.from_ts(ts)
        assert M.shape == (num_haplotypes, 1, 1)
        np.testing.assert_array_equal(M[..., 0], np.sum(G, axis=1, keepdims=True))
        # ref implementation
        M_ref = _binned_haplotype_matrix_from_ts(
            ts, num_samples=num_haplotypes, num_loci=1, maf_thresh=0
        )
        np.testing.assert_array_equal(M, M_ref)

        # Feature matrix is the genotype matrix, including invariant sites.
        bhm = dinf.BinnedHaplotypeMatrix(
            num_individuals=num_individuals,
            num_loci=sequence_length,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        M = bhm.from_ts(ts)
        has_variant = np.where(np.sum(M, axis=0) > 0)[0]
        assert len(has_variant) == ts.num_sites
        np.testing.assert_array_equal(M[:, has_variant, 0], G)
        # ref implementation
        M_ref = _binned_haplotype_matrix_from_ts(
            ts, num_samples=num_haplotypes, num_loci=sequence_length, maf_thresh=0
        )
        np.testing.assert_array_equal(M, M_ref)

    @pytest.mark.usefixtures("tmp_path")
    def test_from_ts_no_seg_sites(self, tmp_path):
        ploidy = 2
        num_individuals = 128
        num_loci = 64
        sequence_length = 100_000
        ts = do_sim(
            num_individuals=num_individuals,
            ploidy=ploidy,
            sequence_length=sequence_length,
            mutation_rate=0,
        )
        bhm = dinf.BinnedHaplotypeMatrix(
            num_individuals=num_individuals,
            num_loci=num_loci,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        M = bhm.from_ts(ts)
        assert np.all(M == 0)

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
            bhm = dinf.BinnedHaplotypeMatrix(
                num_individuals=num_individuals,
                num_loci=64,
                maf_thresh=maf_thresh,
                ploidy=ploidy,
                phased=True,
            )
            M = bhm.from_ts(ts)
            M_list.append(M)
            # ref implementation
            M_ref = _binned_haplotype_matrix_from_ts(
                ts, num_samples=num_haplotypes, num_loci=64, maf_thresh=maf_thresh
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
            num_loci=1024,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        ts = do_sim(num_individuals=32, ploidy=ploidy, sequence_length=100_000)
        with pytest.raises(ValueError, match="Expected.*haplotypes"):
            bhm.from_ts(ts)
        ts = do_sim(num_individuals=64, ploidy=ploidy, sequence_length=100)
        with pytest.raises(ValueError, match="Sequence length"):
            bhm.from_ts(ts)

    @pytest.mark.parametrize("maf_thresh", [-5, 10, np.inf])
    def test_bad_maf_thresh(self, maf_thresh):
        with pytest.raises(ValueError):
            dinf.BinnedHaplotypeMatrix(
                num_individuals=128,
                num_loci=128,
                maf_thresh=maf_thresh,
                ploidy=2,
                phased=True,
            )

    @pytest.mark.parametrize("num_individuals", [0, -5])
    def test_bad_num_individuals(self, num_individuals):
        with pytest.raises(ValueError, match="num_individuals"):
            dinf.BinnedHaplotypeMatrix(
                num_individuals=num_individuals,
                num_loci=128,
                maf_thresh=0,
                ploidy=2,
                phased=True,
            )

    def test_bad_num_pseudo_haplotypes(self):
        with pytest.raises(ValueError, match="at least two pseudo-haplotypes"):
            dinf.BinnedHaplotypeMatrix(
                num_individuals=1,
                num_loci=128,
                maf_thresh=0,
                ploidy=1,
                phased=True,
            )
        with pytest.raises(ValueError, match="at least two pseudo-haplotypes"):
            dinf.BinnedHaplotypeMatrix(
                num_individuals=1,
                num_loci=128,
                maf_thresh=0,
                ploidy=2,
                phased=False,
            )

    @pytest.mark.parametrize("num_loci", [0, -5])
    def test_bad_num_loci(self, num_loci):
        with pytest.raises(ValueError, match="num_loci"):
            dinf.BinnedHaplotypeMatrix(
                num_individuals=128,
                num_loci=num_loci,
                maf_thresh=0,
                ploidy=2,
                phased=True,
            )

    @pytest.mark.parametrize("num_individuals", [31, 64])
    @pytest.mark.parametrize("phased", [True, False])
    @pytest.mark.parametrize("ploidy", [1, 3])
    @pytest.mark.usefixtures("tmp_path")
    def test_from_vcf(self, tmp_path, ploidy, phased, num_individuals):
        num_loci = 8
        sequence_length = 100_000
        bhm = dinf.BinnedHaplotypeMatrix(
            num_individuals=num_individuals,
            num_loci=num_loci,
            maf_thresh=0,
            ploidy=ploidy,
            phased=phased,
        )
        ts = do_sim(
            num_individuals=num_individuals,
            ploidy=ploidy,
            sequence_length=sequence_length,
        )
        Mts = bhm.from_ts(ts)

        Gts = ts.genotype_matrix()  # shape is (num_sites, num_haplotypes)
        ac0 = np.sum(Gts == 0, axis=1)
        ac1 = np.sum(Gts == 1, axis=1)
        # Get positions of variable sites.
        segregating = np.minimum(ac0, ac1) >= 1
        ts_positions = ts.tables.sites.position[segregating]

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            ts.write_vcf(f, contig_id="1", position_transform=lambda x: np.floor(x) + 1)
        bcftools_index(vcf_path)
        vb = dinf.BagOfVcf([f"{vcf_path}.gz"])

        _, positions = vb.sample_genotype_matrix(
            sequence_length=sequence_length,
            max_missing_genotypes=0,
            min_seg_sites=1,
            require_phased=phased,
            rng=np.random.default_rng(1234),
        )
        np.testing.assert_array_equal(ts_positions, positions)

        Mvcf = bhm.from_vcf(
            vb,
            sequence_length=sequence_length,
            max_missing_genotypes=0,
            min_seg_sites=1,
            rng=np.random.default_rng(1234),
        )

        def row_sorted(A):
            """Sort the rows of A."""
            return np.array(sorted(A, key=tuple))

        # Sort haplotypes because from_vcf() shuffles them.
        Mts = row_sorted(Mts)
        Mvcf = row_sorted(Mvcf)

        np.testing.assert_array_equal(Mts.shape, Mvcf.shape)
        np.testing.assert_array_equal(Mts, Mvcf)

    @pytest.mark.usefixtures("tmp_path")
    def test_from_vcf_insufficient_individuals(self, tmp_path):
        ploidy = 2
        create_vcf_dataset(tmp_path, contig_lengths=[100_000], ploidy=ploidy)
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        num_samples = len(vb["1"].samples)

        bhm = dinf.BinnedHaplotypeMatrix(
            num_individuals=num_samples + 1,
            num_loci=128,
            maf_thresh=0,
            ploidy=ploidy,
            phased=True,
        )
        with pytest.raises(ValueError, match="at least .* individuals in the vcf"):
            bhm.from_vcf(
                vb,
                sequence_length=10_000,
                max_missing_genotypes=0,
                min_seg_sites=20,
                rng=np.random.default_rng(123),
            )

    @pytest.mark.usefixtures("tmp_path")
    def test_from_vcf_mismatched_ploidy(self, tmp_path):
        demography = msprime.Demography()
        demography.add_population(name="a", initial_size=10_000)
        sequence_length = 100_000
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=sequence_length,
            demography=demography,
            samples=[
                msprime.SampleSet(20, ploidy=1),
                msprime.SampleSet(20, ploidy=2),
            ],
        )

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            ts.write_vcf(
                f,
                contig_id="1",
                position_transform=lambda x: np.floor(x) + 1,
            )
        bcftools_index(vcf_path)
        vb = dinf.BagOfVcf([str(vcf_path) + ".gz"])

        bhm = dinf.BinnedHaplotypeMatrix(
            num_individuals=40,
            num_loci=128,
            maf_thresh=0,
            ploidy=2,
            phased=True,
        )
        with pytest.raises(ValueError, match="Mismatched ploidy"):
            bhm.from_vcf(
                vb,
                sequence_length=sequence_length,
                max_missing_genotypes=0,
                min_seg_sites=1,
                rng=np.random.default_rng(123),
            )


class _TestMultiple:
    cls: Callable  # The multiple feature matrices class

    def setup_class(cls):
        demography = msprime.Demography()
        demography.add_population(name="a", initial_size=10_000)
        demography.add_population(name="b", initial_size=10_000)
        demography.add_population(name="c", initial_size=10_000)
        demography.add_population_split(time=1000, derived=["b", "c"], ancestral="a")
        cls.demography = demography

    @pytest.mark.parametrize("num_individuals", [10, 20])
    @pytest.mark.parametrize("num_loci", [32, 64])
    @pytest.mark.parametrize("ploidy", [1, 2, 3])
    @pytest.mark.parametrize("global_maf_thresh", [0, 0.05])
    @pytest.mark.parametrize("global_phased", [True, False])
    def test_init(
        self, global_phased, global_maf_thresh, ploidy, num_loci, num_individuals
    ):
        populations = ["b", "c"]
        self.cls(
            num_individuals={pop: num_individuals for pop in populations},
            num_loci={pop: num_loci for pop in populations},
            ploidy={pop: ploidy for pop in populations},
            global_phased=global_phased,
            global_maf_thresh=global_maf_thresh,
        )

    def test_init_not_a_dict(self):
        num_individuals = 64
        num_loci = 128
        ploidy = 2
        global_phased = True
        global_maf_thresh = 0
        populations = ["b", "c"]
        with pytest.raises(TypeError, match="Expected dict"):
            self.cls(
                num_individuals=num_individuals,
                num_loci={pop: num_loci for pop in populations},
                ploidy={pop: ploidy for pop in populations},
                global_phased=global_phased,
                global_maf_thresh=global_maf_thresh,
            )
        with pytest.raises(TypeError, match="Expected dict"):
            self.cls(
                num_individuals={pop: num_individuals for pop in populations},
                num_loci=num_loci,
                ploidy={pop: ploidy for pop in populations},
                global_phased=global_phased,
                global_maf_thresh=global_maf_thresh,
            )
        with pytest.raises(TypeError, match="Expected dict"):
            self.cls(
                num_individuals={pop: num_individuals for pop in populations},
                num_loci={pop: num_loci for pop in populations},
                ploidy=ploidy,
                global_phased=global_phased,
                global_maf_thresh=global_maf_thresh,
            )

    def test_init_inconsistent_dict_keys(self):
        num_individuals = 64
        num_loci = 128
        ploidy = 2
        global_phased = True
        global_maf_thresh = 0
        populations = ["b", "c"]
        with pytest.raises(ValueError, match="Must use the same dict keys"):
            self.cls(
                num_individuals={pop: num_individuals for pop in ["a", "c"]},
                num_loci={pop: num_loci for pop in populations},
                ploidy={pop: ploidy for pop in populations},
                global_phased=global_phased,
                global_maf_thresh=global_maf_thresh,
            )
        with pytest.raises(ValueError, match="Must use the same dict keys"):
            self.cls(
                num_individuals={pop: num_individuals for pop in populations},
                num_loci={pop: num_loci for pop in ["a", "c"]},
                ploidy={pop: ploidy for pop in populations},
                global_phased=global_phased,
                global_maf_thresh=global_maf_thresh,
            )
        with pytest.raises(ValueError, match="Must use the same dict keys"):
            self.cls(
                num_individuals={pop: num_individuals for pop in populations},
                num_loci={pop: num_loci for pop in populations},
                ploidy={pop: ploidy for pop in ["a", "c"]},
                global_phased=global_phased,
                global_maf_thresh=global_maf_thresh,
            )

    @pytest.mark.parametrize("ploidy", [1, 2, 3])
    @pytest.mark.parametrize("phased", [True, False])
    def test_from_ts(self, phased, ploidy):
        num_individuals = 32
        populations = ["b", "c"]  # sampled populations
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=100_000,
            demography=self.demography,
            samples=[
                msprime.SampleSet(num_individuals, ploidy=ploidy, population=pop)
                for pop in populations
            ],
        )
        individuals = {pop: dinf.ts_individuals(ts, pop) for pop in populations}

        feature_extractor = self.cls(
            num_individuals={pop: num_individuals for pop in populations},
            num_loci={pop: 24 for pop in populations},
            ploidy={pop: ploidy for pop in populations},
            global_phased=phased,
            global_maf_thresh=0,
        )

        features = feature_extractor.from_ts(ts, individuals=individuals)
        assert dinf.misc.tree_equal(
            feature_extractor.shape, dinf.misc.tree_shape(features)
        )

    @pytest.mark.usefixtures("tmp_path")
    def test_from_ts_no_seg_sites(self, tmp_path):
        ploidy = 2
        num_individuals = 32
        populations = ["b", "c"]  # sampled populations
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=100_000,
            demography=self.demography,
            samples=[
                msprime.SampleSet(num_individuals, ploidy=ploidy, population=pop)
                for pop in populations
            ],
            mutation_rate=0,
        )
        individuals = {pop: dinf.ts_individuals(ts, pop) for pop in populations}

        feature_extractor = self.cls(
            num_individuals={pop: num_individuals for pop in populations},
            num_loci={pop: 24 for pop in populations},
            ploidy={pop: ploidy for pop in populations},
            global_phased=True,
            global_maf_thresh=0,
        )

        features = feature_extractor.from_ts(ts, individuals=individuals)
        for M in jax.tree_leaves(features):
            assert np.all(M == 0)

    def test_from_ts_mismatched_individuals_labels(self):
        ploidy = 2
        phased = True
        num_individuals = 32
        populations = ["b", "c"]  # sampled populations
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=100_000,
            demography=self.demography,
            samples=[
                msprime.SampleSet(num_individuals, ploidy=ploidy, population=pop)
                for pop in populations
            ],
        )
        individuals = {pop: dinf.ts_individuals(ts, pop) for pop in populations}

        wrong_populations = ["a", "c"]
        feature_extractor = self.cls(
            num_individuals={pop: num_individuals for pop in wrong_populations},
            num_loci={pop: 24 for pop in wrong_populations},
            ploidy={pop: ploidy for pop in wrong_populations},
            global_phased=phased,
            global_maf_thresh=0,
        )

        with pytest.raises(
            ValueError, match="Labels of individuals.*don't match feature labels"
        ):
            feature_extractor.from_ts(ts, individuals=individuals)

    def test_from_ts_sequence_length_too_short(self):
        ploidy = 2
        phased = True
        num_individuals = 32
        populations = ["b", "c"]  # sampled populations
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=100_000,
            demography=self.demography,
            samples=[
                msprime.SampleSet(num_individuals, ploidy=ploidy, population=pop)
                for pop in populations
            ],
        )
        individuals = {pop: dinf.ts_individuals(ts, pop) for pop in populations}

        feature_extractor = self.cls(
            num_individuals={pop: num_individuals for pop in populations},
            num_loci={"b": 24, "c": 10**6},
            ploidy={pop: ploidy for pop in populations},
            global_phased=phased,
            global_maf_thresh=0,
        )

        with pytest.raises(
            ValueError, match="sequence length.*is shorter than the number of loci"
        ):
            feature_extractor.from_ts(ts, individuals=individuals)

    def test_from_ts_bad_number_of_individuals(self):
        ploidy = 2
        phased = True
        num_individuals = 32
        populations = ["b", "c"]  # sampled populations
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=100_000,
            demography=self.demography,
            samples=[
                msprime.SampleSet(num_individuals, ploidy=ploidy, population=pop)
                for pop in populations
            ],
        )
        individuals = {pop: dinf.ts_individuals(ts, pop) for pop in populations}

        feature_extractor = self.cls(
            num_individuals={"b": num_individuals, "c": num_individuals + 1},
            num_loci={pop: 24 for pop in populations},
            ploidy={pop: ploidy for pop in populations},
            global_phased=phased,
            global_maf_thresh=0,
        )

        with pytest.raises(ValueError, match="expected.*individuals, but got"):
            feature_extractor.from_ts(ts, individuals=individuals)

    def test_from_ts_mismatched_ploidy(self):
        ploidy = 2
        phased = True
        num_individuals = 32
        populations = ["b", "c"]  # sampled populations
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=100_000,
            demography=self.demography,
            samples=[
                msprime.SampleSet(num_individuals, ploidy=ploidy, population=pop)
                for pop in populations
            ],
        )
        individuals = {pop: dinf.ts_individuals(ts, pop) for pop in populations}

        feature_extractor = self.cls(
            num_individuals={pop: num_individuals for pop in populations},
            num_loci={pop: 24 for pop in populations},
            ploidy={"b": ploidy, "c": ploidy + 1},
            global_phased=phased,
            global_maf_thresh=0,
        )

        with pytest.raises(ValueError, match="not all individuals have ploidy"):
            feature_extractor.from_ts(ts, individuals=individuals)

    @pytest.mark.parametrize("global_maf_thresh", [0, 0.05])
    @pytest.mark.parametrize("global_phased", [True, False])
    @pytest.mark.usefixtures("tmp_path")
    def test_from_vcf(self, tmp_path, global_phased, global_maf_thresh):
        num_individuals = {"b": 31, "c": 9}
        ploidy = {"b": 1, "c": 3}
        assert all(
            (j * k) % 2 != 0 for j, k in zip(num_individuals.values(), ploidy.values())
        )
        populations = list(num_individuals)  # sampled populations
        num_loci = 8
        sequence_length = 100_000
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=sequence_length,
            demography=self.demography,
            samples=[
                msprime.SampleSet(num_inds, ploidy=k, population=pop)
                for (pop, num_inds), k in zip(num_individuals.items(), ploidy.values())
            ],
        )
        individuals = {pop: dinf.ts_individuals(ts, pop) for pop in populations}

        feature_extractor = self.cls(
            num_individuals=num_individuals,
            num_loci={pop: num_loci for pop in populations},
            ploidy=ploidy,
            global_phased=global_phased,
            global_maf_thresh=global_maf_thresh,
        )
        ts_features = feature_extractor.from_ts(ts, individuals=individuals)
        assert dinf.misc.tree_equal(
            feature_extractor.shape, dinf.misc.tree_shape(ts_features)
        )

        vcf_path = tmp_path / "1.vcf"
        individual_names = [f"ind{j:03d}" for j in range(sum(num_individuals.values()))]
        with open(vcf_path, "w") as f:
            ts.write_vcf(
                f,
                contig_id="1",
                position_transform=lambda x: np.floor(x) + 1,
                individual_names=individual_names,
            )
        bcftools_index(vcf_path)
        vb = dinf.BagOfVcf(
            [f"{vcf_path}.gz"],
            samples={
                "b": individual_names[: num_individuals["b"]],
                "c": individual_names[num_individuals["b"] :],
            },
        )

        vcf_features = feature_extractor.from_vcf(
            vb,
            sequence_length=sequence_length,
            max_missing_genotypes=0,
            min_seg_sites=1,
            rng=np.random.default_rng(1234),
        )
        assert dinf.misc.tree_equal(
            feature_extractor.shape, dinf.misc.tree_shape(vcf_features)
        )

        def row_sorted(A):
            """
            Sort the rows of A. Rows in each channel are sorted independently.
            """
            return np.concatenate(
                [np.array(sorted(A[..., j], key=tuple)) for j in range(A.shape[-1])],
                axis=-1,
            )

        for pop in populations:
            Mts = ts_features[pop]
            Mvcf = vcf_features[pop]

            # Sort haplotypes because from_vcf() shuffles them.
            Mts = row_sorted(Mts)
            Mvcf = row_sorted(Mvcf)

            np.testing.assert_array_equal(Mts.shape, Mvcf.shape)
            np.testing.assert_array_equal(Mts, Mvcf)

    @pytest.mark.parametrize("wrong_ploidy", [{"b": 1, "c": 1}, {"b": 3, "c": 3}])
    @pytest.mark.usefixtures("tmp_path")
    def test_from_vcf_mismatched_ploidy(self, tmp_path, wrong_ploidy):
        num_individuals = {"b": 31, "c": 9}
        ploidy = {"b": 1, "c": 3}
        populations = list(num_individuals)  # sampled populations
        num_loci = 8
        sequence_length = 100_000
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=sequence_length,
            demography=self.demography,
            samples=[
                msprime.SampleSet(num_inds, ploidy=k, population=pop)
                for (pop, num_inds), k in zip(num_individuals.items(), ploidy.values())
            ],
        )

        vcf_path = tmp_path / "1.vcf"
        individual_names = [f"ind{j:03d}" for j in range(sum(num_individuals.values()))]
        with open(vcf_path, "w") as f:
            ts.write_vcf(
                f,
                contig_id="1",
                position_transform=lambda x: np.floor(x) + 1,
                individual_names=individual_names,
            )
        bcftools_index(vcf_path)
        vb = dinf.BagOfVcf(
            [f"{vcf_path}.gz"],
            samples={
                "b": individual_names[: num_individuals["b"]],
                "c": individual_names[num_individuals["b"] :],
            },
        )

        feature_extractor = self.cls(
            num_individuals=num_individuals,
            num_loci={pop: num_loci for pop in populations},
            ploidy=wrong_ploidy,
            global_phased=True,
            global_maf_thresh=0,
        )

        with pytest.raises(ValueError, match="mismatched ploidy"):
            feature_extractor.from_vcf(
                vb,
                sequence_length=sequence_length,
                max_missing_genotypes=0,
                min_seg_sites=1,
                rng=np.random.default_rng(1234),
            )

    @pytest.mark.usefixtures("tmp_path")
    def test_from_vcf_wrong_sample_labels(self, tmp_path):
        num_individuals = {"b": 31, "c": 9}
        ploidy = {"b": 1, "c": 3}
        populations = list(num_individuals)  # sampled populations
        num_loci = 8
        sequence_length = 100_000
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=sequence_length,
            demography=self.demography,
            samples=[
                msprime.SampleSet(num_inds, ploidy=k, population=pop)
                for (pop, num_inds), k in zip(num_individuals.items(), ploidy.values())
            ],
        )

        feature_extractor = self.cls(
            num_individuals=num_individuals,
            num_loci={pop: num_loci for pop in populations},
            ploidy=ploidy,
            global_phased=True,
            global_maf_thresh=0,
        )

        vcf_path = tmp_path / "1.vcf"
        individual_names = [f"ind{j:03d}" for j in range(sum(num_individuals.values()))]
        with open(vcf_path, "w") as f:
            ts.write_vcf(
                f,
                contig_id="1",
                position_transform=lambda x: np.floor(x) + 1,
                individual_names=individual_names,
            )
        bcftools_index(vcf_path)

        # No sample labels, which is incompatible with multipop sampling.
        vb = dinf.BagOfVcf([f"{vcf_path}.gz"])
        with pytest.raises(
            ValueError,
            match="Feature labels .* don't match the vcf bag's sample labels: None",
        ):
            feature_extractor.from_vcf(
                vb,
                sequence_length=sequence_length,
                max_missing_genotypes=0,
                min_seg_sites=1,
                rng=np.random.default_rng(1234),
            )

        # Wrong labels.
        vb = dinf.BagOfVcf(
            [f"{vcf_path}.gz"],
            samples={
                "a": individual_names[: num_individuals["b"]],
                "c": individual_names[num_individuals["b"] :],
            },
        )
        with pytest.raises(
            ValueError,
            match="Feature labels .* don't match the vcf bag's sample labels",
        ):
            feature_extractor.from_vcf(
                vb,
                sequence_length=sequence_length,
                max_missing_genotypes=0,
                min_seg_sites=1,
                rng=np.random.default_rng(1234),
            )

    @pytest.mark.usefixtures("tmp_path")
    def test_from_vcf_not_enough_individuals(self, tmp_path):
        num_individuals = {"b": 31, "c": 9}
        ploidy = {"b": 1, "c": 3}
        populations = list(num_individuals)  # sampled populations
        num_loci = 8
        sequence_length = 100_000
        ts = do_sim(
            ploidy=None,  # per-sample values are provided
            sequence_length=sequence_length,
            demography=self.demography,
            samples=[
                msprime.SampleSet(num_inds, ploidy=k, population=pop)
                for (pop, num_inds), k in zip(num_individuals.items(), ploidy.values())
            ],
        )

        feature_extractor = self.cls(
            num_individuals=num_individuals,
            num_loci={pop: num_loci for pop in populations},
            ploidy=ploidy,
            global_phased=True,
            global_maf_thresh=0,
        )

        vcf_path = tmp_path / "1.vcf"
        individual_names = [f"ind{j:03d}" for j in range(sum(num_individuals.values()))]
        with open(vcf_path, "w") as f:
            ts.write_vcf(
                f,
                contig_id="1",
                position_transform=lambda x: np.floor(x) + 1,
                individual_names=individual_names,
            )
        bcftools_index(vcf_path)

        # Fewer individuals than we're trying to sample.
        vb = dinf.BagOfVcf(
            [f"{vcf_path}.gz"],
            samples={
                "b": individual_names[: num_individuals["b"] - 1],
                "c": individual_names[num_individuals["b"] :],
            },
        )
        with pytest.raises(
            ValueError,
            match="Expected at least .* individuals .* but only found",
        ):
            feature_extractor.from_vcf(
                vb,
                sequence_length=sequence_length,
                max_missing_genotypes=0,
                min_seg_sites=1,
                rng=np.random.default_rng(1234),
            )


class TestMultipleHaplotypeMatrices(_TestMultiple):
    cls = dinf.MultipleHaplotypeMatrices


class TestMultipleBinnedHaplotypeMatrices(_TestMultiple):
    cls = dinf.MultipleBinnedHaplotypeMatrices
