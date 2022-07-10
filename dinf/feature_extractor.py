from __future__ import annotations
import collections
from typing import Callable, Dict, Mapping, Tuple

import numpy as np
import numpy.typing as npt
import tskit

from .vcf import BagOfVcf
from .misc import ts_ploidy_of_individuals, ts_nodes_of_individuals


class _FeatureMatrix:
    """
    Common functionality of HaplotypeMatrix and BinnedHaplotypeMatrix
    """

    def __init__(
        self,
        *,
        num_individuals: int,
        num_loci: int,
        ploidy: int,
        phased: bool,
        maf_thresh: float | None = None,
    ):
        """
        :param num_individuals:
            The number of individuals to include in the feature matrix.
        :param num_loci:
            Dimensionality along the sequence length. This might be the number
            of snps, or the number of bins into which the sequence is partitioned.
        :param ploidy:
            Ploidy of the individuals.
        :param phased:
            If True, the individuals' haplotypes will each be included as
            independent rows in the feature matrix and the shape of the
            feature matrix will be ``(ploidy * num_individuals, num_loci, c)``.
            If False, the allele counts for each individual will be summed
            across their chromosome copies and the shape of the feature matrix
            will be ``(num_individuals, num_loci, c)``.
        :param maf_thresh:
            Minor allele frequency (MAF) threshold. Sites with MAF lower than
            this value are ignored. If None, only invariant sites will be excluded.
        """
        if num_individuals < 1:
            raise ValueError("must have num_individuals >= 1")
        if num_loci < 1:
            raise ValueError("must have num_loci >= 1")
        self._num_individuals = num_individuals
        self._num_loci = num_loci
        self._phased = phased
        self._ploidy = ploidy
        self._num_haplotypes = num_individuals * ploidy

        if maf_thresh is None:
            # Exclude invariant sites.
            self._allele_count_threshold = 1.0
        else:
            if maf_thresh < 0 or maf_thresh > 1:
                raise ValueError("must have 0 <= maf_thresh <= 1")
            self._allele_count_threshold = maf_thresh * self._num_haplotypes

        # The number of rows in the feature matrix is determined by
        # whether or not the data should be treated as phased.
        # If the data are not phased, we sum genotypes across the
        # chromosome copies. We call the rows "pseudo haplotypes",
        # regardless of the phasing.
        if phased:
            self._num_pseudo_haplotypes = self._num_haplotypes
        else:
            self._num_pseudo_haplotypes = num_individuals

        if self._num_pseudo_haplotypes < 2:
            raise ValueError("must have at least two pseudo-haplotypes")

    @property
    def shape(self):
        """Shape of the feature matrix."""
        raise NotImplementedError

    def _from_genotype_matrix(
        self,
        G: np.ndarray,
        *,
        positions: np.ndarray,
        sequence_length: int,
    ) -> np.ndarray:
        """
        Create a feature matrix from a regular genotype matrix.

        Missing genotypes in the input are assumed to take the value -1,
        and the first allele has genotype 0, second allele genotype 1, etc.

        :param G:
            Genotype matrix with shape (num_sites, num_individuals, ploidy).
            The genotypes may be phased or unphased.
        :param positions:
            Vector of variant positions.
        :param sequence_length:
            The length of the sequence from which the matrix is derived.
        :return:
            Array with shape ``(num_pseudo_haplotypes, num_loci, c)``.
        """
        raise NotImplementedError

    def from_ts(self, ts: tskit.TreeSequence) -> np.ndarray:
        """
        Create a feature matrix from a :ref:`tskit <tskit:sec_introduction>`
        tree sequence.

        :param ts: The tree sequence.
        :return:
            Array with shape ``(num_pseudo_haplotypes, num_loci, c)``.
        """
        if ts.num_samples != self._num_haplotypes:
            raise ValueError(
                f"Expected {self._num_haplotypes} haplotypes, "
                f"but ts.num_samples == {ts.num_samples}."
            )
        if ts.sequence_length < self._num_loci:
            raise ValueError("Sequence length is shorter than the number of loci")

        G = ts.genotype_matrix()  # shape is (num_sites, num_haplotypes)
        G = np.reshape(G, (G.shape[0], self._num_individuals, self._ploidy))
        positions = np.array(ts.tables.sites.position)
        return self._from_genotype_matrix(
            G, positions=positions, sequence_length=ts.sequence_length
        )

    def from_vcf(
        self,
        vb: BagOfVcf,
        *,
        sequence_length: int,
        max_missing_genotypes: int,
        min_seg_sites: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Create a feature matrix from a region of a VCF/BCF.

        The genomic window for a feature matrix is drawn uniformly at random
        from the contigs defined in the given :class:`BagOfVcf`, ``vb``.

        Individuals in the VCFs are sampled (without replacement) for
        inclusion in the output matrix. The size of the feature space
        can therefore be vastly increased by having more individuals
        in the VCFs than are needed for the feature dimensions.

        :param vb:
            The BagOfVcf object that describes the VCF/BCF files.
        :param sequence_length:
            Length of the genomic window to be sampled.
        :param max_missing_genotypes:
            Consider only sites with at most this many missing genotype calls.
        :param min_seg_sites:
            Sampled genotype matrix must have at least this many variable
            sites (after filtering sites for missingness).
        :param numpy.random.Generator rng:
            Numpy random number generator.
        :return:
            Array with shape ``(num_pseudo_haplotypes, num_loci, c)``.
        """
        G, positions = vb.sample_genotype_matrix(
            sequence_length=sequence_length,
            max_missing_genotypes=max_missing_genotypes,
            min_seg_sites=min_seg_sites,
            require_phased=self._phased,
            rng=rng,
        )

        G_individuals = G.shape[1]
        if G_individuals < self._num_individuals:
            raise ValueError(
                f"Expected at least {self._num_individuals} individuals in "
                f"the vcf bag, but only found {G_individuals}."
            )

        # Subsample individuals.
        idx = rng.choice(G_individuals, size=self._num_individuals, replace=False)
        G = G[:, idx, :]
        if np.any(G == -2):
            raise ValueError("Mismatched ploidy among individuals.")

        return self._from_genotype_matrix(
            G, positions=positions, sequence_length=sequence_length
        )


class HaplotypeMatrix(_FeatureMatrix):
    """
    A factory for feature matrices consisting of haplotypes and relative positions.

    The feature is an :math:`n \\times m \\times c` array, where the channel
    dimension :math:`c` is 2. The first channel is a haplotype matrix and
    the second channel is a matrix of relative SNP positions.

    The haplotype matrix is an :math:`n \\times m` matrix,
    where :math:`n` corresponds to the number of haplotypes
    (or number of individuals, for unphased data) and :math:`m` corresponds to
    the number of SNPs along the sequence.
    For phased data, each entry is a 0 or 1 corresponding to
    the major or minor allele respectively. For unphased data, each entry
    is the count of minor alleles across all chromosomes in the individual.
    Only polymorphic SNPs are considered, and for multiallelic sites,
    only the first two alleles are used.
    Alleles are polarised by choosing the most frequent allele to be encoded
    as 0 (the major allele), and the second most frequent allele as 1
    (the minor allele).

    The position matrix is an :math:`n \\times m` matrix, where the vector
    of :math:`m` inter-SNP distances are repeated :math:`n` times---once
    for each haplotype (or each individual, for unphased data). Each entry
    is the distance from the previous SNP (as a proportion of the sequence
    length). The first inter-SNP distance in the vector is always zero.

    | Chan et al. 2018, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7687905/
    | Wang et al. 2021, https://doi.org/10.1111/1755-0998.13386
    """

    def __init__(
        self,
        *,
        num_individuals: int,
        num_loci: int,
        ploidy: int,
        phased: bool,
        maf_thresh: float | None = None,
    ):
        """
        :param num_individuals:
            The number of individuals to include in the feature matrix.
        :param num_loci:
            The number of SNP sites to extract.
            The central ``num_loci`` SNPs in the sequence will be used.
            If there are fewer than ``num_loci`` SNPs, the feature matrix will
            be padded on both sides with zeros.
        :param ploidy:
            Ploidy of the individuals.
        :param maf_thresh:
            Minor allele frequency (MAF) threshold. Sites with MAF lower than
            this value are ignored. If None, only invariant sites will be excluded.
        :param phased:
            If True, the individuals' haplotypes will each be included as
            independent rows in the feature matrix and the shape of the
            feature matrix will be ``(ploidy * num_individuals, num_loci, c)``.
            If False, the allele counts for each individual will be summed
            across their chromosome copies and the shape of the feature matrix
            will be ``(num_individuals, num_loci, c)``.
        """
        super().__init__(
            num_individuals=num_individuals,
            num_loci=num_loci,
            ploidy=ploidy,
            maf_thresh=maf_thresh,
            phased=phased,
        )
        self._dtype = np.float32

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the feature matrix."""
        return (self._num_pseudo_haplotypes, self._num_loci, 2)

    def _from_genotype_matrix(
        self,
        G: np.ndarray,
        *,
        positions: np.ndarray,
        sequence_length: int,
    ) -> np.ndarray:
        """
        Create feature matrix from a regular genotype matrix.

        Missing genotypes in the input are assumed to take the value -1,
        and the first allele has genotype 0, second allele genotype 1, etc.
        We consider only allele 0 and 1. For multiallelic sites this means
        all but the first two alleles are ignored.

        :param G:
            Genotype matrix with shape (num_sites, num_individuals, ploidy).
            The genotypes must be phased or unphased.
        :param positions:
            Vector of variant positions.
        :param sequence_length:
            Unused.
        :return:
            Array with shape ``(num_haplotypes, num_loci, 2)``.
            For a matrix :math:`M`, the :math:`M[i][j][0]`'th entry is the
            genotype of haplotype :math:`i` at the :math:`j`'th site.
            The :math:`M[i][j][1]`'th entry is the number of basepairs
            between sites :math:`j` and :math:`j-1`.
        """
        assert len(G) == len(positions)
        G_sites, G_individuals, G_ploidy = G.shape
        assert G_individuals == self._num_individuals
        assert G_ploidy == self._ploidy
        G = np.reshape(G, (G_sites, G_individuals * G_ploidy))

        # Filter sites with insufficient minor allele count.
        ac0 = np.sum(G == 0, axis=1)
        ac1 = np.sum(G == 1, axis=1)
        keep = np.minimum(ac0, ac1) >= self._allele_count_threshold
        G = G[keep]
        positions = positions[keep]

        G, positions = self._get_fixed_num_snps(
            G, positions=positions, num_snps=self._num_loci
        )

        # Identify genotypes that aren't 0 or 1. These could be missing
        # (encoded as -1), or high-numbered alleles at multiallelic sites.
        missing = np.logical_or(G == -1, G >= 2)

        ac0 = np.sum(G == 0, axis=1)
        ac1 = np.sum(G == 1, axis=1)

        # Polarise 0 and 1 in genotype matrix by major allele frequency.
        flip = ac1 > ac0
        G ^= np.expand_dims(flip, -1)

        # Treat missing genotypes as the majority allele.
        G[missing] = 0

        if not self._phased and self._ploidy > 1:
            # Collapse each individual's chromosome copies by summing counts.
            G = np.reshape(G, (G.shape[0], self._num_individuals, self._ploidy))
            G = np.sum(G, axis=2)

        positions = np.around(positions)
        delta_positions = np.diff(positions, prepend=positions[0])
        delta_positions = delta_positions.astype(self._dtype) / sequence_length

        M = np.zeros(self.shape, dtype=self._dtype)
        M[..., 0] = G.T
        M[..., 1] = np.tile(delta_positions, [self._num_pseudo_haplotypes, 1])

        return M

    def _get_fixed_num_snps(self, G: np.ndarray, *, positions: np.ndarray, num_snps):
        """
        Trim or pad the genotype matrix and positions to be of fixed size.
        """
        assert len(G) == len(positions)
        G_sites, G_haplotypes = G.shape
        delta = G_sites - num_snps
        if delta >= 0:
            # trim
            left = delta // 2
            right = left + num_snps
            G = G[left:right]
            positions = positions[left:right]
        else:
            # pad
            pad_left = -delta // 2
            pad_right = num_snps - G_sites - pad_left
            G_left = np.zeros((pad_left, G_haplotypes), dtype=G.dtype)
            G_right = np.zeros((pad_right, G_haplotypes), dtype=G.dtype)
            G = np.concatenate((G_left, G, G_right))

            positions_left = np.zeros(pad_left, dtype=positions.dtype)
            right_pad_value = 0 if len(positions) == 0 else positions[-1]
            positions_right = np.full(pad_right, right_pad_value, dtype=positions.dtype)
            positions = np.concatenate((positions_left, positions, positions_right))

        return G, positions


class BinnedHaplotypeMatrix(_FeatureMatrix):
    """
    A factory for feature matrices of pseudo-haplotypes.

    The binned haplotype matrix is an :math:`n \\times m` matrix,
    where :math:`n` corresponds to the number of haplotypes
    (or polyploid genotypes, for unphased data) and :math:`m` corresponds to
    a set of equally sized bins along the sequence length. Each matrix entry
    contains the count of minor alleles in an individual's haplotype
    (or polyploid genotype) in a given bin.

    Only polymorphic SNPs are considered, and for multiallelic sites,
    only the first two alleles are used.
    Alleles are polarised by choosing the most frequent allele to be encoded
    as 0, and the second most frequent allele as 1.

    As the features are intended to be passed to a covolutional neural network,
    the output dimensions are actually :math:`n \\times m \\times 1`, where the
    final dimension is the (unused) "channels" dimension for the convolution.

    Gower et al. 2021, https://doi.org/10.7554/eLife.64669
    """

    def __init__(
        self,
        *,
        num_individuals: int,
        num_loci: int,
        ploidy: int,
        phased: bool,
        maf_thresh: float | None = None,
    ):
        """
        :param num_individuals:
            The number of individuals to include in the feature matrix.
        :param num_loci:
            The number of bins into which the sequence is partitioned.
            Each bin spans ``sequence_length / num_loci`` base pairs.
        :param ploidy:
            Ploidy of the individuals.
        :param phased:
            If True, the individuals' haplotypes will each be included as
            independent rows in the feature matrix and the shape of the
            feature matrix will be ``(ploidy * num_individuals, num_loci, 1)``.
            If False, the allele counts for each individual will be summed
            across their chromosome copies and the shape of the feature matrix
            will be ``(num_individuals, num_loci, 1)``.
        :param maf_thresh:
            Minor allele frequency (MAF) threshold. Sites with MAF lower than
            this value are ignored. If None, only invariant sites will be excluded.
        """
        super().__init__(
            num_individuals=num_individuals,
            num_loci=num_loci,
            ploidy=ploidy,
            maf_thresh=maf_thresh,
            phased=phased,
        )

        # The numpy data type of the feature matrix. To save memory we use
        # an np.int8, which assumes that counts are small.
        # However, counts may overflow int8 for very large values of
        #   mu * Ne * num_individuals * sequence_length / num_loci,
        # Probably num_loci could be increased in such cases.
        # TODO: print a warning if overflow occurs during conversion.
        self._dtype = np.int8

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the feature matrix."""
        return (self._num_pseudo_haplotypes, self._num_loci, 1)

    def _from_genotype_matrix(
        self,
        G: np.ndarray,
        *,
        positions: np.ndarray,
        sequence_length: int,
    ) -> np.ndarray:
        """
        Create a pseudo-genotype matrix from a regular genotype matrix.

        Missing genotypes in the input are assumed to take the value -1,
        and the first allele has genotype 0, second allele genotype 1, etc.
        We consider only allele 0 and 1. For multiallelic sites this means
        all but the first two alleles are ignored.

        :param G:
            Genotype matrix with shape (num_sites, num_individuals, ploidy).
            The genotypes may be phased or unphased.
        :param positions:
            Vector of variant positions.
        :param sequence_length:
            The length of the sequence from which the matrix is derived.
        :return:
            Array with shape ``(num_pseudo_haplotypes, num_loci, 1)``.
            For a matrix :math:`M`, the :math:`M[i][j][0]`'th entry is the
            count of minor alleles in the :math:`j`'th bin of psdeudo-haplotype
            :math:`i`.
        """
        assert len(G) == len(positions)
        M = np.zeros(self.shape, dtype=self._dtype)
        if len(positions) == 0:
            return M
        bins = np.floor_divide(positions * self._num_loci, sequence_length).astype(
            np.int32
        )

        G_sites, G_individuals, G_ploidy = G.shape
        assert G_individuals == self._num_individuals
        assert G_ploidy == self._ploidy
        G = np.reshape(G, (G_sites, G_individuals * G_ploidy))

        # Identify genotypes that aren't 0 or 1. These will be ignored later.
        missing = np.logical_or(G == -1, G >= 2)

        ac0 = np.sum(G == 0, axis=1)
        ac1 = np.sum(G == 1, axis=1)
        keep = np.minimum(ac0, ac1) >= self._allele_count_threshold
        # Polarise 0 and 1 in genotype matrix by major allele frequency.
        flip = ac1 > ac0
        G ^= np.expand_dims(flip, -1)

        # Exclude missing genotypes from the sums below.
        G[missing] = 0

        if not self._phased:
            # Collapse each individual's chromosome copies by summing counts.
            G = np.reshape(G, (G_sites, self._num_individuals, self._ploidy))
            G = np.sum(G, axis=2)

        for j, genotypes in zip(bins[keep], G[keep]):
            M[:, j, 0] += genotypes

        return M


class _MultipleFeatureMatrices:
    """
    A factory for labelled collections of :class:`_FeatureMatrix` objects.

    One feature matrix is produced for each label. Labels correspond to
    collections of individuals that will be treated as exchangeable
    (e.g. populations).
    """

    _feature_matrix_cls: Callable

    def __init__(
        self,
        *,
        num_individuals: Mapping[str, int],
        num_loci: Mapping[str, int],
        ploidy: Mapping[str, int],
        # TODO: label-specific option for maf_thresh
        # maf_thresh: Mapping[str, float] | None = None,
        global_maf_thresh: float | None = None,
        # TODO: label-specific option for phased.
        # phased: Mapping[str, bool] | None = None,
        global_phased: bool,
    ):
        """
        :param num_individuals:
            A dict that maps labels to the number of individuals
            in the feature matrix.
        :param num_loci:
            A dict that maps labels to the number of feature loci to be
            extracted from the sequence.
        :param ploidy:
            A dict that maps labels to the ploidy of the individuals.
        :param global_maf_thresh:
            Minor allele frequency (MAF) threshold. Sites with MAF lower than
            this value are ignored. MAF is calculated across all individuals.
            If None, only invariant sites will be excluded.
        :param global_phased:
            If True, the individuals' haplotypes will each be included as
            independent rows in each feature matrix and the shape of the
            feature matrix for label ``l`` will be
            ``(ploidy[l] * num_individuals[l], num_loci[l], c)``.
            If False, the allele counts for each individual will be summed
            across their chromosome copies and the shape of the feature matrix
            will be ``(num_individuals[l], num_loci[l], c)``.
        """
        dict_args = [num_individuals, num_loci, ploidy]
        dict_strs = "num_individuals, num_loci, ploidy"
        for d in dict_args:
            if not isinstance(d, collections.abc.Mapping):
                raise TypeError(f"Expected dict for each of: {dict_strs}.")
        keys_list = [d.keys() for d in dict_args]
        labels = keys_list[0]
        if any(labels != other for other in keys_list[1:]):
            raise ValueError(f"Must use the same dict keys for each of: {dict_strs}.")

        self.features = {
            label: self._feature_matrix_cls(
                num_individuals=num_individuals[label],
                num_loci=num_loci[label],
                ploidy=ploidy[label],
                maf_thresh=0,
                phased=global_phased,
            )
            for label in labels
        }
        self._num_individuals = num_individuals
        self._num_loci = num_loci
        self._ploidy = ploidy
        self._global_phased = global_phased

        # We use a minimum threshold of 1 to exclude invariant sites.
        if global_maf_thresh is None:
            global_maf_thresh = 0
        total_haplotypes = sum(
            num_individuals[label] * ploidy[label] for label in labels
        )
        self._global_allele_count_threshold = max(
            1, global_maf_thresh * total_haplotypes
        )

    @property
    def shape(self) -> Dict[str, Tuple[int, ...]]:
        """Shape of the feature matrices."""
        return {label: bhm.shape for label, bhm in self.features.items()}

    def from_ts(
        self,
        ts: tskit.TreeSequence,
        *,
        individuals: Mapping[str, npt.NDArray[np.integer]],
    ) -> Dict[str, np.ndarray]:
        """
        Create pseudo-genotype matrices from a tree sequence.

        :param ts: The tree sequence.
        :param numpy.random.Generator rng: Numpy random number generator.
        :param individuals:
            A mapping from label to an array of individuals.
        :return:
            A dictionary mapping a label ``l`` to a feature array.
            Each array has shape ``(num_pseudo_haplotypes[l], num_loci[l], c)``.
            For an array :math:`M`, the :math:`M[i][j][0]`'th entry is the
            count of minor alleles in the :math:`j`'th bin of psdeudo-haplotype
            :math:`i`.
        """

        if individuals.keys() != self.features.keys():
            raise ValueError(
                f"Labels of individuals {list(individuals)} don't match "
                f"feature labels {list(self.features)}."
            )
        G = ts.genotype_matrix()  # shape is (num_sites, num_haplotypes)
        positions = np.array(ts.tables.sites.position)

        labelled_nodes = {}
        ac0 = np.zeros(len(positions))
        ac1 = np.zeros(len(positions))
        for label, l_individuals in individuals.items():
            if ts.sequence_length < self._num_loci[label]:
                raise ValueError(
                    f"{label}: sequence length ({ts.sequence_length}) is "
                    f"shorter than the number of loci ({self._num_loci[label]})."
                )
            if len(l_individuals) != self._num_individuals[label]:
                raise ValueError(
                    f"{label}: expected {self._num_individuals[label]} "
                    f"individuals, but got {len(l_individuals)}."
                )
            ploidy = ts_ploidy_of_individuals(ts, l_individuals)
            if not np.all(ploidy == self._ploidy[label]):
                raise ValueError(
                    f"{label}: not all individuals have ploidy == {self._ploidy[label]}"
                    f"\n{label} ploidies: {ploidy}."
                )
            nodes = ts_nodes_of_individuals(ts, l_individuals)
            labelled_nodes[label] = nodes
            H = G[:, labelled_nodes[label]]
            ac0 += np.sum(H == 0, axis=1)
            ac1 += np.sum(H == 1, axis=1)

        # Filter sites with insufficient minor allele count.
        keep = np.minimum(ac0, ac1) >= self._global_allele_count_threshold
        G = G[keep]
        positions = positions[keep]

        labelled_features = {}
        for label, l_individuals in individuals.items():
            H = G[:, labelled_nodes[label]]
            H = np.reshape(
                H, (H.shape[0], self._num_individuals[label], self._ploidy[label])
            )
            labelled_features[label] = self.features[label]._from_genotype_matrix(
                H, positions=positions, sequence_length=ts.sequence_length
            )

        return labelled_features

    def from_vcf(
        self,
        vb: BagOfVcf,
        *,
        sequence_length: int,
        max_missing_genotypes: int,
        min_seg_sites: int,
        rng: np.random.Generator,
    ) -> Dict[str, np.ndarray]:
        """
        Create pseudo-genotype matrices from a region of a VCF/BCF.

        The genomic window is drawn uniformly at random from the sequences
        defined in the given :class:`BagOfVcf`.

        Individuals in the VCFs are sampled (without replacement) for
        inclusion in the output matrix. The size of the feature space
        can therefore be vastly increased by having more individuals
        in the VCFs than are needed for the feature dimensions.

        :param vb:
            A collection of indexed VCF/BCF files.
        :param sequence_length:
            Length of the genomic window to be sampled.
        :param max_missing_genotypes:
            Consider only sites with fewer missing genotype calls than
            this number.
        :param min_seg_sites:
            Sampled genotype matrix must have at least this many variable
            sites (after filtering sites for missingness).
        :param numpy.random.Generator rng:
            Numpy random number generator.
        :return:
            A dictionary mapping a label ``l`` to a feature array.
            Each array has shape ``(num_pseudo_haplotypes[l], num_loci[l], c)``.
            For an array :math:`M`, the :math:`M[i][j][0]`'th entry is the
            count of minor alleles in the :math:`j`'th bin of psdeudo-haplotype
            :math:`i`.
        """
        if vb.samples is None or self.features.keys() != vb.samples.keys():
            sample_labels = None if vb.samples is None else list(vb.samples)
            raise ValueError(
                f"Feature labels {list(self.features)} don't match the "
                f"vcf bag's sample labels: {sample_labels}."
            )
        G, positions = vb.sample_genotype_matrix(
            sequence_length=sequence_length,
            max_missing_genotypes=max_missing_genotypes,
            min_seg_sites=min_seg_sites,
            require_phased=self._global_phased,
            rng=rng,
        )
        num_samples = [len(v) for v in vb.samples.values()]
        offsets = np.cumsum([0] + num_samples[:-1])

        labelled_indexes = {}
        ac0 = np.zeros(len(positions))
        ac1 = np.zeros(len(positions))
        for j, label in enumerate(vb.samples.keys()):
            if num_samples[j] < self._num_individuals[label]:
                raise ValueError(
                    f"{label}: Expected at least {self._num_individuals[label]} "
                    f"individuals in the vcf bag, but only found {num_samples[j]}."
                )

            # Subsample individuals.
            idx = offsets[j] + rng.choice(
                num_samples[j], size=self._num_individuals[label], replace=False
            )
            labelled_indexes[label] = idx
            H = G[:, idx, : self._ploidy[label]]
            ac0 += np.sum(H == 0, axis=(1, 2))
            ac1 += np.sum(H == 1, axis=(1, 2))

        # Filter sites with insufficient minor allele count.
        keep = np.minimum(ac0, ac1) >= self._global_allele_count_threshold
        G = G[keep]
        positions = positions[keep]

        labelled_features = {}
        for label, feature_matrix in self.features.items():
            idx = labelled_indexes[label]
            H = G[:, idx, : self._ploidy[label]]
            ploidy_pad = G[:, idx, self._ploidy[label] :]
            if np.any(H == -2) or np.any(ploidy_pad != -2):
                raise ValueError(f"{label}: mismatched ploidy among individuals.")

            labelled_features[label] = feature_matrix._from_genotype_matrix(
                H, positions=positions, sequence_length=sequence_length
            )
        return labelled_features


class MultipleHaplotypeMatrices(_MultipleFeatureMatrices):
    """
    A factory for labelled collections of :class:`HaplotypeMatrix` objects.

    One feature matrix is produced for each label. Labels correspond to
    collections of individuals that will be treated as exchangeable
    (e.g. populations).
    """

    _feature_matrix_cls = HaplotypeMatrix


class MultipleBinnedHaplotypeMatrices(_MultipleFeatureMatrices):
    """
    A factory for labelled collections of :class:`BinnedHaplotypeMatrix` objects.

    One feature matrix is produced for each label. Labels correspond to
    collections of individuals that will be treated as exchangeable
    (e.g. populations).
    """

    _feature_matrix_cls = BinnedHaplotypeMatrix
