import abc
from typing import Sequence

import numpy as np
import tskit

from .vcf import BagOfVcf


class _FeatureExtractor(abc.ABC):
    """
    Abstract base class for feature extractors.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> Sequence[int]:
        """Shape of the features."""

    @abc.abstractmethod
    def from_ts(
        self, ts: tskit.TreeSequence, *, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Create a feature array from a tskit tree sequence.

        :param ts: The tree sequence.
        :param rng: Random number generator.
        :return: An n-dimensional feature array.
        """

    @abc.abstractmethod
    def from_vcf(
        self,
        vb: BagOfVcf,
        *,
        sequence_length: int,
        max_missing_genotypes: int,
        min_seg_sites: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Create a feature array by sampling from a collection of VCFs."""


class BinnedHaplotypeMatrix(_FeatureExtractor):
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
    as 0, and the other allele as 1. In the event that both alleles are equally
    frequent, the polarisation is chosen at random.

    As the features are intended to be passed to a covolutional neural network,
    the output dimensions are actually :math:`n \\times m \\times 1`, where the
    final dimension is the (unused) "channels" dimension for the convolution.

    Gower et al. 2021, https://doi.org/10.7554/eLife.64669
    """

    def __init__(
        self,
        *,
        num_individuals: int,
        num_bins: int,
        ploidy: int,
        phased: bool,
        maf_thresh: float,
        dtype=np.int8,
    ):
        """
        :param num_individuals:
            The number of individuals to include in the feature matrix.
        :param num_bins:
            The number of bins into which the sequence is partitioned.
            Each bin spans ``sequence_length / num_bins`` base pairs.
        :param ploidy:
            Ploidy of the individuals.
        :param phased:
            If True, the individuals' haplotypes will each be included as
            independent rows in the feature matrix and the shape of the
            feature matrix will be ``(ploidy * num_individuals, num_bins, 1)``.
            If False, the allele counts for each individual will be summed
            across their chromosome copies and the shape of the feature matrix
            will be ``(num_individuals, num_bins, 1)``.
        :param maf_thresh:
            Minor allele frequency (MAF) threshold. Sites with MAF lower than
            this value are ignored.
        :param dtype:
            The numpy data type of the feature matrix. To save memory, we use
            an np.int8 by default, which assumes that counts are small.
            However, counts may overflow int8 for very large values of
            ``mu * Ne * num_individuals * sequence_length / num_bins``,
            in which case np.int16 might be preferred. Such cases have
            not been tested, but it's likely that increasing ``num_bins``
            is a better solution.
        """
        if num_individuals < 1:
            raise ValueError("must have num_individuals >= 1")
        if num_bins < 1:
            raise ValueError("must have num_bins >= 1")
        if maf_thresh < 0 or maf_thresh > 1:
            raise ValueError("must have 0 <= maf_thresh <= 1")
        if dtype not in (np.int8, np.int16, np.int32):
            raise ValueError("dtype must be np.int8, np.int16, or np.in32")
        self._num_individuals = num_individuals
        self._num_bins = num_bins
        self._phased = phased
        self._ploidy = ploidy
        self._num_haplotypes = num_individuals * ploidy
        self._dtype = dtype
        # We use a minimum threshold of 1 to exclude invariant sites.
        self._allele_count_threshold = max(1, maf_thresh * self._num_haplotypes)

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
    def shape(self) -> Sequence[int]:
        """Shape of the feature matrix."""
        return (self._num_pseudo_haplotypes, self._num_bins, 1)

    def _from_genotype_matrix(
        self,
        G: np.ndarray,
        *,
        positions: np.ndarray,
        sequence_length: int,
        rng: np.random.Generator,
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
        :param rng:
            Numpy random number generator. Used to randomly polarise alleles
            when there are multiple alleles with the same frequency.
        :return:
            Array with shape ``(num_pseudo_haplotypes, num_bins, 1)``.
            For a matrix :math:`M`, the :math:`M[i][j][0]`'th entry is the
            count of minor alleles in the :math:`j`'th bin of psdeudo-haplotype
            :math:`i`.
        """
        assert len(G) == len(positions)
        bins = np.floor_divide(positions * self._num_bins, sequence_length).astype(
            np.int32
        )

        G_sites, G_individuals, G_ploidy = G.shape
        assert G_individuals == self._num_individuals
        assert G_ploidy == self._ploidy
        G = np.reshape(G, (G_sites, -1))

        # Identify genotypes that aren't 0 or 1. These will be ignored later.
        missing = np.logical_or(G == -1, G >= 2)

        ac0 = np.sum(G == 0, axis=1)
        ac1 = np.sum(G == 1, axis=1)
        keep = np.minimum(ac0, ac1) >= self._allele_count_threshold
        # Polarise 0 and 1 in genotype matrix by major allele frequency.
        flip = np.logical_or(
            ac1 > ac0,
            # If allele counts are the same, randomly assign major allele.
            np.logical_and(ac1 == ac0, rng.random(len(positions)) > 0.5),
        )
        G ^= np.expand_dims(flip, -1)

        # Exclude missing genotypes from the sums below.
        G[missing] = 0

        if not self._phased:
            # Collapse each individual's chromosome copies by summing counts.
            G = np.reshape(G, (-1, self._num_individuals, self._ploidy))
            G = np.sum(G, axis=2)

        M = np.zeros(self.shape, dtype=self._dtype)
        for j, genotypes in zip(bins[keep], G[keep]):
            M[:, j, 0] += genotypes

        return M

    def from_ts(
        self, ts: tskit.TreeSequence, *, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Create a pseudo-genotype matrix from a tree sequence.

        :param ts: The tree sequence.
        :param rng: Numpy random number generator.
        :return:
            Array with shape ``(num_pseudo_haplotypes, num_bins, 1)``.
            For a matrix :math:`M`, the :math:`M[i][j][0]`'th entry is the
            count of minor alleles in the :math:`j`'th bin of psdeudo-haplotype
            :math:`i`.
        """
        if ts.num_samples != self._num_haplotypes:
            raise ValueError(
                f"Expected {self._num_haplotypes} haplotypes, "
                f"but ts.num_samples == {ts.num_samples}."
            )
        if ts.sequence_length < self._num_bins:
            raise ValueError("Sequence length is shorter than the number of bins")
        if ts.num_populations != 1:
            raise ValueError("Multi-population tree sequences not yet supported")

        G = ts.genotype_matrix()  # shape is (num_sites, num_haplotypes)
        G = np.reshape(G, (-1, self._num_individuals, self._ploidy))
        positions = np.array(ts.tables.sites.position)
        return self._from_genotype_matrix(
            G, positions=positions, sequence_length=ts.sequence_length, rng=rng
        )

    def from_vcf(
        self,
        vb: BagOfVcf,
        sequence_length: int,
        max_missing_genotypes: int,
        min_seg_sites: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Create a pseudo-genotype matrix from a region of a VCF/BCF.

        The genomic window is drawn uniformly at random from the sequences
        defined in the given :class:`BagOfVcf`.

        Individuals in the VCFs are sampled (without replacement) for
        inclusion in the output matrix. The size of the feature space
        can therefore be vastly increased by having more individuals
        in the VCFs than are needed for the feature dimensions.

        :param vb:
            The BagOfVcf object that describes the VCF/BCF files.
        :param sequence_length:
            Length of the genomic window to be sampled.
        :param max_missing_genotypes:
            Consider only sites with fewer missing genotype calls than
            this number.
        :param min_seg_sites:
            Sampled genotype matrix must have at least this many variable
            sites (after filtering sites for missingness).
        :param rng:
            Numpy random number generator.
        :return:
            Array with shape ``(num_pseudo_haplotypes, num_bins, 1)``.
            For a matrix :math:`M`, the :math:`M[i][j][0]`'th entry is the
            count of minor alleles in the :math:`j`'th bin of psdeudo-haplotype
            :math:`i`.
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

        return self._from_genotype_matrix(
            G, positions=positions, sequence_length=sequence_length, rng=rng
        )
