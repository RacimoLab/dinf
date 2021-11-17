import abc
from typing import Sequence

import numpy as np
import tskit


class _FeatureExtractor(abc.ABC):
    """
    Abstract base class for feature extractors.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> Sequence[int]:
        """Shape of the feature matrix."""
        pass

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
        pass

    @abc.abstractmethod
    def from_vcf(self, rng: np.random.Generator) -> np.ndarray:
        """Create a feature array from a vcf."""
        pass


class BinnedHaplotypeMatrix(_FeatureExtractor):
    """
    Feature matrix with dimension (n, m), where n is the number of haplotypes
    in the tree sequence (the "samples" dimension) and m is the number of
    bins into which the haplotypes are partitioned (the "sites" dimension).
    If the sequence length is L, then each bin spans L / m nucleotides.
    For a matrix M, the M[i][j]'th entry is the count of minor alleles in
    the j'th bin of haplotype i.
    Alleles are polarised by choosing the most frequent allele to be encoded
    as 0, and the other allele as 1. In the event that both alleles are equally
    frequent, the polarisation is chosen at random.

    .. note::

        This class only makes sense for biallelic polymorphisms.
    """

    def __init__(
        self, *, num_samples: int, num_bins: int, maf_thresh: float, dtype=np.int8
    ):
        """
        :param num_samples:
            The number of haploid samples.
        :param num_bins:
            The number of bins, m, into which the sequence is partitioned.
            If the sequence length is l, then each bin spans l/m base pairs.
        :param maf_thresh:
            Minor allele frequency (MAF) threshold. Sites with MAF lower than
            this value are ignored.
        :param dtype:
            The numpy data type of the feature matrix. To save memory, we use
            an np.int8 by default, which assumes that counts are small.
            However, this may not be true for very large values of
                num_samples * mu * Ne * sequence_length / num_bins,
            in which case np.int16 might be preferred.
        """
        if maf_thresh < 0 or maf_thresh > 1:
            raise ValueError("must have 0 <= maf_thresh <= 1")
        if num_samples < 2:
            raise ValueError("must have num_samples >= 2")
        if num_bins < 1:
            raise ValueError("must have num_bins >= 1")
        if dtype not in (np.int8, np.int16, np.int32):
            raise ValueError("dtype must be np.int8, np.int16, or np.in32")
        self._num_samples = num_samples
        self._num_bins = num_bins
        # We use a minimum threshold of 1 to exclude invariant sites.
        self._allele_count_threshold = max(1, maf_thresh * num_samples)
        self._dtype = dtype

    @property
    def shape(self) -> Sequence[int]:
        """Shape of the feature matrix."""
        return (self._num_samples, self._num_bins, 1)

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

        :param G: Genotype matrix with shape (num_sites, num_samples).
        :param positions: Vector of variant positions.
        :param sequence_length:
            The length of the sequence from which the matrix is derived.
        :param rng:
            Numpy random number generator. Used to randomly polarise alleles
            when there are multiple alleles with the same frequency.
        :return:
            The genotype matrix with shape (n, m, 1). For a matrix M,
            the M[i][j][0]'th entry is the count of minor alleles in the
            j'th bin of haplotype i.
        """
        assert len(G) == len(positions)
        bins = np.floor_divide(positions * self._num_bins, sequence_length).astype(
            np.int32
        )

        ac1 = np.sum(G, axis=1)
        ac0 = self._num_samples - ac1
        keep = np.minimum(ac0, ac1) >= self._allele_count_threshold
        # Polarise 0 and 1 in genotype matrix by major allele frequency.
        flip = np.logical_or(
            ac1 > ac0,
            # If allele counts are the same, randomly assign major allele.
            np.logical_and(ac1 == ac0, rng.random(len(positions)) > 0.5),
        )
        G ^= np.expand_dims(flip, -1)

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
        :param rng: Random number generator.
        :return:
            The genotype matrix with shape (n, m, 1). For a matrix M,
            the M[i][j][0]'th entry is the count of minor alleles in the
            j'th bin of haplotype i.
        """
        if ts.num_samples != self._num_samples:
            raise ValueError("Number of samples doesn't match feature matrix shape")
        if ts.sequence_length < self._num_bins:
            raise ValueError("Sequence length is shorter than fixed dimension")
        if ts.num_populations != 1:
            raise ValueError("Multi-population tree sequences not yet supported")

        G = ts.genotype_matrix()  # shape is (sites, samples)
        positions = np.array(ts.tables.sites.position)
        return self._from_genotype_matrix(
            G, positions=positions, sequence_length=ts.sequence_length, rng=rng
        )

    def from_vcf(self, rng: np.random.Generator) -> np.ndarray:
        # TODO
        raise NotImplementedError
