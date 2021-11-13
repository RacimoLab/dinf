import abc
from typing import Tuple

import numpy as np
import tskit


class FeatureExtractor(abc.ABC):
    """
    Abstract base class for feature extractors.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Shape of the feature matrix."""
        raise NotImplementedError

    @abc.abstractmethod
    def from_ts(
        self, ts: tskit.TreeSequence, *, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Create a feature matrix from a tskit tree sequence.

        :param ts: The tree sequence.
        :param rng: Random number generator.
        :return:
            The genotype matrix with shape (n, m). For a matrix M,
            the M[i][j]'th entry is the count of minor alleles in the j'th bin
            of haplotype i.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_vcf(self, rng: np.random.Generator) -> np.ndarray:
        """Create a feature matrix from a vcf."""
        raise NotImplementedError


class BinnedHaplotypeMatrix(FeatureExtractor):
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
        self,
        *,
        num_samples: int,
        fixed_dimension: int,
        maf_thresh: float,
        dtype=np.int8
    ):
        """
        :param num_samples:
            The number of haploid samples.
        :param fixed_dimension:
            The number of bins, m, into which the sequence is partitioned.
            If the sequence length is l, then each bin spans l/m base pairs.
        :param maf_thresh:
            Minor allele frequency (MAF) threshold. Sites with MAF lower than
            this value are ignored.
        :param dtype:
            The numpy data type of the feature matrix. To save memory, we use
            an np.int8 by default, which assumes that counts are small.
            However, this may not be true for very large values of
                num_samples * mu * Ne * sequence_length / fixed_dimension,
            in which case np.int16 might be preferred.
        """
        if maf_thresh < 0 or maf_thresh > 1:
            raise ValueError("must have 0 <= maf_thresh <= 1")
        if num_samples < 2:
            raise ValueError("must have num_samples >= 2")
        if fixed_dimension < 1:
            raise ValueError("must have fixed_dimension >= 1")
        if dtype not in (np.int8, np.int16, np.int32):
            raise ValueError("dtype must be np.int8, np.int16, or np.in32")
        self._shape = (num_samples, fixed_dimension, 1)
        self.allele_count_threshold = maf_thresh * num_samples
        self.dtype = dtype

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the feature matrix."""
        return self._shape

    def from_ts(self, ts: tskit.TreeSequence, *, rng: np.random.Generator):
        """
        Create a pseudo-genotype matrix from a tree sequence.

        :param ts: The tree sequence.
        :param rng: Random number generator.
        :return:
            The genotype matrix with shape (n, m). For a matrix M,
            the M[i][j]'th entry is the count of minor alleles in the j'th bin
            of haplotype i.
        """
        if ts.num_samples != self.shape[0]:
            raise ValueError("Number of samples doesn't match feature matrix shape")
        if ts.sequence_length < self.shape[1]:
            raise ValueError("Sequence length is shorter than fixed dimension")
        if ts.num_populations != 1:
            raise ValueError("Multi-population tree sequences not yet supported")

        M = np.zeros(self.shape, dtype=self.dtype)

        for variant in ts.variants():
            if len(variant.alleles) > 2:
                # TODO: figure out a strategy for multi-allelic simulations
                raise ValueError("Must use a binary mutation model")

            # Filter by MAF
            genotypes = variant.genotypes
            ac1 = np.sum(genotypes)
            ac0 = len(genotypes) - ac1
            if min(ac0, ac1) < self.allele_count_threshold:
                continue

            # Polarise 0 and 1 in genotype matrix by major allele frequency.
            # If allele counts are the same, randomly choose a major allele.
            if ac1 > ac0 or (ac1 == ac0 and rng.random() > 0.5):
                genotypes ^= 1

            j = int(variant.site.position * self.shape[1] / ts.sequence_length)
            M[:, j] += genotypes

        return M

    def from_vcf(self, rng: np.random.Generator):
        # TODO
        pass
