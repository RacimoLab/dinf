from __future__ import annotations
from typing import Dict, Iterable, Tuple
import collections
import pathlib

import numpy as np
import cyvcf2


def _load_bed(filename):
    return np.loadtxt(
        filename,
        dtype=[("chrom", object), ("start", int), ("end", int)],
        usecols=(0, 1, 2),
    )


def bag_of_vcf(files: Iterable[str | pathlib.Path]) -> Tuple[dict, dict, dict]:
    """
    A mapping from sequence name (e.g. chromosome) to cyvcf2.VCF object.

    :param files:
        An iterable of filenames. Each file must be an indexed vcf or bcf.
    """

    seqname2file: Dict[str, str | pathlib.Path] = {}
    seqname2len: Dict[str, int] = {}
    seqname2vcf: Dict[str, cyvcf2.VCF] = {}

    files = list(files)
    if len(set(files)) != len(files):
        raise ValueError("File list contains duplicates.")

    for file in files:
        vcf = cyvcf2.VCF(file)
        if "GT" not in vcf:
            raise ValueError(f"{file} doesn't contain GT field.")
        for seqname, seqlen in zip(vcf.seqnames, vcf.seqlens):
            try:
                next(vcf(seqname))
            except StopIteration:
                continue
            if seqname in seqname2file:
                first_file = seqname2file[seqname]
                raise ValueError(
                    f"Both {file} and {first_file} contain records for "
                    f"sequence '{seqname}'."
                )
            seqname2file[seqname] = file
            seqname2len[seqname] = seqlen
            seqname2vcf[seqname] = vcf

    return seqname2file, seqname2len, seqname2vcf


def get_genotype_matrix(
    vcf: cyvcf2.VCF, chrom: str, start: int, end: int
) -> np.ndarray:
    G = []
    phased = True
    ploidy = None
    for variant in vcf(f"{chrom}:{start}-{end}"):
        if ploidy is None:
            ploidy = variant.ploidy
        if variant.ploidy != ploidy:
            raise ValueError(f"ploidy mismatch at {chrom}:{variant.POS}")
        a = variant.genotype.array()
        phased = phased and all(a[:, -1])
        g = np.reshape(a[:, :-1], -1)
        G.append(g)
    return np.array(G)


class VcfSampler:
    """
    Sample genomic windows from a collection of vcf/bcf files.
    """

    def __init__(
        self,
        files: Iterable[str | pathlib.Path],
        *,
        length: int,
    ):
        """
        :param files:
            An iterable of filenames. Each file must be an indexed vcf or bcf.
        :param length: Size of the genomic region to sample.
        """
        self._seqname2file, self._seqname2len, self._seqname2vcf = bag_of_vcf(files)
        self._length = length
        for seqname, seqlen in self._seqname2len.items():
            if length > seqlen:
                raise ValueError(
                    f"{seqname} is shorter than requested length ({length})."
                )

        self._seqnames = np.array(list(self._seqname2len.keys()))
        self._seqlens = np.array(list(self._seqname2len.values()))

    def sample_regions(
        self, size: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param size: Number of genomic windows to sample.
        :param rng: The numpy random number generator.
        """
        num_sequences = len(self._seqnames)
        p = self._seqlens / np.sum(self._seqlens)
        idx = rng.choice(num_sequences, size=size, replace=True, p=p)
        upper_limit = (self._seqlens - self._length)[idx]
        start = rng.integers(low=0, high=upper_limit, size=size)
        end = start + self._length
        chrom = self._seqnames[idx]
        return chrom, start, end
