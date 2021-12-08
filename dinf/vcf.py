from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import collections
import logging
import pathlib
import warnings

import numpy as np
import cyvcf2

logger = logging.getLogger(__name__)


def get_genotype_matrix(
    vcf: cyvcf2.VCF,
    *,
    chrom: str,
    start: int,
    end: int,
    max_missing_genotypes: int,
    require_phased: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a genotype matrix for the specified genomic window.

    The input region 'chrom:start-end' uses 1-based closed coordinates,
    to match htslib (upon which cyvcf2 is built).
    The returned matrix has shape (num_sites, num_individuals, ploidy),
    where each site is a segregating SNP.

    :param vcf:
        A cyvcf2.VCF instance from which the genotypes will be obtained.
    :param chrom:
        Chromosome name (or non-chromosome sequence name).
    :param start:
        1-based start coordinate of the genomic window.
    :param end:
        1-based end coordinate of the genomic window.
    :param max_missing_genotypes:
        Only retain sites with fewer missing genotype calls than this number.
    :param require_phased:
        If True, raise an error if genotypes are not phased.
    :return:
        A 2-tuple of (G, positions) where
         - ``G`` is a (num_sites, num_individuals, ploidy) genotype matrix,
         - ``positions`` are the site coordinates, as a zero-based offset from
           the ``start`` coordinate.
    """
    G = []
    positions = []
    for variant in vcf(f"{chrom}:{start}-{end}"):
        # XXX: Many variant fields are broken for ploidy > 2.
        # https://github.com/brentp/cyvcf2/issues/227

        if not variant.is_snp:
            continue

        a = variant.genotype.array()
        gt = a[:, :-1]
        if np.sum(gt == -1) > max_missing_genotypes:
            continue
        if len(np.unique(gt[gt >= 0])) == 1:
            # Invariant site.
            continue

        # Check phasing. For ploidy == 1, genotypes are reported as unphased.
        if require_phased and gt.shape[1] > 1 and not np.all(a[:, -1]):
            raise ValueError(f"Unphased genotypes at {chrom}:{variant.POS}.\n{variant}")

        G.append(gt)
        positions.append(variant.POS)

    try:
        np_G = np.array(G, dtype=np.int8)
    except ValueError as e:
        # Most likely an "inhomogeneous shape" error, caused by different
        # ploidies at different sites. Catch and reraise to provide
        # a more user-friendly error message.
        raise ValueError(
            f"Mismatched ploidy among sites in {chrom}:{start}-{end}."
        ) from e
    if np.any(np_G == -2):
        raise ValueError(
            f"Mismatched ploidy among individuals in {chrom}:{start}-{end}."
        )

    np_positions = np.array(positions, dtype=int) - start
    return np_G, np_positions


class BagOfVcf(collections.abc.Mapping):
    """
    A collection of indexed VCF or BCF files.

    VCF data are sometimes contained in a single file, and sometimes split into
    multiple files by chromosome. To remove the burden of dealing with both
    of these common cases, this class maps a contig id to a :class:`cyvcf2.VCF`
    object for that contig. The interface is provided via Python's
    :class:`collections.abc.Mapping` protocol. In addition, the class provides
    methods for sampling regions of the genome at random.
    """

    contig_lengths: np.ndarray
    """
    Lengths of the contigs in the bag. The order matches the contig order
    obtained by iterating over the bag.
    """

    def __init__(
        self,
        files: Iterable[str | pathlib.Path],
        *,
        individuals: Iterable[str] | None = None,
        contigs: Iterable[str] | None = None,
    ):
        """
        :param files:
            An iterable of filenames. Each file must be an indexed VCF or BCF,
            and contain the FORMAT/GT field.
        :param individuals:
            An iterable of individual names corresponding to the VCF columns
            for which genotypes will be sampled.
        :param contigs:
            An iterable of contig names to use. If not specified, all contigs
            identified in the VCFs/BCFs will be used.
        """
        # Silence some warnings from cyvcf2.
        with warnings.catch_warnings():
            # We check if a contig is usable for a given vcf by querying
            # that contig for variants. If there are no variants (e.g. vcfs are
            # split by chromosome so each vcf has data for only one contig),
            # then cyvcf2 warns us.
            warnings.filterwarnings(
                "ignore", message="no intervals found", category=UserWarning
            )
            # We detect this condition and explicitly raise an error.
            warnings.filterwarnings(
                "ignore",
                message="not all requested samples found",
                category=UserWarning,
            )
            self._fill_bag(files=files, individuals=individuals, contigs=contigs)
        self._regions: List[Tuple[str, int, int]] = []

    def _fill_bag(
        self,
        *,
        files: Iterable[str | pathlib.Path],
        individuals: Iterable[str] | None = None,
        contigs: Iterable[str] | None = None,
    ) -> None:
        """
        Construct a mapping from contig id to cyvcf2.VCF object.
        """

        contig2file: Dict[str, str | pathlib.Path] = {}
        contig2vcf: Dict[str, cyvcf2.VCF] = {}
        contig_lengths = []

        files = list(files)
        if len(set(files)) != len(files):
            raise ValueError("File list contains duplicates.")

        if individuals is not None:
            individuals = list(individuals)
            if len(set(individuals)) != len(individuals):
                raise ValueError("Individuals list contains duplicates.")

        if contigs is not None:
            contigs = list(contigs)
            if len(set(contigs)) != len(contigs):
                raise ValueError("Contigs list contains duplicates.")
            contigs = set(contigs)

        contigs_seen = set()

        for file in files:
            vcf = cyvcf2.VCF(file, samples=individuals, lazy=True, threads=1)
            if "GT" not in vcf:
                raise ValueError(f"{file} doesn't contain GT field.")
            if not (
                pathlib.Path(f"{file}.tbi").exists()
                or pathlib.Path(f"{file}.csi").exists()
            ):
                raise ValueError(f"No index found for {file}.")
            if individuals is not None:
                individuals_not_found = set(individuals) - set(vcf.samples)
                if len(individuals_not_found) > 0:
                    raise ValueError(
                        f"Requested individuals not found in {file}: "
                        f"{', '.join(individuals_not_found)}"
                    )
            for contig_id, contig_length in zip(vcf.seqnames, vcf.seqlens):
                contigs_seen.add(contig_id)
                if contigs is not None and contig_id not in contigs:
                    continue
                # Check there's at least one variant. If present, we assume
                # there exist usable variants (i.e. SNPs) for the contig.
                try:
                    next(vcf(contig_id))
                except StopIteration:
                    continue
                if contig_id in contig2file:
                    first_file = contig2file[contig_id]
                    raise ValueError(
                        f"Both {file} and {first_file} contain records for "
                        f"sequence '{contig_id}'."
                    )
                contig2file[contig_id] = file
                contig_lengths.append(contig_length)
                contig2vcf[contig_id] = vcf

        if contigs is not None:
            contigs_not_found = contigs - contigs_seen
            if len(contigs_not_found) > 0:
                raise ValueError(
                    f"Requested contigs not found: {', '.join(contigs_not_found)}"
                )

        if len(contig2file) == 0:
            raise ValueError("No usable vcf/bcf files in the list.")

        self._contig2vcf = contig2vcf
        self.contig_lengths = np.array(contig_lengths)

    def __iter__(self):
        yield from self._contig2vcf

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        return self._contig2vcf[key]

    def __len__(self):
        return len(self._contig2vcf)

    def sample_regions(
        self, size: int, sequence_length: int, rng: np.random.Generator
    ) -> List[Tuple[str, int, int]]:
        """
        Sample a list of (chrom, start, end) triplets.

        :param size: Number of genomic windows to sample.
        :param sequence_length: Length of the sequence to sample.
        :param rng: The numpy random number generator.
        :return:
            List of 3-tuples: (chrom, start, end).
            The start and end coordinates are 1-based and inclusive, to match
            the usual convention for 'chrom:start-end', e.g. in bcftools.
        """

        long_enough = self.contig_lengths >= sequence_length
        if sum(long_enough) == 0:
            raise ValueError(f"No contigs with length >= {sequence_length}.")
        contig_ids = np.array(self, dtype=object)[long_enough]
        contig_lengths = self.contig_lengths[long_enough]

        p = contig_lengths / np.sum(contig_lengths)
        idx = rng.choice(len(contig_ids), size=size, replace=True, p=p)
        upper_limit = (contig_lengths - sequence_length)[idx]
        start = 1 + rng.integers(low=0, high=upper_limit, size=size, endpoint=True)
        end = start + sequence_length - 1
        chrom = contig_ids[idx]
        regions = list(zip(chrom, start, end))
        return regions

    def sample_genotype_matrix(
        self,
        *,
        sequence_length: int,
        min_seg_sites: int,
        max_missing_genotypes: int,
        require_phased: bool,
        rng: np.random.Generator,
        retries: int = 1000,
    ):
        """
        Sample a genotype matrix uniformly at random from the genome.

        :param sequence_length:
            Length of the sequence to sample.
        :param max_missing_genotypes:
            Consider only sites with fewer missing genotype calls than
            this number.
        :param min_seg_sites:
            Sampled genotype matrix must have at least this many variable
            sites (after filtering sites for missingness).
        :param require_phased:
            If True, raise an error if genotypes are not phased.
        :param rng:
            Numpy random number generator.
        :param retries:
            Maximum number of attempts allowed to find a genomic window with
            the required number of segregating sites.
        :return:
            A 2-tuple of (G, positions) where
             - ``G`` is a (num_sites, num_individuals, ploidy) genotype matrix,
             - ``positions`` are the site coordinates, as an offset from the
               the start of the genomic window.
        """
        for _ in range(retries):
            if len(self._regions) == 0:
                self._regions = self.sample_regions(1000, sequence_length, rng)
            chrom, start, end = self._regions.pop()
            G, positions = get_genotype_matrix(
                self[chrom],
                chrom=chrom,
                start=start,
                end=end,
                max_missing_genotypes=max_missing_genotypes,
                require_phased=require_phased,
            )
            if len(positions) < min_seg_sites:
                continue
            break
        else:
            raise ValueError(
                f"Failed to sample genotype matrix with at least {min_seg_sites} "
                f"segregating sites after {retries} attempts."
            )
        return G, positions
