from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import collections
import itertools
import logging
import os
import pathlib
import warnings

import numpy as np
import cyvcf2

logger = logging.getLogger(__name__)


def get_samples_from_1kgp_metadata(filename: str, /, *, populations: list) -> dict:
    """
    Get sample IDs for 1000 Genomes Project populations.
    Related individuals are removed based on the FatherID and MotherID.

    :param filename:
        Path to the file "20130606_g1k_3202_samples_ped_population.txt"
        This file can be downloaded from
        http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/
    :param populations:
        List of populations to extract.
    :return:
        A dict mapping population names to a list of sample IDs.
    """
    data = np.recfromtxt(filename, names=True, encoding="ascii")
    # Remove related individuals.
    data = data[data.FatherID == "0"]
    data = data[data.MotherID == "0"]
    return {pop: data.SampleID[data.Population == pop].tolist() for pop in populations}


def get_contig_lengths(
    filename: pathlib.Path | str, /, keep_contigs: Iterable[str] | None = None
) -> dict:
    """
    Load contig lengths from a whitespace-separated file like an fai (fasta index).

    The file must have (at least) two columns, the first specifies the
    contig ID, and the second is the contig length. Additional columns
    are ignored.

    :param filename: The path to the file.
    :param keep_contigs: Use only the specified contigs.
    """
    data = {}
    if keep_contigs is not None:
        keep_contigs = set(keep_contigs)
        seen_contigs = set()
    with open(filename) as f:
        for line in f:
            fields = line.split(maxsplit=2)
            assert len(fields) >= 2
            contig = fields[0]
            if keep_contigs is not None:
                seen_contigs.add(contig)
                if contig not in keep_contigs:
                    continue
            length = int(fields[1])
            data[contig] = length
    if keep_contigs is not None:
        contigs_not_found = keep_contigs - seen_contigs
        if len(contigs_not_found) > 0:
            raise ValueError(
                f"Requested contigs not found: {', '.join(contigs_not_found)}"
            )
    return data


def filter_missingness(G, max_missing_genotypes):
    return (G == -1).sum(axis=(1, 2)) <= max_missing_genotypes


def filter_invariant(G):
    num_non_missing = (G >= 0).sum(axis=(1, 2))
    ac0 = (G == 0).sum(axis=(1, 2))
    ac1 = (G == 1).sum(axis=(1, 2))
    ac2 = (G == 2).sum(axis=(1, 2))
    ac3 = (G == 3).sum(axis=(1, 2))
    return np.logical_and(
        np.logical_and(num_non_missing != ac0, num_non_missing != ac1),
        np.logical_and(num_non_missing != ac2, num_non_missing != ac3),
    )


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
        Only retain sites with at most this many missing genotype calls.
    :param require_phased:
        If True, raise an error if genotypes are not phased.
    :return:
        A 2-tuple of (G, positions) where
         - ``G`` is a (num_sites, num_individuals, ploidy) genotype matrix,
         - ``positions`` are the site coordinates, as a zero-based offset from
           the ``start`` coordinate.
    """
    gt_list = []
    positions_list = []
    for variant in vcf(f"{chrom}:{start}-{end}"):
        # XXX: Many variant fields are broken for ploidy > 2.
        # https://github.com/brentp/cyvcf2/issues/227

        if not variant.is_snp:
            continue

        a = variant.genotype.array()
        gt = a[:, :-1]

        # Check phasing. For ploidy == 1, genotypes are reported as unphased.
        if require_phased and gt.shape[1] > 1 and not (a[:, -1]).all():
            raise ValueError(f"Unphased genotypes at {chrom}:{variant.POS}.\n{variant}")

        gt_list.append(gt)
        positions_list.append(variant.POS)

    try:
        G = np.array(gt_list, dtype=np.int8)
    except ValueError as e:
        # Most likely an "inhomogeneous shape" error, caused by different
        # ploidies at different sites. Catch and reraise to provide
        # a more user-friendly error message.
        raise ValueError(
            f"Mismatched ploidy among sites in {chrom}:{start}-{end}."
        ) from e

    positions = np.array(positions_list, dtype=int) - start

    if len(G) > 0:
        # Filter for missingness.
        keep = filter_missingness(G, max_missing_genotypes)
        G = G[keep]
        positions = positions[keep]

    if len(G) > 0:
        # Filter invariant sites.
        keep = filter_invariant(G)
        G = G[keep]
        positions = positions[keep]

    return G, positions


class BagOfVcf(collections.abc.Mapping):
    """
    A collection of indexed VCF or BCF files.

    VCF data are sometimes contained in a single file, and sometimes split into
    multiple files by chromosome. To remove the burden of dealing with both
    of these common cases, this class maps a contig ID to a :class:`cyvcf2.VCF`
    object for that contig. The class implements Python's :term:`python:mapping`
    protocol, plus methods for sampling regions of the genome at random.

    .. code::

        import dinf

        vcfs = dinf.BagOfVcf(["chr1.bcf", "chr2.bcf", "chr3.bcf"])

        # Iterate over contig IDs.
        assert len(vcfs) == 3
        assert list(vcfs) == ["chr1", "chr2", "chr3"]

        # Lookup a cyvcf2.VCF object by contig ID.
        chr1 = vcfs["chr1"]
        assert chr1.fname == "chr1.bcf"
        first_variant = next(chr1)
        assert first_variant.CHROM == "chr1"
    """

    lengths: np.ndarray
    """
    Lengths of the contigs in the bag.

    The order matches the contig order obtained when iterating.
    """

    samples: collections.abc.Mapping[str, List[str]] | None
    """
    A dictionary that maps a label to a list of individual names.

    The individual names correspond to the VCF columns for which genotypes
    will be sampled. This is a bookkeeping device that records which genotypes
    belong to which label (e.g. which population). If None, it is assumed
    that all individuals in the VCF will be treated as exchangeable.
    """

    def __init__(
        self,
        files: Iterable[str | pathlib.Path],
        /,
        *,
        samples: collections.abc.Mapping[str, List[str]] | None = None,
        contig_lengths: collections.abc.Mapping[str, int] | None = None,
    ):
        """
        :param files:
            An iterable of filenames. Each file must be an indexed VCF or BCF,
            and must contain the FORMAT/GT field.
        :param contig_lengths:
            A dict mapping a contig name to contig length. Only the contigs in
            this dict will be used.
        :param samples:
            A dictionary that maps a label to a list of individual names,
            where the individual names correspond to the VCF columns
            for which genotypes will be sampled.
        """
        individuals = None
        if samples is not None:
            individuals = itertools.chain(*samples.values())
        # Silence some warnings from cyvcf2.
        with warnings.catch_warnings():
            # We check if a contig is usable for a given vcf by querying
            # that contig for variants. If there are no variants (e.g. vcfs are
            # split by chromosome so each vcf has data for only one contig),
            # then cyvcf2 warns us.
            warnings.filterwarnings(
                "ignore", message="no intervals found", category=UserWarning
            )
            # We detect this condition and raise our own error below.
            warnings.filterwarnings(
                "ignore",
                message="not all requested samples found",
                category=UserWarning,
            )
            self._fill_bag(
                files=files, contig_lengths=contig_lengths, individuals=individuals
            )

        self.samples = samples
        self._regions: List[Tuple[str, int, int]] = []

    def _fill_bag(
        self,
        *,
        files: Iterable[str | pathlib.Path],
        contig_lengths: collections.abc.Mapping[str, int] | None = None,
        individuals: Iterable[str] | None = None,
    ) -> None:
        """
        Construct a mapping from contig ID to :class:`cyvcf2.VCF` object.
        """

        contig2file: Dict[str, str | pathlib.Path] = {}
        contig2vcf: Dict[str, cyvcf2.VCF] = {}

        files = list(files)
        if len(set(files)) != len(files):
            raise ValueError("File list contains duplicates.")

        if individuals is not None:
            individuals = list(individuals)
            if len(set(individuals)) != len(individuals):
                raise ValueError("Individuals list contains duplicates.")

        all_samples = None
        all_samples_file = None

        if contig_lengths is None:
            contig_lengths = {}
            contigs = None
        else:
            contigs = set(contig_lengths)
        contigs_seen = set()

        for file in files:
            vcf = cyvcf2.VCF(file, samples=individuals, lazy=True, threads=None)
            if "GT" not in vcf:
                raise ValueError(f"{file} doesn't contain GT field.")
            if not (
                pathlib.Path(f"{file}.tbi").exists()
                or pathlib.Path(f"{file}.csi").exists()
            ):
                raise ValueError(f"No index found for {file}.")
            if individuals is None:
                if all_samples is None:
                    all_samples = set(vcf.samples)
                    all_samples_file = file
                elif all_samples != set(vcf.samples):
                    raise ValueError(
                        f"{file} has different samples than {all_samples_file}."
                    )
            else:
                individuals_not_found = set(individuals) - set(vcf.samples)
                if len(individuals_not_found) > 0:
                    raise ValueError(
                        f"Requested individuals not found in {file}: "
                        f"{', '.join(individuals_not_found)}."
                    )
            if contigs is None:
                try:
                    vcf.seqlens
                except AttributeError as e:
                    raise ValueError(
                        f"{file} doesn't contain contig lengths. "
                        "You must provide a contig_lengths argument."
                    ) from e
            for j, contig_id in enumerate(vcf.seqnames):
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
                contig2vcf[contig_id] = vcf
                if contigs is None:
                    contig_lengths[contig_id] = vcf.seqlens[j]  # type: ignore[index]

        if contigs is not None:
            contigs_not_found = contigs - contigs_seen
            if len(contigs_not_found) > 0:
                raise ValueError(
                    f"Requested contigs not found: {', '.join(contigs_not_found)}."
                )

        if len(contig2file) == 0:
            raise ValueError("No usable vcf/bcf files in the list.")

        if individuals is None:
            assert all_samples is not None
            individuals = list(all_samples)
        self._individuals = individuals
        self._contig2file = contig2file
        self._contig2vcf = contig2vcf
        self.lengths = np.fromiter(contig_lengths.values(), dtype=int)

        # The VCF objects are not valid in child processes, so we set a flag
        # before forking to ensure they're reopened when used in the child.
        self._needs_reopen = False
        os.register_at_fork(before=self._close)

    def __iter__(self):
        """
        Iterate over contig IDs in the bag.
        """
        yield from self._contig2vcf

    def __getitem__(self, contig_id: str) -> cyvcf2.VCF:
        """
        Get the :class:`cyvcf2.VCF` containing the given contig.

        :param contig_id:
            The contig ID.
        :return:
            The :class:`cyvcf2.VCF` object.
        :rtype: cyvcf2.VCF
        """
        if not isinstance(contig_id, str):
            raise TypeError("key must be a string")
        if self._needs_reopen:
            self._reopen()
        return self._contig2vcf[contig_id]

    def __len__(self) -> int:
        """
        The number of contigs in the mapping.

        :return:
            The size of the bag.
        """
        return len(self._contig2vcf)

    def _close(self):
        for contig_id in list(self):
            del self._contig2vcf[contig_id]
            self._contig2vcf[contig_id] = None
        self._needs_reopen = True

    def _reopen(self):
        file2vcf = {
            file: cyvcf2.VCF(file, samples=self._individuals, lazy=True, threads=1)
            for file in set(self._contig2file.values())
        }
        self._contig2vcf = {
            contig_id: file2vcf[file] for contig_id, file in self._contig2file.items()
        }
        self._needs_reopen = False

    def sample_regions(
        self, size: int, sequence_length: int, rng: np.random.Generator
    ) -> List[Tuple[str, int, int]]:
        """
        Sample a list of (chrom, start, end) triplets.

        :param size:
            Number of genomic windows to sample.
        :param sequence_length:
            Length of the sequence to sample.
        :param numpy.random.Generator rng:
            The numpy random number generator.
        :return:
            List of 3-tuples: (chrom, start, end).
            The start and end coordinates are 1-based and inclusive, to match
            the usual convention for 'chrom:start-end', e.g. in bcftools.
        """

        long_enough = self.lengths >= sequence_length
        if not np.any(long_enough):
            raise ValueError(f"No contigs with length >= {sequence_length}.")
        contig_ids = np.array(self, dtype=object)[long_enough]
        contig_lengths = self.lengths[long_enough]

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a genotype matrix uniformly at random from the genome.

        :param sequence_length:
            Length of the sequence to sample.
        :param max_missing_genotypes:
            Consider only sites with at most this many missing genotype calls.
        :param min_seg_sites:
            Sampled genotype matrix must have at least this many variable
            sites (after filtering sites for missingness).
        :param require_phased:
            If True, raise an error if genotypes are not phased.
        :param numpy.random.Generator rng:
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
        assert retries > 0
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
