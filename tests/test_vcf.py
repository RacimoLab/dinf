from __future__ import annotations
import atexit
import collections
import multiprocessing
import pathlib
import re
import subprocess
import tempfile
from typing import Callable, List, Tuple

import cyvcf2
import msprime
import numpy as np
import pytest

import dinf

_1kgp_test_metadata = """\
FamilyID SampleID FatherID MotherID Sex Population Superpopulation
HG00096 HG00096 0 0 1 GBR EUR
HG00097 HG00097 0 0 2 GBR EUR
SH001 HG00403 0 0 1 CHS EAS
SH001 HG00404 0 0 2 CHS EAS
SH001 HG00405 HG00403 HG00404 2 CHS EAS
"""


class TestGetSamplesFrom1KgpMetadata:
    @pytest.mark.usefixtures("tmp_path")
    def test_simple(self, tmp_path):
        filename = tmp_path / "1kgp_metadata.txt"
        with open(filename, "w") as f:
            f.write(_1kgp_test_metadata)
        gbr = dinf.get_samples_from_1kgp_metadata(filename, populations=["GBR"])
        assert gbr == {"GBR": ["HG00096", "HG00097"]}
        chs = dinf.get_samples_from_1kgp_metadata(filename, populations=["CHS"])
        assert chs == {"CHS": ["HG00403", "HG00404"]}
        samples = dinf.get_samples_from_1kgp_metadata(
            filename, populations=["GBR", "CHS"]
        )
        assert samples == dict(**gbr, **chs)


class TestGetContigLengths:
    @pytest.mark.usefixtures("tmp_path")
    def test_simple(self, tmp_path):
        contigs = {"a": 100, "b": 200, "c": 400}
        with open(tmp_path / "contigs.txt", "w") as f:
            for c_id, c_len in contigs.items():
                print(c_id, c_len, file=f)
        loaded_contigs = dinf.get_contig_lengths(tmp_path / "contigs.txt")
        assert loaded_contigs == contigs

    @pytest.mark.usefixtures("tmp_path")
    def test_extra_columns(self, tmp_path):
        contigs = {"a": 100, "b": 200, "c": 400}
        with open(tmp_path / "contigs.txt", "w") as f:
            for j, (c_id, c_len) in enumerate(contigs.items()):
                print(c_id, c_len, j, j * j + 10, file=f)
        loaded_contigs = dinf.get_contig_lengths(tmp_path / "contigs.txt")
        assert loaded_contigs == contigs

    @pytest.mark.usefixtures("tmp_path")
    def test_keep_contigs(self, tmp_path):
        contigs = {"a": 100, "b": 200, "c": 400}
        with open(tmp_path / "contigs.txt", "w") as f:
            for c_id, c_len in contigs.items():
                print(c_id, c_len, file=f)
        keep_contigs = ["a", "c"]
        loaded_contigs = dinf.get_contig_lengths(
            tmp_path / "contigs.txt", keep_contigs=keep_contigs
        )
        assert loaded_contigs == {k: contigs[k] for k in keep_contigs}

    @pytest.mark.usefixtures("tmp_path")
    def test_missing_contigs(self, tmp_path):
        contigs = {"1": 100, "2": 200, "3": 400}
        with open(tmp_path / "contigs.txt", "w") as f:
            for c_id, c_len in contigs.items():
                print(c_id, c_len, file=f)
        with pytest.raises(ValueError, match="contigs not found") as err:
            dinf.get_contig_lengths(
                tmp_path / "contigs.txt",
                keep_contigs=["nonexistent_a", "1", "2", "nonexistent_b"],
            )
        assert "nonexistent_a" in err.value.args[0]
        assert "nonexistent_b" in err.value.args[0]
        assert "1" not in err.value.args[0]
        assert "2" not in err.value.args[0]


def create_ts(*, length: int, ploidy: int, seeds: Tuple[int, int], num_samples: int):
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=10_000)
    demography.add_population(name="B", initial_size=5_000)
    demography.add_population(name="C", initial_size=1_000)
    demography.add_population_split(time=1000, derived=["A", "B"], ancestral="C")
    ts = msprime.sim_ancestry(
        samples={"A": num_samples, "B": num_samples},
        demography=demography,
        sequence_length=length,
        recombination_rate=1.25e-8,
        ploidy=ploidy,
        random_seed=seeds[0],
    )
    return msprime.sim_mutations(ts, rate=1.25e-8, random_seed=seeds[1])


def bcftools_index(filename: str | pathlib.Path):
    if str(filename).endswith(".vcf"):
        subprocess.run(f"bgzip {filename}".split())
        filename = str(filename) + ".gz"
    subprocess.run(f"bcftools index {filename}".split())
    return filename


def create_vcf_dataset(
    path: pathlib.Path,
    contig_lengths: List[int],
    ploidy: int = 2,
    transform_func: Callable | None = None,
    num_samples: int = 128,
):
    for contig_id, contig_length in enumerate(contig_lengths, 1):
        seeds = (contig_id + 1, contig_id + 2)
        ts = create_ts(
            length=contig_length, ploidy=ploidy, seeds=seeds, num_samples=num_samples
        )
        for ind in ts.individuals():
            assert len(ind.nodes) == ploidy
            assert len(set(ts.node(n).population for n in ind.nodes)) == 1
        pop = {j: pop.metadata["name"] for j, pop in enumerate(ts.populations())}
        individual_names = [
            pop[ts.node(ind.nodes[0]).population] + f"_{ind.id:03d}"
            for ind in ts.individuals()
        ]
        vcf_path = path / f"{contig_id}.vcf"
        with open(vcf_path, "w") as f:
            ts.write_vcf(
                f,
                contig_id=str(contig_id),
                individual_names=individual_names,
                position_transform=lambda x: np.floor(x) + 1,
            )
        if transform_func is not None:
            transform_func(vcf_path)
        bcftools_index(vcf_path)

    samples = collections.defaultdict(list)
    for ind, ind_name in zip(ts.individuals(), individual_names):
        pop_name = pop[ts.node(ind.nodes[0]).population]
        samples[pop_name].append(ind_name)

    return samples


def create_bcf_dataset(path: pathlib.Path, contig_lengths: List[int], ploidy: int = 2):
    samples = create_vcf_dataset(path, contig_lengths, ploidy)
    for vcf_path in path.glob("*.vcf.gz"):
        bcf_path = pathlib.Path(str(vcf_path)[: -len(".vcf.gz")] + ".bcf")
        subprocess.run(f"bcftools view -Ob -o {bcf_path} {vcf_path}".split())
        assert bcf_path.exists()
        bcftools_index(bcf_path)
        # remove vcf bits
        vcf_path.unlink()
        pathlib.Path(f"{vcf_path}.csi").unlink()
    return samples


def get_vcf_header(*, num_individuals: int, contig_length: int):
    top_part = f"""\
##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=1,length={contig_length}>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##ALT=<ID=DEL,Description="Deletion">
##ALT=<ID=DEL:ME:ALU,Description="Deletion of ALU element">
##ALT=<ID=DEL:ME:L1,Description="Deletion of L1 element">
##ALT=<ID=DUP,Description="Duplication">
##ALT=<ID=DUP:TANDEM,Description="Tandem Duplication">
##ALT=<ID=INS,Description="Insertion of novel sequence">
##ALT=<ID=INS:ME:ALU,Description="Insertion of ALU element">
##ALT=<ID=INS:ME:L1,Description="Insertion of L1 element">
##ALT=<ID=INV,Description="Inversion">
##ALT=<ID=CNV,Description="Copy number variable region">
"""
    chrom_line = (
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(f"ind_{j}" for j in range(num_individuals))
        + "\n"
    )

    return top_part + chrom_line


class TestGetGenotypeMatrix:
    @pytest.mark.usefixtures("tmp_path")
    def test_simple(self, tmp_path):
        num_individuals = 3
        contig_length = 100_000
        header = get_vcf_header(
            num_individuals=num_individuals, contig_length=contig_length
        )

        vline1 = "1\t1234\t.\tA\tC\t.\tPASS\t.\tGT\t"
        vline1 += "\t".join(["0|1"] * num_individuals)
        vline1 += "\n"

        vline2 = "1\t4321\t.\tA\tG\t.\tPASS\t.\tGT\t"
        vline2 += "\t".join(["1|0"] * num_individuals)
        vline2 += "\n"

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            f.write(header + vline1 + vline2)
        vcf_path = bcftools_index(vcf_path)

        vcf = cyvcf2.VCF(vcf_path)
        start = 1
        end = contig_length
        G, positions = dinf.vcf.get_genotype_matrix(
            vcf,
            chrom="1",
            start=start,
            end=end,
            max_missing_genotypes=0,
            require_phased=True,
        )
        assert len(G) == len(positions)
        # positions are 0-based, so subtract the start.
        np.testing.assert_array_equal(positions, [1234 - start, 4321 - start])
        assert G.shape == (2, num_individuals, 2)
        np.testing.assert_array_equal(
            G, [[[0, 1], [0, 1], [0, 1]], [[1, 0], [1, 0], [1, 0]]]
        )

    @pytest.mark.parametrize(
        "ref,alt",
        [
            ("A", "AA"),  # insertion
            ("AA", "A"),  # deletion
            ("A", "."),  # ref only
            ("ACGT", "<INV>"),  # inversion
            ("AC", "<DUP>"),  # duplicate
            ("G", "G]17:198982]"),  # breakend
            ("T", "]13:123456]T"),  # breakend
        ],
    )
    @pytest.mark.usefixtures("tmp_path")
    def test_filter_non_snps(self, tmp_path, ref, alt):
        num_individuals = 3
        contig_length = 100_000
        header = get_vcf_header(
            num_individuals=num_individuals, contig_length=contig_length
        )
        # Should get filtered.
        vline1 = f"1\t1234\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t"
        vline1 += "\t".join(["0|1"] * num_individuals)
        vline1 += "\n"
        # Shouldn't get filtered.
        vline2 = "1\t4321\t.\tA\tG\t.\tPASS\t.\tGT\t"
        vline2 += "\t".join(["1|0"] * num_individuals)
        vline2 += "\n"

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            f.write(header + vline1 + vline2)
        vcf_path = bcftools_index(vcf_path)

        vcf = cyvcf2.VCF(vcf_path)
        start = 1
        end = contig_length
        G, positions = dinf.vcf.get_genotype_matrix(
            vcf,
            chrom="1",
            start=start,
            end=end,
            max_missing_genotypes=0,
            require_phased=True,
        )
        assert len(G) == len(positions)
        # positions are 0-based, so subtract the start.
        np.testing.assert_array_equal(positions, [4321 - start])
        assert G.shape == (1, num_individuals, 2)
        np.testing.assert_array_equal(G, [[[1, 0], [1, 0], [1, 0]]])

    @pytest.mark.usefixtures("tmp_path")
    def test_filter_invariant_sites(self, tmp_path):
        num_individuals = 3
        contig_length = 100_000
        header = get_vcf_header(
            num_individuals=num_individuals, contig_length=contig_length
        )

        # Should get filtered.
        vline1 = "1\t1234\t.\tA\tC\t.\tPASS\t.\tGT\t"
        vline1 += "\t".join(["0|0"] * num_individuals)
        vline1 += "\n"
        # Shouldn't get filtered.
        vline2 = "1\t4321\t.\tA\tG\t.\tPASS\t.\tGT\t"
        vline2 += "\t".join(["1|0"] * num_individuals)
        vline2 += "\n"

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            f.write(header + vline1 + vline2)
        vcf_path = bcftools_index(vcf_path)

        vcf = cyvcf2.VCF(vcf_path)
        start = 1
        end = contig_length
        G, positions = dinf.vcf.get_genotype_matrix(
            vcf,
            chrom="1",
            start=start,
            end=end,
            max_missing_genotypes=0,
            require_phased=True,
        )
        assert len(G) == len(positions)
        # positions are 0-based, so subtract the start.
        np.testing.assert_array_equal(positions, [4321 - start])
        assert G.shape == (1, num_individuals, 2)
        np.testing.assert_array_equal(G, [[[1, 0], [1, 0], [1, 0]]])

    @pytest.mark.parametrize(
        "gts,max_missing_genotypes",
        [
            ([".|0", "1|1", "0|0"], 0),
            ([".|0", ".|1", "0|1"], 1),
            ([".|.", ".|1", "0|1"], 2),
        ],
    )
    @pytest.mark.usefixtures("tmp_path")
    def test_max_missing_genotypes(self, tmp_path, gts, max_missing_genotypes):
        num_individuals = len(gts)
        contig_length = 100_000
        header = get_vcf_header(
            num_individuals=num_individuals, contig_length=contig_length
        )

        # Should get filtered.
        vline1 = "1\t1234\t.\tA\tC\t.\tPASS\t.\tGT\t"
        vline1 += "\t".join(gts)
        vline1 += "\n"
        # Shouldn't get filtered.
        vline2 = "1\t4321\t.\tA\tG\t.\tPASS\t.\tGT\t"
        vline2 += "\t".join(["1|0"] * num_individuals)
        vline2 += "\n"

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            f.write(header + vline1 + vline2)
        vcf_path = bcftools_index(vcf_path)

        vcf = cyvcf2.VCF(vcf_path)
        start = 1
        end = contig_length
        G, positions = dinf.vcf.get_genotype_matrix(
            vcf,
            chrom="1",
            start=start,
            end=end,
            max_missing_genotypes=max_missing_genotypes,
            require_phased=True,
        )
        assert len(G) == len(positions)
        # positions are 0-based, so subtract the start.
        np.testing.assert_array_equal(positions, [4321 - start])
        assert G.shape == (1, num_individuals, 2)
        np.testing.assert_array_equal(G, [[[1, 0], [1, 0], [1, 0]]])

        # If we relax the threshold by 1, the site should be retained.
        G, positions = dinf.vcf.get_genotype_matrix(
            vcf,
            chrom="1",
            start=start,
            end=end,
            max_missing_genotypes=max_missing_genotypes + 1,
            require_phased=True,
        )
        assert len(G) == len(positions)
        # positions are 0-based, so subtract the start.
        np.testing.assert_array_equal(positions, [1234 - start, 4321 - start])
        gt2i = {".": -1, "0": 0, "1": 1}
        gtv = [[gt2i[g] for g in gt.split("|")] for gt in gts]
        np.testing.assert_array_equal(G, [gtv, [[1, 0], [1, 0], [1, 0]]])

    @pytest.mark.usefixtures("tmp_path")
    def test_require_phased(self, tmp_path):
        num_individuals = 3
        contig_length = 100_000
        header = get_vcf_header(
            num_individuals=num_individuals, contig_length=contig_length
        )

        vline1 = "1\t1234\t.\tA\tC\t.\tPASS\t.\tGT\t"
        vline1 += "\t".join(["0|1"] * num_individuals)
        vline1 += "\n"

        vline2 = "1\t4321\t.\tA\tG\t.\tPASS\t.\tGT\t"
        vline2 += "\t".join(["1/0"] * num_individuals)
        vline2 += "\n"

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            f.write(header + vline1 + vline2)
        vcf_path = bcftools_index(vcf_path)

        vcf = cyvcf2.VCF(vcf_path)
        start = 1
        end = contig_length
        G, positions = dinf.vcf.get_genotype_matrix(
            vcf,
            chrom="1",
            start=start,
            end=end,
            max_missing_genotypes=0,
            require_phased=False,
        )
        assert len(G) == len(positions)
        # positions are 0-based, so subtract the start.
        np.testing.assert_array_equal(positions, [1234 - start, 4321 - start])
        assert G.shape == (2, num_individuals, 2)
        np.testing.assert_array_equal(
            G, [[[0, 1], [0, 1], [0, 1]], [[1, 0], [1, 0], [1, 0]]]
        )

        with pytest.raises(ValueError, match="Unphased genotypes"):
            dinf.vcf.get_genotype_matrix(
                vcf,
                chrom="1",
                start=start,
                end=end,
                max_missing_genotypes=0,
                require_phased=True,
            )

    @pytest.mark.usefixtures("tmp_path")
    def test_mismatched_ploidy_among_sites(self, tmp_path):
        num_individuals = 3
        contig_length = 100_000
        header = get_vcf_header(
            num_individuals=num_individuals, contig_length=contig_length
        )

        vline1 = "1\t1234\t.\tA\tC\t.\tPASS\t.\tGT\t"
        vline1 += "\t".join(["0|1"] * num_individuals)
        vline1 += "\n"

        vline2 = "1\t4321\t.\tA\tG\t.\tPASS\t.\tGT\t"
        vline2 += "\t".join(["0"] + ["1"] * (num_individuals - 1))
        vline2 += "\n"

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            f.write(header + vline1 + vline2)
        vcf_path = bcftools_index(vcf_path)

        vcf = cyvcf2.VCF(vcf_path)
        start = 1
        end = contig_length

        with pytest.raises(ValueError, match="Mismatched ploidy among sites"):
            G, positions = dinf.vcf.get_genotype_matrix(
                vcf,
                chrom="1",
                start=start,
                end=end,
                max_missing_genotypes=0,
                require_phased=True,
            )


class TestBagOfVcf:
    @pytest.mark.usefixtures("tmp_path")
    def test_create_vcf_bag(self, tmp_path):
        contig_lengths = [200_000, 100_000]
        create_vcf_dataset(tmp_path, contig_lengths=contig_lengths)
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        assert "0" not in vb
        assert "1" in vb
        assert "2" in vb
        assert "3" not in vb
        assert len(vb) == 2
        assert set(vb) == set(["1", "2"])
        # tskit outputs vcf with wrong contig lengths.
        # https://github.com/tskit-dev/tskit/discussions/1993
        c2len = dict(zip(["1", "2"], contig_lengths))
        for c_id, c_len in zip(vb, vb.lengths):
            assert c_len - 1 == c2len[c_id]

    @pytest.mark.usefixtures("tmp_path")
    def test_create_bcf_bag(self, tmp_path):
        contig_lengths = [200_000, 100_000]
        create_bcf_dataset(tmp_path, contig_lengths=contig_lengths)
        vb = dinf.BagOfVcf(tmp_path.glob("*.bcf"))
        assert "0" not in vb
        assert "1" in vb
        assert "2" in vb
        assert "3" not in vb
        assert len(vb) == 2
        assert set(vb) == set(["1", "2"])
        # tskit outputs vcf with wrong contig lengths.
        # https://github.com/tskit-dev/tskit/discussions/1993
        c2len = dict(zip(["1", "2"], contig_lengths))
        for c_id, c_len in zip(vb, vb.lengths):
            assert c_len - 1 == c2len[c_id]

    @pytest.mark.usefixtures("tmp_path")
    def test_bad_getitem(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[200_000, 100_000])
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        with pytest.raises(TypeError):
            vb[1]
        with pytest.raises(TypeError):
            vb[2]

    @pytest.mark.usefixtures("tmp_path")
    def test_unindexed_vcf(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        for index in tmp_path.glob("*.csi"):
            index.unlink()
        with pytest.raises(ValueError, match="No index"):
            dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))

    @pytest.mark.usefixtures("tmp_path")
    def test_unindexed_bcf(self, tmp_path):
        create_bcf_dataset(tmp_path, contig_lengths=[100_000])
        for index in tmp_path.glob("*.csi"):
            index.unlink()
        with pytest.raises(ValueError, match="No index"):
            dinf.BagOfVcf(tmp_path.glob("*.bcf"))

    def test_missing_file(self):
        missing_file = "nonexistent.vcf.gz"
        with pytest.raises(OSError, match=missing_file):
            dinf.BagOfVcf([missing_file])

    def test_no_files(self):
        with pytest.raises(ValueError, match="No usable vcf/bcf files"):
            dinf.BagOfVcf([])

    @pytest.mark.usefixtures("tmp_path")
    def test_duplicate_files(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        files = 2 * list(tmp_path.glob("*.vcf.gz"))
        with pytest.raises(ValueError, match="File list contains duplicates"):
            dinf.BagOfVcf(files)

    @pytest.mark.usefixtures("tmp_path")
    def test_multiple_files_claim_a_contig(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        files = list(tmp_path.glob("*.vcf.gz"))
        dupe_dir = tmp_path / "dupe_dir"
        dupe_dir.symlink_to(tmp_path)
        files += list(dupe_dir.glob("*.vcf.gz"))
        with pytest.raises(ValueError, match="Both .* contain records for sequence"):
            dinf.BagOfVcf(files)

    @pytest.mark.usefixtures("tmp_path")
    def test_no_GT_field(self, tmp_path):
        def remove_GT_header(filename):
            with open(filename) as f:
                lines = f.readlines()
            lines = filter(lambda line: not line.startswith("##FORMAT=<ID=GT,"), lines)
            with open(filename, "w") as f:
                f.write("".join(lines))

        create_vcf_dataset(
            tmp_path, contig_lengths=[100_000], transform_func=remove_GT_header
        )
        with pytest.raises(ValueError, match="doesn't contain GT field"):
            dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))

    @pytest.mark.usefixtures("tmp_path")
    def test_samples(self, tmp_path):
        all_samples = create_vcf_dataset(tmp_path, contig_lengths=[100_000, 200_000])
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), samples=None)
        assert vb["1"].samples == all_samples["A"] + all_samples["B"]

    @pytest.mark.parametrize("num_individuals", [20, 97])
    @pytest.mark.usefixtures("tmp_path")
    def test_samples_one_population(self, tmp_path, num_individuals):
        all_samples = create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        samples = {"A": all_samples["A"][:num_individuals]}
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), samples=samples)
        assert vb["1"].samples == samples["A"]

    @pytest.mark.parametrize("num_individuals", [20, 97])
    @pytest.mark.usefixtures("tmp_path")
    def test_samples_two_populations(self, tmp_path, num_individuals):
        all_samples = create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        samples = {pop: all_samples[pop][:num_individuals] for pop in ("A", "B")}
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), samples=samples)
        assert vb["1"].samples == samples["A"] + samples["B"]

    @pytest.mark.filterwarnings("ignore:not all requested samples found:UserWarning")
    @pytest.mark.usefixtures("tmp_path")
    def test_missing_samples(self, tmp_path):
        all_samples = create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        samples = {"A": ["nonexistent_1"] + all_samples["A"] + ["nonexistent_2"]}
        with pytest.raises(ValueError, match="individuals not found") as err:
            dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), samples=samples)
        assert "nonexistent_1" in err.value.args[0]
        assert "nonexistent_2" in err.value.args[0]
        for ind in all_samples["A"]:
            assert ind not in err.value.args[0]

    @pytest.mark.usefixtures("tmp_path")
    def test_duplicate_samples(self, tmp_path):
        all_samples = create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        samples = {"A": all_samples["A"] + [all_samples["A"][0]]}
        with pytest.raises(ValueError, match="Individuals list contains duplicates"):
            dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), samples=samples)

    @pytest.mark.usefixtures("tmp_path")
    def test_contigs(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[100_000, 200_000, 300_000])
        contigs = {"1": 100_000, "3": 300_000}
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), contig_lengths=contigs)
        assert set(vb) == set(contigs)

    @pytest.mark.usefixtures("tmp_path")
    def test_missing_contigs(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[100_000, 200_000])
        contigs = {
            "nonexistent_a": 1000,
            "1": 100_000,
            "2": 200_000,
            "nonexistent_b": 1000,
        }
        with pytest.raises(ValueError, match="contigs not found") as err:
            dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), contig_lengths=contigs)
        assert "nonexistent_a" in err.value.args[0]
        assert "nonexistent_b" in err.value.args[0]
        assert "1" not in err.value.args[0]
        assert "2" not in err.value.args[0]

    @pytest.mark.usefixtures("tmp_path")
    def test_unused_contigs(self, tmp_path):
        # Contigs in the header should be ignored if they have no variants.
        def add_contig_c(filename):
            with open(filename) as f:
                lines = f.readlines()

            index = -1
            for j, line in enumerate(lines):
                if line.startswith("##contig"):
                    index = j + 1
            assert index > 0

            lines.insert(index, "##contig=<ID=c,length=100001>\n")

            with open(filename, "w") as f:
                f.write("".join(lines))

        create_vcf_dataset(
            tmp_path, contig_lengths=[100_000], transform_func=add_contig_c
        )
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        assert "c" not in vb

    @pytest.mark.usefixtures("tmp_path")
    def test_no_contig_lengths_in_vcf(self, tmp_path):
        def rm_contig_lengths(filename):
            with open(filename) as f:
                lines = f.readlines()

            new_lines = [
                re.sub(r",length=[0-9]+", "", line)
                if line.startswith("##contig")
                else line
                for line in lines
            ]

            with open(filename, "w") as f:
                f.write("".join(new_lines))

        create_vcf_dataset(
            tmp_path, contig_lengths=[100_000], transform_func=rm_contig_lengths
        )
        with pytest.raises(ValueError, match="provide a contig_lengths argument"):
            dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))

    @pytest.mark.usefixtures("tmp_path")
    def test_different_samples_in_different_files(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        create_vcf_dataset(
            tmp_path / "a", contig_lengths=[100_000, 200_000], num_samples=20
        )
        create_vcf_dataset(
            tmp_path / "b", contig_lengths=[100_000, 200_000], num_samples=40
        )
        with pytest.raises(ValueError, match="different samples"):
            dinf.BagOfVcf(
                [(tmp_path / "a" / "1.vcf.gz"), (tmp_path / "b" / "2.vcf.gz")]
            )

    @pytest.mark.parametrize("sequence_length", [50, 2027, 10_000])
    @pytest.mark.usefixtures("tmp_path")
    def test_sample_regions(self, tmp_path, sequence_length):
        create_vcf_dataset(tmp_path, contig_lengths=[200_000, 100_000])
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        num_regions = 10_000
        regions = vb.sample_regions(
            num_regions,
            sequence_length=sequence_length,
            rng=np.random.default_rng(1234),
        )
        contig2length = {
            contig_id: vb.lengths[j] for j, contig_id in enumerate(vb.keys())
        }
        assert len(regions) == num_regions
        assert all(chrom in vb for chrom, _, _ in regions)
        assert all(
            1 <= start <= end <= contig2length[chrom] for chrom, start, end in regions
        )
        assert all(end - start == sequence_length - 1 for _, start, end in regions)

    @pytest.mark.parametrize("ploidy", [1, 2, 3, 4])
    @pytest.mark.parametrize("num_individuals", [20, 97])
    @pytest.mark.parametrize("sequence_length", [10_000])
    @pytest.mark.usefixtures("tmp_path")
    def test_sample_genotype_matrix(
        self, tmp_path, sequence_length, num_individuals, ploidy
    ):
        vcf_samples = create_vcf_dataset(
            tmp_path, contig_lengths=[200_000, 100_000], ploidy=ploidy
        )
        vb = dinf.BagOfVcf(
            tmp_path.glob("*.vcf.gz"), samples={"A": vcf_samples["A"][:num_individuals]}
        )
        G, positions = vb.sample_genotype_matrix(
            sequence_length=sequence_length,
            min_seg_sites=1,
            max_missing_genotypes=0,
            require_phased=True,
            rng=np.random.default_rng(1234),
        )
        assert G.shape == (len(positions), num_individuals, ploidy)
        assert all(0 <= pos < sequence_length for pos in positions)

    @pytest.mark.usefixtures("tmp_path")
    def test_sample_genotype_matrix_retries(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[200_000, 100_000])
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        with pytest.raises(ValueError, match="Failed to sample genotype matrix"):
            vb.sample_genotype_matrix(
                sequence_length=10_000,
                min_seg_sites=10_000,
                max_missing_genotypes=0,
                require_phased=True,
                rng=np.random.default_rng(1234),
                retries=5,
            )

    @pytest.mark.usefixtures("tmp_path")
    def test_contigs_are_all_too_short(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        assert len(vb._regions) == 0
        assert all(vb.lengths < 1_000_000)
        with pytest.raises(ValueError, match=r"No contigs with length >="):
            vb.sample_genotype_matrix(
                sequence_length=1_000_000,
                min_seg_sites=1,
                max_missing_genotypes=0,
                require_phased=True,
                rng=np.random.default_rng(1234),
                retries=1,
            )


# When using a VCF inside a multiprocessing process pool, we must ensure that
# the cyvcf2.VCF() handles are not shared between distinct processes.
# https://github.com/RacimoLab/dinf/issues/40
# Actually, it's hard to test for this, because distinct worker processes may
# use the same virtual memory address for their own copy of the VCF handle,
# and thus id(vcf) could return the same value in distinct workers despite
# being a distinct handle (i.e. using a distinct OS file descriptor).
# So we just check that the worker process's vcf has a different id to
# the parent process's vcf. These ids aren't guaranteed to be different
# when using "forkserver" or "spawn" start methods, but in practice they will
# be different. In the "fork" case, both will be in memory simultaneously
# and thus are guaranteed to be different.

tmpdir = tempfile.TemporaryDirectory()
atexit.register(tmpdir.cleanup)
tmp_path = pathlib.Path(tmpdir.name)
create_vcf_dataset(tmp_path, contig_lengths=[100_000])
vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))


def _worker_using_vcf(parent_chrom_id):
    chrom1 = vb["1"]
    assert id(chrom1) != parent_chrom_id
    regions = list(chrom1("1:1-100000"))
    assert len(regions) >= 0
    pid = multiprocessing.current_process().pid
    return pid, id(chrom1)


@pytest.mark.parametrize("start_method", multiprocessing.get_all_start_methods())
def test_vcf_inside_process_pool(start_method):

    # Check VCF in the parent process.
    chrom1 = vb["1"]
    regions = list(chrom1("1:1-100000"))
    assert len(regions) >= 0
    parent_chrom1_id = id(chrom1)

    num_processes = 8
    num_jobs = 32
    worker_chrom1_ids_by_pid = collections.defaultdict(list)
    ctx = multiprocessing.get_context(start_method)
    pool = ctx.Pool(processes=num_processes)
    for pid, chrom1_id in pool.map(_worker_using_vcf, [(parent_chrom1_id)] * num_jobs):
        worker_chrom1_ids_by_pid[pid].append(chrom1_id)
    pool.close()
    pool.join()

    for pid, id_list in worker_chrom1_ids_by_pid.items():
        # Id should be the same for all jobs in a given worker.
        assert len(set(id_list)) == 1
