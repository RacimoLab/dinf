from __future__ import annotations
import collections
import pathlib
import subprocess
from typing import Callable, List, Tuple

import cyvcf2
import msprime
import numpy as np
import pytest

import dinf


def create_ts(*, length: int, ploidy: int, seeds: Tuple[int, int]):
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=10_000)
    demography.add_population(name="B", initial_size=5_000)
    demography.add_population(name="C", initial_size=1_000)
    demography.add_population_split(time=1000, derived=["A", "B"], ancestral="C")
    ts = msprime.sim_ancestry(
        samples={"A": 128, "B": 128},
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
):
    for contig_id, contig_length in enumerate(contig_lengths, 1):
        seeds = (contig_id + 1, contig_id + 2)
        ts = create_ts(length=contig_length, ploidy=ploidy, seeds=seeds)
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

    @pytest.mark.usefixtures("tmp_path")
    def test_mismatched_ploidy_among_individuals(self, tmp_path):
        num_individuals = 3
        contig_length = 100_000
        header = get_vcf_header(
            num_individuals=num_individuals, contig_length=contig_length
        )

        vline1 = "1\t1234\t.\tA\tC\t.\tPASS\t.\tGT\t"
        vline1 += "\t".join(["0|1"] * num_individuals)
        vline1 += "\n"

        vline2 = "1\t4321\t.\tA\tG\t.\tPASS\t.\tGT\t"
        vline2 += "\t".join(["1|0"] + ["1"] * (num_individuals - 1))
        vline2 += "\n"

        vcf_path = tmp_path / "1.vcf"
        with open(vcf_path, "w") as f:
            f.write(header + vline1 + vline2)
        vcf_path = bcftools_index(vcf_path)

        vcf = cyvcf2.VCF(vcf_path)
        start = 1
        end = contig_length

        with pytest.raises(ValueError, match="Mismatched ploidy among individuals"):
            dinf.vcf.get_genotype_matrix(
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
        for c_id, c_len in zip(vb, vb.contig_lengths):
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
        for c_id, c_len in zip(vb, vb.contig_lengths):
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

    @pytest.mark.parametrize("num_individuals", [20, 97])
    @pytest.mark.usefixtures("tmp_path")
    def test_individuals(self, tmp_path, num_individuals):
        samples = create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        individuals = samples["A"][:num_individuals]
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), individuals=individuals)
        assert vb["1"].samples == individuals

    @pytest.mark.filterwarnings("ignore:not all requested samples found:UserWarning")
    @pytest.mark.usefixtures("tmp_path")
    def test_missing_individuals(self, tmp_path):
        samples = create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        individuals = ["nonexistent_1"] + samples["A"] + ["nonexistent_2"]
        with pytest.raises(ValueError, match="individuals not found") as err:
            dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), individuals=individuals)
        assert "nonexistent_1" in err.value.args[0]
        assert "nonexistent_2" in err.value.args[0]
        for ind in samples["A"]:
            assert ind not in err.value.args[0]

    @pytest.mark.usefixtures("tmp_path")
    def test_duplicate_individuals(self, tmp_path):
        samples = create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        individuals = samples["A"] + [samples["A"][0]]
        with pytest.raises(ValueError, match="Individuals list contains duplicates"):
            dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), individuals=individuals)

    @pytest.mark.usefixtures("tmp_path")
    def test_contigs(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[100_000, 200_000, 300_000])
        contigs = ["1", "3"]
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), contigs=contigs)
        assert set(vb) == set(contigs)

    @pytest.mark.usefixtures("tmp_path")
    def test_missing_contigs(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[100_000, 200_000])
        contigs = ["nonexistent_a", "1", "2", "nonexistent_b"]
        with pytest.raises(ValueError, match="contigs not found") as err:
            dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), contigs=contigs)
        assert "nonexistent_a" in err.value.args[0]
        assert "nonexistent_b" in err.value.args[0]
        assert "1" not in err.value.args[0]
        assert "2" not in err.value.args[0]

    @pytest.mark.usefixtures("tmp_path")
    def test_duplicate_contigs(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        with pytest.raises(ValueError, match="Contigs list contains duplicates"):
            dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"), contigs=["1", "1"])

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
            contig_id: vb.contig_lengths[j] for j, contig_id in enumerate(vb.keys())
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
            tmp_path.glob("*.vcf.gz"), individuals=vcf_samples["A"][:num_individuals]
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
        with pytest.raises(ValueError, match="No contigs with length >="):
            vb.sample_genotype_matrix(
                sequence_length=1_000_000,
                min_seg_sites=1,
                max_missing_genotypes=0,
                require_phased=True,
                rng=np.random.default_rng(1234),
            )
