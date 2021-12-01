from __future__ import annotations
import collections
import pathlib
import subprocess
from typing import List, Tuple

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


def create_vcf_dataset(path: pathlib.Path, contig_lengths: List[int], ploidy: int = 2):
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


class TestBagOfVcf:
    @pytest.mark.usefixtures("tmp_path")
    def test_create_vcf_bag(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[200_000, 100_000])
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        assert "0" not in vb
        assert "1" in vb
        assert "2" in vb
        assert "3" not in vb

    @pytest.mark.usefixtures("tmp_path")
    def test_create_bcf_bag(self, tmp_path):
        create_bcf_dataset(tmp_path, contig_lengths=[200_000, 100_000])
        vb = dinf.BagOfVcf(tmp_path.glob("*.bcf"))
        assert "0" not in vb
        assert "1" in vb
        assert "2" in vb
        assert "3" not in vb

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

    @pytest.mark.usefixtures("tmp_path")
    def test_file_list_duplicates(self, tmp_path):
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
    @pytest.mark.parametrize("sequence_length", [50, 2027, 10_000])
    def test_sample_regions(self, tmp_path, sequence_length):
        create_vcf_dataset(tmp_path, contig_lengths=[200_000, 100_000])
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        rng = np.random.default_rng(1234)
        num_regions = 10_000
        regions = vb.sample_regions(
            num_regions, sequence_length=sequence_length, rng=rng
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
        rng = np.random.default_rng(1234)
        G, positions = vb.sample_genotype_matrix(
            sequence_length=sequence_length,
            min_seg_sites=1,
            max_missing_genotypes=0,
            require_phased=True,
            rng=rng,
        )
        assert G.shape == (len(positions), num_individuals, ploidy)
        assert all(0 <= pos < sequence_length for pos in positions)

    @pytest.mark.usefixtures("tmp_path")
    def test_contigs_are_all_too_short(self, tmp_path):
        create_vcf_dataset(tmp_path, contig_lengths=[100_000])
        rng = np.random.default_rng(1234)
        vb = dinf.BagOfVcf(tmp_path.glob("*.vcf.gz"))
        with pytest.raises(ValueError, match="No contigs with length >="):
            vb.sample_genotype_matrix(
                sequence_length=1_000_000,
                min_seg_sites=1,
                max_missing_genotypes=0,
                require_phased=True,
                rng=rng,
            )
