import subprocess

import pytest

import dinf
import dinf.__main__

from .test_dinf import check_discriminator, check_ncf


class TestTopLevel:
    def test_help(self):
        out1 = subprocess.run(
            "python -m dinf -h".split(), check=True, stdout=subprocess.PIPE
        )
        assert b"mcmc-gan" in out1.stdout
        assert b"check" in out1.stdout

        out2 = subprocess.run(
            "python -m dinf --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

        # No args should also output the help.
        out3 = subprocess.run("python -m dinf".split(), stdout=subprocess.PIPE)
        assert out1.stdout == out3.stdout
        assert out3.returncode != 0

    def test_version(self):
        out = subprocess.run(
            "python -m dinf --version".split(),
            check=True,
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
        assert out.stdout.strip() == dinf.__version__


class TestCheck:
    def test_help(self):
        out1 = subprocess.run(
            "python -m dinf check -h".split(), check=True, stdout=subprocess.PIPE
        )
        out2 = subprocess.run(
            "python -m dinf check --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

    def test_example(self):
        ex = "examples/bottleneck/model.py"
        out = subprocess.run(
            f"python -m dinf check {ex}".split(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert not out.stdout
        assert not out.stderr


class TestAbcGan:
    def test_help(self):
        out1 = subprocess.run(
            "python -m dinf abc-gan -h".split(), check=True, stdout=subprocess.PIPE
        )
        out2 = subprocess.run(
            "python -m dinf abc-gan --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

    @pytest.mark.usefixtures("tmp_path")
    def test_abc_gan_example(self, tmp_path):
        working_directory = tmp_path / "work_dir"
        ex = "examples/bottleneck/model.py"
        subprocess.run(
            f"""
            python -m dinf abc-gan
                --parallelism 2
                --iterations 2
                --training-replicates 16
                --test-replicates 0
                --epochs 1
                --proposals 20
                --posteriors 7
                --working-directory {working_directory}
                {ex}
            """.split(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert working_directory.exists()
        genobuilder = dinf.Genobuilder._from_file(ex)
        for i in range(2):
            check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
            check_ncf(
                working_directory / f"{i}" / "abc.ncf",
                chains=1,
                draws=7,
                var_names=genobuilder.parameters,
                check_acceptance_rate=False,
            )


class TestAlfiMcmcGan:
    def test_help(self):
        out1 = subprocess.run(
            "python -m dinf alfi-mcmc-gan -h".split(),
            check=True,
            stdout=subprocess.PIPE,
        )
        out2 = subprocess.run(
            "python -m dinf alfi-mcmc-gan --help".split(),
            check=True,
            stdout=subprocess.PIPE,
        )
        assert out1.stdout == out2.stdout

    @pytest.mark.usefixtures("tmp_path")
    def test_alfi_mcmc_gan_example(self, tmp_path):
        working_directory = tmp_path / "work_dir"
        ex = "examples/bottleneck/model.py"
        subprocess.run(
            f"""
            python -m dinf alfi-mcmc-gan
                --parallelism 2
                --iterations 2
                --training-replicates 10
                --test-replicates 0
                --epochs 1
                --walkers 6
                --steps 1
                --working-directory {working_directory}
                {ex}
            """.split(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert working_directory.exists()
        genobuilder = dinf.Genobuilder._from_file(ex)
        for i in range(2):
            check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
            check_ncf(
                working_directory / f"{i}" / "mcmc.ncf",
                chains=6,
                draws=2,
                var_names=genobuilder.parameters,
                check_acceptance_rate=True,
            )


class TestMcmcGan:
    def test_help(self):
        out1 = subprocess.run(
            "python -m dinf mcmc-gan -h".split(), check=True, stdout=subprocess.PIPE
        )
        out2 = subprocess.run(
            "python -m dinf mcmc-gan --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

    @pytest.mark.usefixtures("tmp_path")
    def test_mcmc_gan_example(self, tmp_path):
        working_directory = tmp_path / "work_dir"
        ex = "examples/bottleneck/model.py"
        subprocess.run(
            f"""
            python -m dinf mcmc-gan
                --parallelism 2
                --iterations 2
                --training-replicates 10
                --test-replicates 0
                --epochs 1
                --walkers 6
                --steps 1
                --Dx-replicates 2
                --working-directory {working_directory}
                {ex}
            """.split(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert working_directory.exists()
        genobuilder = dinf.Genobuilder._from_file(ex)
        for i in range(2):
            check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
            check_ncf(
                working_directory / f"{i}" / "mcmc.ncf",
                chains=6,
                draws=1,
                var_names=genobuilder.parameters,
                check_acceptance_rate=True,
            )


class TestPgGan:
    def test_help(self):
        out1 = subprocess.run(
            "python -m dinf pg-gan -h".split(), check=True, stdout=subprocess.PIPE
        )
        out2 = subprocess.run(
            "python -m dinf pg-gan --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

    @pytest.mark.usefixtures("tmp_path")
    def test_mcmc_gan_example(self, tmp_path):
        num_proposals = 2
        working_directory = tmp_path / "work_dir"
        ex = "examples/bottleneck/model.py"
        subprocess.run(
            f"""
            python -m dinf pg-gan
                --parallelism 2
                --iterations 2
                --training-replicates 10
                --test-replicates 0
                --epochs 1
                --Dx-replicates 2
                --num-proposals {num_proposals}
                --working-directory {working_directory}
                {ex}
            """.split(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert working_directory.exists()
        genobuilder = dinf.Genobuilder._from_file(ex)
        num_parameters = len(genobuilder.parameters)
        for i in range(2):
            check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
            check_ncf(
                working_directory / f"{i}" / "pg-gan-proposals.ncf",
                chains=num_parameters * num_proposals + 1,
                draws=1,
                var_names=genobuilder.parameters,
                check_acceptance_rate=False,
            )
