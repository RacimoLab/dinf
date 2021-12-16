import subprocess
import sys

import pytest

import dinf
import dinf.__main__

from .test_dinf import check_discriminator, check_ncf


def test_get_user_genobuilder_file_not_found():
    with pytest.raises(FileNotFoundError, match=r"nonexistent.py"):
        dinf.__main__._get_user_genobuilder("nonexistent.py")


@pytest.mark.usefixtures("tmp_path")
def test_get_user_genobuilder_obj_not_found(tmp_path):
    filename = tmp_path / "model.py"
    with open(filename, "w") as f:
        f.write("geeenobilder = {}\n")
    with pytest.raises(AttributeError, match="genobuilder not found"):
        dinf.__main__._get_user_genobuilder(filename)


@pytest.mark.usefixtures("tmp_path")
def test_get_user_genobuilder_obj_wrong_type(tmp_path):
    filename = tmp_path / "model.py"
    with open(filename, "w") as f:
        f.write("genobuilder = {}\n")
    with pytest.raises(TypeError, match="not a .*Genobuilder"):
        dinf.__main__._get_user_genobuilder(filename)


class TestTopLevel:
    def test_help(self):
        out1 = subprocess.run(
            "python -m dinf -h".split(), check=True, stdout=subprocess.PIPE
        )
        assert b"mcmc-gan" in out1.stdout
        assert b"check" in out1.stdout
        # Not supported.
        assert b"abc-gan" not in out1.stdout

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

    @pytest.mark.skipif(
        sys.platform.startswith("darwin"), reason="Deadlock on Github Actions."
    )
    @pytest.mark.usefixtures("tmp_path")
    def test_example(self, tmp_path):
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
        genobuilder = dinf.__main__._get_user_genobuilder(ex)
        for i in range(2):
            check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
            check_ncf(
                working_directory / f"{i}" / "abc.ncf",
                chains=1,
                draws=7,
                var_names=genobuilder.parameters,
                check_acceptance_rate=False,
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

    @pytest.mark.skipif(
        sys.platform.startswith("darwin"), reason="Deadlock on Github Actions."
    )
    @pytest.mark.usefixtures("tmp_path")
    def test_example(self, tmp_path):
        working_directory = tmp_path / "work_dir"
        ex = "examples/bottleneck/model.py"
        subprocess.run(
            f"""
            python -m dinf mcmc-gan
                --parallelism 2
                --iterations 2
                --training-replicates 16
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
        genobuilder = dinf.__main__._get_user_genobuilder(ex)
        for i in range(2):
            check_discriminator(working_directory / f"{i}" / "discriminator.pkl")
            check_ncf(
                working_directory / f"{i}" / "mcmc.ncf",
                chains=6,
                draws=1,
                var_names=genobuilder.parameters,
                check_acceptance_rate=True,
            )
