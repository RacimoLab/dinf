import subprocess

import pytest

import dinf
import dinf.__main__

from .test_dinf import check_discriminator, check_npz


@pytest.mark.usefixtures("tmp_path")
def test_check_output_file(tmp_path):
    file = tmp_path / "foo"
    dinf.cli.check_output_file(file)
    # If it succeeds, it should succeed again.
    dinf.cli.check_output_file(file)

    file.touch()
    # The file is not allowed to exist already.
    with pytest.raises(ValueError, match="file already exists"):
        dinf.cli.check_output_file(file)

    # Directory. Should raise an error, but we don't care what type.
    with pytest.raises(Exception):
        dinf.cli.check_output_file(tmp_path)

    inaccessible_folder = tmp_path / "inaccessible"
    inaccessible_folder.mkdir()
    inaccessible_folder.chmod(0o000)
    # Inacessible. Should raise an error, but we don't care what type.
    with pytest.raises(Exception):
        dinf.cli.check_output_file(inaccessible_folder / "foo")


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
                --seed 1
                --parallelism 2
                --iterations 2
                --training-replicates 8
                --test-replicates 8
                --epochs 1
                --working-directory {working_directory}
                {ex}
            """.split(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert working_directory.exists()
        dinf_model = dinf.DinfModel.from_file(ex)
        for i in range(2):
            check_discriminator(
                working_directory / f"{i}" / "discriminator.nn", dinf_model
            )
            check_npz(
                working_directory / f"{i}" / "abc.npz",
                chains=1,
                draws=4,
                parameters=dinf_model.parameters,
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
                --seed 1
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
        dinf_model = dinf.DinfModel.from_file(ex)
        for i in range(2):
            check_discriminator(
                working_directory / f"{i}" / "discriminator.nn", dinf_model
            )
            check_npz(
                working_directory / f"{i}" / "mcmc.npz",
                chains=6,
                draws=2,
                parameters=dinf_model.parameters,
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
                --seed 1
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
        dinf_model = dinf.DinfModel.from_file(ex)
        for i in range(2):
            check_discriminator(
                working_directory / f"{i}" / "discriminator.nn", dinf_model
            )
            check_npz(
                working_directory / f"{i}" / "mcmc.npz",
                chains=6,
                draws=1,
                parameters=dinf_model.parameters,
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
                --seed 1
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
        dinf_model = dinf.DinfModel.from_file(ex)
        num_parameters = len(dinf_model.parameters)
        for i in range(2):
            check_discriminator(
                working_directory / f"{i}" / "discriminator.nn", dinf_model
            )
            check_npz(
                working_directory / f"{i}" / "pg-gan-proposals.npz",
                chains=1,
                draws=num_parameters * num_proposals + 1,
                parameters=dinf_model.parameters,
            )


class TestTrain:
    def test_help(self):
        out1 = subprocess.run(
            "python -m dinf train -h".split(), check=True, stdout=subprocess.PIPE
        )
        out2 = subprocess.run(
            "python -m dinf train --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

    @pytest.mark.usefixtures("tmp_path")
    def test_train_example(self, tmp_path):
        discriminator_file = tmp_path / "discriminator.nn"
        ex = "examples/bottleneck/model.py"
        subprocess.run(
            f"""
            python -m dinf train
                --seed 1
                --parallelism 2
                --training-replicates 10
                --test-replicates 0
                --epochs 1
                {ex}
                {discriminator_file}
            """.split(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        dinf_model = dinf.DinfModel.from_file(ex)
        check_discriminator(discriminator_file, dinf_model)


class TestPredict:
    def test_help(self):
        out1 = subprocess.run(
            "python -m dinf predict -h".split(), check=True, stdout=subprocess.PIPE
        )
        out2 = subprocess.run(
            "python -m dinf predict --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

    @pytest.mark.parametrize("sample_target", [True, False])
    @pytest.mark.usefixtures("tmp_path")
    def test_predict_example(self, tmp_path, sample_target):
        discriminator_file = tmp_path / "discriminator.nn"
        ex = "examples/bottleneck/model.py"
        subprocess.run(
            f"""
            python -m dinf train
                --seed 1
                --parallelism 2
                --training-replicates 10
                --test-replicates 0
                --epochs 1
                {ex}
                {discriminator_file}
            """.split(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        dinf_model = dinf.DinfModel.from_file(ex)
        check_discriminator(discriminator_file, dinf_model)

        target = "--target" if sample_target else ""
        output_file = tmp_path / "output.npz"
        subprocess.run(
            f"""
            python -m dinf predict
                --seed 1
                --parallelism 2
                --replicates 10
                {target}
                {ex}
                {discriminator_file}
                {output_file}
            """.split(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        data = check_npz(
            output_file,
            chains=1,
            draws=10,
            parameters=dinf_model.parameters if not sample_target else None,
        )
        if sample_target:
            assert data.dtype.names == ("_Pr",)
