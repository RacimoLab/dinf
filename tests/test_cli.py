import pytest

import dinf
import dinf.cli
from tests import capture
from .test_dinf import check_discriminator, check_npz


@pytest.mark.usefixtures("tmp_path")
def test_check_output_file(tmp_path):
    file = tmp_path / "foo"
    dinf.cli._check_output_file(file)
    # If it succeeds, it should succeed again.
    dinf.cli._check_output_file(file)

    file.touch()
    # The file is not allowed to exist already.
    with pytest.raises(ValueError, match="file already exists"):
        dinf.cli._check_output_file(file)

    # Directory. Should raise an error, but we don't care what type.
    with pytest.raises(Exception):
        dinf.cli._check_output_file(tmp_path)

    inaccessible_folder = tmp_path / "inaccessible"
    inaccessible_folder.mkdir()
    inaccessible_folder.chmod(0o000)
    # Inacessible. Should raise an error, but we don't care what type.
    with pytest.raises(Exception):
        dinf.cli._check_output_file(inaccessible_folder / "foo")


class TestTopLevel:
    def test_help(self):
        with capture() as cap1:
            dinf.cli.main(["-h"])
        assert cap1.ret == 0
        assert "mcmc-gan" in cap1.out
        assert "check" in cap1.out

        with capture() as cap2:
            dinf.cli.main(["--help"])
        assert cap2.ret == 0
        assert cap2.out == cap1.out

        # No args should also output the help.
        with capture() as cap3:
            dinf.cli.main([])
        assert cap3.ret != 0
        assert cap3.out == cap1.out

    def test_version(self):
        with capture() as cap:
            dinf.cli.main(["--version"])
        assert cap.ret == 0
        assert cap.out.strip() == dinf.__version__


class TestCheck:
    def test_help(self):
        with capture() as cap1:
            dinf.cli.main("check -h".split())
        assert cap1.ret == 0

        with capture() as cap2:
            dinf.cli.main("check --help".split())
        assert cap2.ret == 0

        assert cap1.out == cap2.out

    def test_example(self):
        ex = "examples/bottleneck/model.py"
        with capture() as cap:
            dinf.cli.main(f"check --model {ex}".split())
        assert cap.ret == 0
        assert not cap.out


class TestAbcGan:
    def test_help(self):
        with capture() as cap1:
            dinf.cli.main("abc-gan -h".split())
        assert cap1.ret == 0

        with capture() as cap2:
            dinf.cli.main("abc-gan --help".split())
        assert cap2.ret == 0

        assert cap1.out == cap2.out

    @pytest.mark.usefixtures("tmp_path")
    def test_abc_gan_example(self, tmp_path):
        output_folder = tmp_path / "work_dir"
        ex = "examples/bottleneck/model.py"
        with capture() as cap:
            dinf.cli.main(
                f"""
                abc-gan
                    --seed 1
                    --parallelism 2
                    --iterations 2
                    --training-replicates 4
                    --test-replicates 4
                    --proposal-replicates 4
                    --epochs 1
                    --output-folder {output_folder}
                    --model {ex}
                """.split()
            )
        assert cap.ret == 0
        assert output_folder.exists()

        dinf_model = dinf.DinfModel.from_file(ex)
        for i in range(2):
            check_discriminator(output_folder / f"{i}" / "discriminator.nn", dinf_model)
            check_npz(
                output_folder / f"{i}" / "abc.npz",
                chains=1,
                draws=4,
                parameters=dinf_model.parameters,
            )


class TestMcmcGan:
    def test_help(self):
        with capture() as cap1:
            dinf.cli.main("mcmc-gan -h".split())
        assert cap1.ret == 0

        with capture() as cap2:
            dinf.cli.main("mcmc-gan --help".split())
        assert cap2.ret == 0

        assert cap1.out == cap2.out

    @pytest.mark.usefixtures("tmp_path")
    def test_mcmc_gan_example(self, tmp_path):
        output_folder = tmp_path / "work_dir"
        ex = "examples/bottleneck/model.py"
        with capture() as cap:
            dinf.cli.main(
                f"""
                mcmc-gan
                    --seed 1
                    --parallelism 2
                    --iterations 2
                    --training-replicates 10
                    --test-replicates 0
                    --epochs 1
                    --walkers 6
                    --steps 1
                    --Dx-replicates 2
                    --output-folder {output_folder}
                    --model {ex}
                """.split()
            )
        assert cap.ret == 0
        assert output_folder.exists()

        dinf_model = dinf.DinfModel.from_file(ex)
        for i in range(2):
            check_discriminator(output_folder / f"{i}" / "discriminator.nn", dinf_model)
            check_npz(
                output_folder / f"{i}" / "mcmc.npz",
                chains=6,
                draws=1,
                parameters=dinf_model.parameters,
            )


class TestPgGan:
    def test_help(self):
        with capture() as cap1:
            dinf.cli.main("pg-gan -h".split())
        assert cap1.ret == 0

        with capture() as cap2:
            dinf.cli.main("pg-gan --help".split())
        assert cap2.ret == 0

        assert cap1.out == cap2.out

    @pytest.mark.usefixtures("tmp_path")
    def test_pg_gan_example(self, tmp_path):
        num_proposals = 2
        output_folder = tmp_path / "work_dir"
        ex = "examples/bottleneck/model.py"
        with capture() as cap:
            dinf.cli.main(
                f"""
                pg-gan
                    --seed 1
                    --parallelism 2
                    --iterations 2
                    --training-replicates 10
                    --test-replicates 0
                    --epochs 1
                    --Dx-replicates 2
                    --num-proposals {num_proposals}
                    --output-folder {output_folder}
                    --model {ex}
                """.split()
            )
        assert cap.ret == 0
        assert output_folder.exists()

        dinf_model = dinf.DinfModel.from_file(ex)
        num_parameters = len(dinf_model.parameters)
        for i in range(2):
            check_discriminator(output_folder / f"{i}" / "discriminator.nn", dinf_model)
            check_npz(
                output_folder / f"{i}" / "pg-gan-proposals.npz",
                chains=1,
                draws=num_parameters * num_proposals + 1,
                parameters=dinf_model.parameters,
            )


class TestTrain:
    def test_help(self):
        with capture() as cap1:
            dinf.cli.main("train -h".split())
        assert cap1.ret == 0

        with capture() as cap2:
            dinf.cli.main("train --help".split())
        assert cap2.ret == 0

        assert cap1.out == cap2.out

    @pytest.mark.usefixtures("tmp_path")
    def test_train_example(self, tmp_path):
        discriminator_file = tmp_path / "discriminator.nn"
        ex = "examples/bottleneck/model.py"
        with capture() as cap:
            dinf.cli.main(
                f"""
                train
                    --seed 1
                    --parallelism 2
                    --training-replicates 10
                    --test-replicates 0
                    --epochs 1
                    --model {ex}
                    --discriminator {discriminator_file}
                """.split()
            )
        assert cap.ret == 0
        dinf_model = dinf.DinfModel.from_file(ex)
        check_discriminator(discriminator_file, dinf_model)


class TestPredict:
    def test_help(self):
        with capture() as cap1:
            dinf.cli.main("predict -h".split())
        assert cap1.ret == 0

        with capture() as cap2:
            dinf.cli.main("predict --help".split())
        assert cap2.ret == 0

        assert cap1.out == cap2.out

    @pytest.mark.parametrize("sample_target", [True, False])
    @pytest.mark.usefixtures("tmp_path")
    def test_predict_example(self, tmp_path, sample_target):
        discriminator_file = tmp_path / "discriminator.nn"
        ex = "examples/bottleneck/model.py"
        with capture() as cap:
            dinf.cli.main(
                f"""
                train
                    --seed 1
                    --parallelism 2
                    --training-replicates 10
                    --test-replicates 0
                    --epochs 1
                    --model {ex}
                    --discriminator {discriminator_file}
                """.split()
            )
        assert cap.ret == 0
        dinf_model = dinf.DinfModel.from_file(ex)
        check_discriminator(discriminator_file, dinf_model)

        target = "--target" if sample_target else ""
        output_file = tmp_path / "output.npz"

        with capture() as cap:
            dinf.cli.main(
                f"""
                predict
                    --seed 1
                    --parallelism 2
                    --replicates 10
                    {target}
                    --model {ex}
                    --discriminator {discriminator_file}
                    --output-file {output_file}
                """.split()
            )
        assert cap.ret == 0

        data = check_npz(
            output_file,
            chains=1,
            draws=10,
            parameters=dinf_model.parameters if not sample_target else None,
        )
        if sample_target:
            assert data.dtype.names == ("_Pr",)
