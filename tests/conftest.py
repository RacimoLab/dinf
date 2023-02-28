import pytest

import dinf.cli
from tests import capture


@pytest.fixture(scope="session")
@pytest.mark.usefixtures("tmp_path_factory")
def discriminator_file(tmp_path_factory):
    discriminator_file = tmp_path_factory.mktemp("discr") / "discriminator.nn"
    ex = "examples/bottleneck/model.py"
    with capture() as cap:
        dinf.cli.main(
            f"""
            train
                --seed 1
                --parallelism 2
                --training-replicates 10
                --test-replicates 10
                --epochs 10
                --model {ex}
                --discriminator {discriminator_file}
            """.split()
        )
    assert cap.ret == 0
    assert discriminator_file.exists()
    return discriminator_file


@pytest.fixture(scope="session")
@pytest.mark.usefixtures("tmp_path_factory")
@pytest.mark.usefixtures("discriminator_file")
def data_file(tmp_path_factory, discriminator_file):
    data_file = tmp_path_factory.mktemp("data") / "data.npz"
    ex = "examples/bottleneck/model.py"
    with capture() as cap:
        dinf.cli.main(
            f"""
            predict
                --seed 2
                --parallelism 2
                --replicates 10
                --model {ex}
                --discriminator {discriminator_file}
                --output-file {data_file}
            """.split()
        )
    assert cap.ret == 0
    assert data_file.exists()
    return data_file


@pytest.fixture(scope="session")
@pytest.mark.usefixtures("tmp_path_factory")
def smc_outdir(tmp_path_factory):
    output_folder = tmp_path_factory.mktemp("out")
    ex = "examples/bottleneck/model.py"
    with capture() as cap:
        dinf.cli.main(
            f"""
            smc
                --seed 1
                --parallelism 2
                --training-replicates 10
                --test-replicates 10
                --proposal-replicates 10
                --top 5
                --model {ex}
                --iterations 3
                --output-folder {output_folder}
            """.split()
        )
    assert cap.ret == 0
    assert output_folder.exists()
    return output_folder
