import collections
import subprocess

import pytest


@pytest.fixture(scope="session")
@pytest.mark.usefixtures("tmp_path_factory")
def discriminator_file(tmp_path_factory):
    discriminator_file = tmp_path_factory.mktemp("discr") / "discriminator.nn"
    ex = "examples/bottleneck/model.py"
    subprocess.run(
        f"""
        python -m dinf train
            --seed 1
            --parallelism 2
            --training-replicates 10
            --test-replicates 10
            --epochs 10
            --model {ex}
            --discriminator {discriminator_file}
        """.split(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert discriminator_file.exists()
    return discriminator_file


@pytest.fixture(scope="session")
@pytest.mark.usefixtures("tmp_path_factory")
@pytest.mark.usefixtures("discriminator_file")
def data_file(tmp_path_factory, discriminator_file):
    data_file = tmp_path_factory.mktemp("data") / "data.npz"
    ex = "examples/bottleneck/model.py"
    subprocess.run(
        f"""
        python -m dinf predict
            --seed 2
            --parallelism 2
            --replicates 10
            --model {ex}
            --discriminator {discriminator_file}
            --output-file {data_file}
        """.split(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert data_file.exists()
    return data_file


@pytest.mark.parametrize("sep", ("\t", " ", ","))
@pytest.mark.usefixtures("tmp_path")
@pytest.mark.usefixtures("discriminator_file")
def test_tabulate_metrics(tmp_path, discriminator_file, sep):
    output_file = tmp_path / "output.txt"
    subprocess.run(
        [
            "python",
            "-m",
            "dinf.tabulate",
            "metrics",
            "--output-file",
            str(output_file),
            "--separator",
            sep,
            discriminator_file,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert output_file.exists()

    metrics = collections.defaultdict(list)
    with open(output_file) as f:
        header = next(f).rstrip().split(sep)
        for line in f:
            for k, v in zip(header, line.rstrip().split(sep)):
                v = int(v) if k == "epoch" else float(v)
                metrics[k].append(v)

    for k in ("epoch", "test_accuracy", "test_loss", "train_accuracy", "train_loss"):
        assert k in metrics
        assert len(metrics[k]) == 10


@pytest.mark.parametrize("sep", ("\t", " ", ","))
@pytest.mark.usefixtures("tmp_path")
@pytest.mark.usefixtures("data_file")
def test_tabulate_data(tmp_path, data_file, sep):
    output_file = tmp_path / "output.txt"
    subprocess.run(
        [
            "python",
            "-m",
            "dinf.tabulate",
            "data",
            "--output-file",
            str(output_file),
            "--separator",
            sep,
            data_file,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert output_file.exists()

    data = collections.defaultdict(list)
    with open(output_file) as f:
        header = next(f).rstrip().split(sep)
        for line in f:
            for k, v in zip(header, line.rstrip().split(sep)):
                v = float(v)
                data[k].append(v)

    for k in ("_Pr", "N0", "N1"):
        assert k in data
        assert len(data[k]) == 10


@pytest.mark.parametrize("sep", ("\t", " ", ","))
@pytest.mark.usefixtures("tmp_path")
@pytest.mark.usefixtures("data_file")
def test_tabulate_quantiles(tmp_path, data_file, sep):
    output_file = tmp_path / "output.txt"
    subprocess.run(
        [
            "python",
            "-m",
            "dinf.tabulate",
            "quantiles",
            "--output-file",
            str(output_file),
            "--separator",
            sep,
            data_file,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert output_file.exists()

    data = collections.defaultdict(list)
    with open(output_file) as f:
        header = next(f).rstrip().split(sep)
        for line in f:
            for k, v in zip(header, line.rstrip().split(sep)):
                v = v if k == "Param" else float(v)
                data[k].append(v)

    quantiles = [0.025, 0.5, 0.975]
    for k in ("Param", *map(str, quantiles)):
        assert k in data
        assert len(data[k]) == 2

    assert data["Param"] == ["N0", "N1"]
