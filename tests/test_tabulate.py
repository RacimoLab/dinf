import collections

import pytest

import dinf.tabulate
from tests import capture


@pytest.mark.parametrize("sep", ("\t", " ", ","))
@pytest.mark.usefixtures("tmp_path")
@pytest.mark.usefixtures("discriminator_file")
def test_tabulate_metrics(tmp_path, discriminator_file, sep):
    output_file = tmp_path / "output.txt"
    with capture() as cap:
        dinf.tabulate.main(
            [
                "metrics",
                "--output-file",
                str(output_file),
                "--separator",
                sep,
                str(discriminator_file),
            ]
        )
    assert cap.ret == 0
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
    with capture() as cap:
        dinf.tabulate.main(
            [
                "data",
                "--output-file",
                str(output_file),
                "--separator",
                sep,
                str(data_file),
            ]
        )
    assert cap.ret == 0
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
    with capture() as cap:
        dinf.tabulate.main(
            [
                "quantiles",
                "--output-file",
                str(output_file),
                "--separator",
                sep,
                str(data_file),
            ]
        )
    assert cap.ret == 0
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
