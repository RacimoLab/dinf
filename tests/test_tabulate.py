import collections

import pytest

import dinf.tabulate
from tests import capture
from .test_cli import HelpMixin


class TestTabulateTopLevel:
    def test_help(self):
        with capture() as cap1:
            dinf.tabulate.main(["-h"])
        assert cap1.ret == 0
        assert "metrics" in cap1.out
        assert "data" in cap1.out
        assert "quantiles" in cap1.out

        with capture() as cap2:
            dinf.tabulate.main(["--help"])
        assert cap2.ret == 0
        assert cap2.out == cap1.out

        # No args should also output the help.
        with capture() as cap3:
            dinf.tabulate.main([])
        assert cap3.ret != 0
        assert cap3.out == cap1.out

    def test_version(self):
        with capture() as cap1:
            dinf.tabulate.main(["-V"])
        assert cap1.ret == 0
        assert cap1.out.strip() == dinf.__version__

        with capture() as cap2:
            dinf.tabulate.main(["--version"])
        assert cap2.ret == 0
        assert cap2.out.strip() == dinf.__version__


class TestTabulateMetrics(HelpMixin):
    main = dinf.tabulate.main
    subcommand = "metrics"

    @pytest.mark.parametrize("sep", ("\t", " ", ","))
    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("discriminator_file")
    def test_tabulate_metrics(self, tmp_path, discriminator_file, sep):
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

        for k in (
            "epoch",
            "test_accuracy",
            "test_loss",
            "train_accuracy",
            "train_loss",
        ):
            assert k in metrics
            assert len(metrics[k]) == 10


class TestTabulateData(HelpMixin):
    main = dinf.tabulate.main
    subcommand = "data"

    @pytest.mark.parametrize("sep", ("\t", " ", ","))
    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_tabulate_data(self, tmp_path, data_file, sep):
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


class TestTabulateQuantiles(HelpMixin):
    main = dinf.tabulate.main
    subcommand = "quantiles"

    @pytest.mark.parametrize("sep", ("\t", " ", ","))
    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_tabulate_quantiles(self, tmp_path, data_file, sep):
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
