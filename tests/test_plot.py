"""
These test just check that plots are created, not that they look correct.
TODO: Use matplotlib.testing.decorators.image_comparison() to compare
      output to a baseline image. This means the plotted data would need
      to be useful enough to visually assess the baseline images whenever
      the plotting code is altered.
"""
from unittest import mock

import pytest

import dinf.plot
from tests import capture
from .test_cli import HelpMixin


class TestPlotTopLevel:
    def test_help(self):
        with capture() as cap1:
            dinf.plot.main(["-h"])
        assert cap1.ret == 0
        assert "demes" in cap1.out
        assert "features" in cap1.out
        assert "metrics" in cap1.out
        assert "hist" in cap1.out
        assert "hist2d" in cap1.out
        assert "gan" in cap1.out

        with capture() as cap2:
            dinf.plot.main(["--help"])
        assert cap2.ret == 0
        assert cap2.out == cap1.out

        # No args should also output the help.
        with capture() as cap3:
            dinf.plot.main([])
        assert cap3.ret != 0
        assert cap3.out == cap1.out

    def test_version(self):
        with capture() as cap1:
            dinf.plot.main(["-V"])
        assert cap1.ret == 0
        assert cap1.out.strip() == dinf.__version__

        with capture() as cap2:
            dinf.plot.main(["--version"])
        assert cap2.ret == 0
        assert cap2.out.strip() == dinf.__version__


class TestPlotDemes(HelpMixin):
    main = dinf.plot.main
    subcommand = "demes"

    @pytest.mark.usefixtures("tmp_path")
    def test_plot_demes(self, tmp_path):
        ex = "examples/bottleneck/model.py"
        output_file = tmp_path / "output.pdf"
        with capture() as cap:
            dinf.plot.main(
                f"""
                demes
                    --output-file {output_file}
                    --model {ex}
                """.split()
            )
        assert cap.ret == 0
        assert output_file.exists()

    def test_interactive(self):
        ex = "examples/bottleneck/model.py"
        with mock.patch("matplotlib.pyplot.show", autospec=True) as mocked_plt_show:
            with capture() as cap:
                dinf.plot.main(f"demes --model {ex}".split())
        assert cap.ret == 0
        mocked_plt_show.assert_called_once()


class TestPlotFeatures(HelpMixin):
    main = dinf.plot.main
    subcommand = "features"

    @pytest.mark.parametrize(
        "model_file",
        ["examples/bottleneck/model.py", "examples/isolation_with_migration/model.py"],
    )
    @pytest.mark.parametrize("target_option", ["", "--target"])
    @pytest.mark.usefixtures("tmp_path")
    def test_plot_features(self, tmp_path, target_option, model_file):
        output_file = tmp_path / "output.pdf"
        with capture() as cap:
            dinf.plot.main(
                f"""
                features
                    --output-file {output_file}
                    {target_option}
                    --model {model_file}
                """.split()
            )
        assert cap.ret == 0
        assert output_file.exists()

    def test_interactive(self):
        ex = "examples/bottleneck/model.py"
        with mock.patch("matplotlib.pyplot.show", autospec=True) as mocked_plt_show:
            with capture() as cap:
                dinf.plot.main(f"features --model {ex}".split())
        assert cap.ret == 0
        mocked_plt_show.assert_called_once()


class TestPlotMetrics(HelpMixin):
    main = dinf.plot.main
    subcommand = "metrics"

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("discriminator_file")
    def test_plot_metrics(self, tmp_path, discriminator_file):
        output_file = tmp_path / "output.pdf"
        with capture() as cap:
            dinf.plot.main(
                f"""
                metrics
                    --output-file {output_file}
                    {discriminator_file}
                """.split()
            )
        assert cap.ret == 0
        assert output_file.exists()

    @pytest.mark.usefixtures("discriminator_file")
    def test_interactive(self, discriminator_file):
        with mock.patch("matplotlib.pyplot.show", autospec=True) as mocked_plt_show:
            with capture() as cap:
                dinf.plot.main(f"metrics {discriminator_file}".split())
        assert cap.ret == 0
        mocked_plt_show.assert_called_once()


class TestPlotHist(HelpMixin):
    main = dinf.plot.main
    subcommand = "hist"

    @pytest.mark.parametrize(
        "model_option", ["", "--model examples/bottleneck/model.py"]
    )
    @pytest.mark.parametrize(
        "extra_options",
        [
            "",
            "--x-param N0",
            "--top 5",
            "--weighted",
            "--weighted --top 5",
            "--cumulative",
            "--kde",
            "--weighted --cumulative --kde",
        ],
    )
    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_plot_hist(self, tmp_path, data_file, model_option, extra_options):
        output_file = tmp_path / "output.pdf"
        with capture() as cap:
            dinf.plot.main(
                f"""
                hist
                    --output-file {output_file}
                    {model_option}
                    {extra_options}
                    {data_file}
                """.split()
            )
        assert cap.ret == 0
        assert output_file.exists()

    @pytest.mark.usefixtures("data_file")
    def test_interactive(self, data_file):
        with mock.patch("matplotlib.pyplot.show", autospec=True) as mocked_plt_show:
            with capture() as cap:
                dinf.plot.main(f"hist {data_file}".split())
        assert cap.ret == 0
        assert mocked_plt_show.call_count == 3

        with mock.patch("matplotlib.pyplot.show", autospec=True) as mocked_plt_show:
            with capture() as cap:
                dinf.plot.main(f"hist --x-param _Pr {data_file}".split())
        assert cap.ret == 0
        assert mocked_plt_show.call_count == 1

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_duplicate_x_param(self, tmp_path, data_file):
        output_file = tmp_path / "output.pdf"
        with pytest.raises(ValueError, match="--x-param values are not unique"):
            dinf.plot.main(
                f"""
                hist
                    --output-file {output_file}
                    --x-param N0
                    --x-param N0
                    {data_file}
                """.split()
            )

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_unknown_x_param(self, tmp_path, data_file):
        output_file = tmp_path / "output.pdf"
        with pytest.raises(ValueError, match="npz: parameter `nonexistent' not found"):
            dinf.plot.main(
                f"""
                hist
                    --output-file {output_file}
                    --x-param nonexistent
                    {data_file}
                """.split()
            )

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_mismatched_parameters_and_model(self, tmp_path, data_file):
        output_file = tmp_path / "output.pdf"
        with pytest.raises(ValueError, match="model.py: couldn't find `N0'"):
            dinf.plot.main(
                f"""
                hist
                    --output-file {output_file}
                    --x-param N0
                    --model examples/isolation_with_migration/model.py
                    {data_file}
                """.split()
            )


class TestPlotHist2d(HelpMixin):
    main = dinf.plot.main
    subcommand = "hist2d"

    @pytest.mark.parametrize(
        "model_option", ["", "--model examples/bottleneck/model.py"]
    )
    @pytest.mark.parametrize(
        "extra_options",
        [
            "",
            "--top 5",
            "--weighted",
            "--weighted --top 5",
            "--x-param N0 --y-param N1",
            "--x-param N1 --y-param N0",
        ],
    )
    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_plot_hist2d(self, tmp_path, data_file, model_option, extra_options):
        output_file = tmp_path / "output.pdf"
        with capture() as cap:
            dinf.plot.main(
                f"""
                hist2d
                    --output-file {output_file}
                    {model_option}
                    {extra_options}
                    {data_file}
                """.split()
            )
        assert cap.ret == 0
        assert output_file.exists()

    @pytest.mark.usefixtures("data_file")
    def test_interactive(self, data_file):
        with mock.patch("matplotlib.pyplot.show", autospec=True) as mocked_plt_show:
            with capture() as cap:
                dinf.plot.main(f"hist2d {data_file}".split())
        assert cap.ret == 0
        assert mocked_plt_show.call_count == 1

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_duplicate_x_param(self, tmp_path, data_file):
        output_file = tmp_path / "output.pdf"
        with pytest.raises(ValueError, match="--x-param values are not unique"):
            dinf.plot.main(
                f"""
                hist2d
                    --output-file {output_file}
                    --x-param N0
                    --x-param N0
                    {data_file}
                """.split()
            )

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_duplicate_y_param(self, tmp_path, data_file):
        output_file = tmp_path / "output.pdf"
        with pytest.raises(ValueError, match="--y-param values are not unique"):
            dinf.plot.main(
                f"""
                hist2d
                    --output-file {output_file}
                    --y-param N1
                    --y-param N1
                    {data_file}
                """.split()
            )

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_unknown_param(self, tmp_path, data_file):
        output_file = tmp_path / "output.pdf"
        with pytest.raises(ValueError, match="npz: parameter `nonexistent' not found"):
            dinf.plot.main(
                f"""
                hist2d
                    --output-file {output_file}
                    --x-param nonexistent
                    {data_file}
                """.split()
            )
        with pytest.raises(ValueError, match="npz: parameter `nonexistent' not found"):
            dinf.plot.main(
                f"""
                hist2d
                    --output-file {output_file}
                    --y-param nonexistent
                    {data_file}
                """.split()
            )

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("data_file")
    def test_mismatched_parameters_and_model(self, tmp_path, data_file):
        output_file = tmp_path / "output.pdf"
        with pytest.raises(ValueError, match="model.py: expected parameter `N0'"):
            dinf.plot.main(
                f"""
                hist2d
                    --output-file {output_file}
                    --x-param N0
                    --y-param N1
                    --model examples/isolation_with_migration/model.py
                    {data_file}
                """.split()
            )
        with pytest.raises(ValueError, match="model.py: expected parameter `N0'"):
            dinf.plot.main(
                f"""
                hist2d
                    --output-file {output_file}
                    --x-param N1
                    --y-param N0
                    --model examples/isolation_with_migration/model.py
                    {data_file}
                """.split()
            )


class TestPlotGan(HelpMixin):
    main = dinf.plot.main
    subcommand = "gan"

    @pytest.mark.parametrize(
        "model_option", ["", "--model examples/bottleneck/model.py"]
    )
    @pytest.mark.parametrize(
        "extra_options",
        [
            "",
            "--top 5",
            "--weighted",
            "--weighted --top 5",
        ],
    )
    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.usefixtures("smc_outdir")
    def test_plot_gan(self, tmp_path, smc_outdir, model_option, extra_options):
        output_file = tmp_path / "output.png"
        with capture() as cap:
            dinf.plot.main(
                f"""
                gan
                    --output-file {output_file}
                    {model_option}
                    {extra_options}
                    {smc_outdir}
                """.split()
            )
        assert cap.ret == 0
        assert (tmp_path / "output__Pr.png").exists()
        assert (tmp_path / "output_N0.png").exists()
        assert (tmp_path / "output_N1.png").exists()

    @pytest.mark.usefixtures("smc_outdir")
    def test_interactive(self, smc_outdir):
        with mock.patch("matplotlib.pyplot.show", autospec=True) as mocked_plt_show:
            with capture() as cap:
                dinf.plot.main(f"gan {smc_outdir}".split())
        assert cap.ret == 0
        assert mocked_plt_show.call_count == 5  # metrics, entropy, _Pr, N0, N1
