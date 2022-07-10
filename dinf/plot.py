from __future__ import annotations
import argparse
import contextlib
import pathlib
import textwrap
from typing import Dict

from adjustText import adjust_text
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .cli import ADRDFormatter, _GENOB_MODEL_HELP
import dinf


def feature(
    mat: np.ndarray,
    /,
    *,
    ax: matplotlib.axes.Axes,
    channel: int = 0,
    cmap: str | matplotlib.colors.Colourmap | None = None,
    cb: bool = True,
    cb_label: str | None = None,
    vmax: float | None = None,
):
    """
    Plot a feature matrix as a heatmap.

    :param mat:
        The feature matrix.
    :param ax:
        Matplotlib axes onto which the plot will be drawn.
    :param channel:
        Which channel of the feature matrix to draw.
        I.e., the index into the last dimension of ``mat``.
    :param cmap:
        Matplotlib colour map.
    :param cb:
        If True, add a colour bar that indicates how colours relate to the
        values in the feature matrix.
    :param cb_label:
        Label on the colour bar.
    """
    assert not isinstance(mat, dict)
    if cmap is None:
        if channel == 0:
            cmap = matplotlib.cm.get_cmap("viridis", int(1 + np.max(mat)))
        else:
            cmap = "plasma"
    im = ax.imshow(
        mat[..., channel],
        interpolation="none",
        origin="lower",
        rasterized=True,
        # left, right, bottom, top
        extent=(0, mat.shape[1], 0, mat.shape[0]),
        aspect="auto",
        cmap=cmap,
        vmax=vmax,
    )
    ax.set_ylabel("individuals")
    ax.set_xlabel("loci")
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)

    if cb:
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.04, label=cb_label)
        # Prefer integer ticks.
        cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def features(mats: Dict[str, np.ndarray], /, *, subplots_kw: dict | None = None):
    """
    Plot multiple feature matrices as heatmaps.

    :param mats:
        The multiple feature matrices.
    :param subplots_kw:
        Additional keyword arguments that will be passed to
        :func:`matplotlib.pyplot.subplots`.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    :return:
        A (figure, axes) tuple.
    """
    assert isinstance(mats, dict)
    if subplots_kw is None:
        subplots_kw = {}

    num_mats = len(mats)
    max_channels = max(mat.shape[-1] for mat in mats.values())
    if num_mats == 1:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=max_channels,
            **subplots_kw,
        )
    else:
        fig, axs = plt.subplots(
            nrows=max_channels,
            ncols=num_mats,
            sharex="col",
            **subplots_kw,
        )
    axv = np.atleast_2d(axs)
    fig.set_constrained_layout(True)

    vmaxs = {}
    cmaps = {}
    for j in range(max_channels):
        vmaxs[j] = max(
            np.max(mat[..., j]) for mat in mats.values() if j < mat.shape[-1]
        )
        if j == 0:
            cmaps[j] = matplotlib.cm.get_cmap("viridis", int(1 + vmaxs[j]))
        else:
            cmaps[j] = matplotlib.cm.get_cmap("plasma")

    if num_mats == 1:
        mat = list(mats.values())[0]
        for j in range(max_channels):
            ax = axv[0, j]
            if j == 0:
                cb_label = "Minor alleles"
            else:
                cb_label = None
            feature(
                mat,
                ax=ax,
                channel=j,
                cb_label=cb_label,
                cmap=cmaps[j],
                vmax=vmaxs[j],
            )
            if max_channels > 1:
                ax.set_title(f"channel {j}")
    else:
        for i, (label, mat) in enumerate(mats.items()):
            num_channels = mat.shape[-1]
            for j in range(max_channels):
                ax = axv[j, i]
                if j >= num_channels:
                    ax.set_axis_off()
                    continue

                if j == 0:
                    cb_label = "Minor alleles"
                else:
                    cb_label = None
                feature(
                    mat,
                    ax=ax,
                    channel=j,
                    cb_label=cb_label,
                    cmap=cmaps[j],
                    vmax=vmaxs[j],
                    cb=i == num_mats - 1,
                )

                title = label
                if num_channels > 1:
                    title = f"{label} / channel {j}"
                    if j < num_channels - 1:
                        ax.set_xlabel("")
                if num_mats > 1 or num_channels > 1:
                    ax.set_title(title)
                if i > 0:
                    ax.set_ylabel("")

    for ax in axv.flat:
        # Get the ticklabels back for the shared axis.
        ax.xaxis.set_tick_params(labelbottom=True)
        # Prefer integer ticks.
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    return fig, axs


def metrics(
    *,
    networks: Dict[str, dinf.Discriminator],
    legend_title: str | None = None,
    subplot_mosaic_kw: dict | None = None,
):
    """
    Plot training metrics for a discriminator neural network.

    :param networks:
        A dictionary mapping labels to Discriminator objects.
        The labels will be used in the legend to identify each
        discriminator.
    :param legend_title:
        A title for the legend.
    :param subplot_mosaic_kw:
        Additional keyword arguments that will be passed to
        :func:`matplotlib.pyplot.subplot_mosaic`.
    :rtype: tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
    :return:
        A (figure, axes_dict) tuple, where axes_dict is a dictionary mapping
        metrics to the :class:`matplotlib.axes.Axes` objects.
        (As obtained from :func:`matplotlib.pyplot.subplot_mosaic`).
    """
    if subplot_mosaic_kw is None:
        subplot_mosaic_kw = {}
    kw = dict(sharex=True, constrained_layout=True)
    kw.update(subplot_mosaic_kw)
    if kw.get("tight_layout", False):
        del kw["constrained_layout"]
    fig, axs = plt.subplot_mosaic(
        [
            ["train_loss", "test_loss"],
            ["train_accuracy", "test_accuracy"],
        ],
        **kw,
    )
    axs["train_loss"].get_shared_y_axes().join(axs["train_loss"], axs["test_loss"])
    axs["train_accuracy"].get_shared_y_axes().join(
        axs["train_accuracy"], axs["test_accuracy"]
    )

    metrics = ("train_loss", "test_loss", "train_accuracy", "test_accuracy")
    for label, network in networks.items():
        assert network.trained
        assert network.train_metrics is not None
        for metric in metrics:
            y = network.train_metrics[metric]
            epoch = range(1, len(y) + 1)
            axs[metric].plot(epoch, y, label=label)
            axs[metric].set_title(metric.replace("_", " "))

    axs["train_loss"].set_ylabel("loss")
    axs["train_accuracy"].set_ylabel("accuracy")
    axs["train_accuracy"].set_xlabel("epoch")
    axs["test_accuracy"].set_xlabel("epoch")

    if len(networks) > 1:
        handles, labels = axs["train_loss"].get_legend_handles_labels()
        # Put legend to the right of the test loss.
        axs["test_loss"].legend(
            handles,
            labels,
            title=legend_title,
            loc="upper left",
            borderaxespad=0.0,
            bbox_to_anchor=(1.05, 1),
        )

    return fig, axs


def hist2d(
    x: np.ndarray,
    y: np.ndarray,
    /,
    *,
    x_label: float | None = None,
    y_label: float | None = None,
    x_truth: float | None = None,
    y_truth: float | None = None,
    ax: matplotlib.axes.Axis | None = None,
    hist2d_kw: dict | None = None,
):
    """
    Plot a 2d histogram.

    :param x:
        Values for the horizontal axis.
    :param y:
        Values for the vertical axis.
    :param x_label:
        Name of the parameter on the horizontal axis.
    :param y_label:
        Name of the parameter on the vertical axis.
    :param x_truth:
        True value of the parameter on the horizontal axis.
    :param y_truth:
        True value of the parameter on the vertical axis.
    :param ax:
        Axes onto which the histogram will be drawn.
        If None, an axes will be created with :func:`matplotlib.pyplot.subplots`.
    :param hist2d_kw:
        Further parameters passed to :func:`matplotlib.axes.Axes.hist2d`.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    :return:
        A (figure, axes) tuple.
    """
    assert x.shape == y.shape
    if ax is None:
        fig, ax = plt.subplots()
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)

    if hist2d_kw is None:
        hist2d_kw = {}
    kw = dict(bins=50, cmap="binary", density=True, rasterized=True)
    kw.update(hist2d_kw)

    _, _, _, h = ax.hist2d(x, y, **kw)
    ax.figure.colorbar(h, ax=ax)

    colour = "red"
    line_kw = dict(c=colour, ls="-")
    if x_truth is not None:
        ax.axvline(x_truth, **line_kw)
    if y_truth is not None:
        ax.axhline(y_truth, **line_kw)

    if x_truth is not None and y_truth is not None:
        ax.scatter(
            x_truth,
            y_truth,
            marker="s",
            c=colour,
            label="truth",
        )
    if x_truth is not None or y_truth is not None:
        ax.legend()

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    return ax.figure, ax


def _quantile(x: np.ndarray, /, *, q, weights=None) -> np.ndarray:
    """
    Calculate quantiles of an array.

    :param x:
        Array values.
    :param q:
        Quantiles to calculate.
    :param weights:
        Weights for the array values in ``x``.
    :return:
        The ``q``'th quantiles of ``x``.
    """
    if weights is None:
        return np.quantile(x, q)
    idx = np.argsort(x)
    x = x[idx]
    weights = weights[idx]
    S = np.cumsum(weights)
    wxq = (S - weights / 2) / S[-1]
    return np.interp(q, wxq, x)


def _num2str(n):
    """Reduce precision for large numbers."""
    # XXX: is this a bad idea?
    if n > 20:
        n = round(n, 2)
    return format(n, "g")


def hist(
    x: np.ndarray,
    /,
    *,
    ax: matplotlib.axes.Axes | None = None,
    ci: bool = False,
    truth: float | None = None,
    hist_kw: dict | None = None,
):
    """
    Plot a histogram.

    :param x:
        Array of values for the histogram.
    :param ax:
        Axes onto which the histogram will be drawn.
        If None, an axes will be created with :func:`matplotlib.pyplot.subplots`.
    :param ci:
        If True, draw a 95% credible interval at the bottom.
    :param truth:
        If not None, draw a red vertical line at this location.
    :param hist_kw:
        Further parameters passed to :func:`matplotlib.axes.Axes.hist`.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    :return:
        A (figure, axes) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if hist_kw is None:
        hist_kw = {}
    kw = dict(bins=50, density=True, histtype="step")
    kw.update(hist_kw)

    _, _, patches = ax.hist(x, **kw)
    if kw["histtype"] == "step":
        # Remove vertical lines at left and right edges.
        patches[0].set_xy(patches[0].get_xy()[1:-1])

    if truth is not None:
        ax.axvline(truth, c="red", ls="-")

    if ci:
        # Get median and 95% credible interval.
        q = [0.025, 0.5, 0.975]
        xq = _quantile(x, q=q, weights=kw.get("weights"))

        # Show interval as a horizontal line with whiskers.
        ylim = ax.get_ylim()
        ys = (ylim[0] - ylim[1]) * np.array([0.03, 0.03, 0.04, 0.05, 0.06])
        line_kw = dict(colors="k", lw=3, zorder=5)
        ax.hlines(ys[2], xq[0], xq[2], **line_kw)
        ax.vlines(xq, ys[3], ys[1], **line_kw)

        # Annotate with the median and CI values.
        txt_kw = dict(
            ha="center",
            va="center",
            zorder=10,
            bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.6),
        )
        texts = []
        for j in range(len(xq)):
            t1 = ax.text(xq[j], ys[0], f"{100*q[j]:g}%", **txt_kw)
            t2 = ax.text(xq[j], ys[4], _num2str(xq[j]), **txt_kw)
            texts.extend([t1, t2])
        # Fix text items from overlapping. Maybe!
        adjust_text(
            texts,
            ax=ax,
            only_move=dict(points="", objects="", text="x"),
            avoid_points=False,
            avoid_self=False,
        )

    return ax.figure, ax


class _SubCommand:
    """
    Base class for subcommands.
    """

    def __init__(self, subparsers, command):
        docstring = textwrap.dedent(self.__doc__)
        self.parser = subparsers.add_parser(
            command,
            help=docstring.lstrip().splitlines()[0],
            description=docstring,
            formatter_class=ADRDFormatter,
        )
        self.parser.set_defaults(func=self)

    def add_argument_output_file(self):
        self.parser.add_argument(
            "-o",
            "--output-file",
            type=str,
            metavar="output.pdf",
            help=(
                "Output file for the figure. "
                "If no output file is specified, an interactive plot window "
                "will be opened."
            ),
        )

    def add_argument_seed(self):
        self.parser.add_argument(
            "-S",
            "--seed",
            type=int,
            help="Seed for the random number generator",
        )

    def add_argument_abc_thresholds(self):
        group = self.parser.add_mutually_exclusive_group()
        group.add_argument(
            "-n",
            "--top-n",
            metavar="N",
            type=int,
            help="Accept only the N top parameter values, ranked by log probability.",
        )
        group.add_argument(
            "-t",
            "--probability-threshold",
            metavar="P",
            type=float,
            help="Accept only the parameter values with probabilities greater than P.",
        )

    def add_argument_weighted(self):
        self.parser.add_argument(
            "-W",
            "--weighted",
            action="store_true",
            help="Weight the parameter values by probability.",
        )

    def add_argument_data_file(self):
        self.parser.add_argument(
            "data_file",
            metavar="data.npz",
            type=pathlib.Path,
            help="The datafile containing discriminator predictions.",
        )

    def add_argument_discriminators(self):
        self.parser.add_argument(
            "discriminators",
            metavar="discriminator.pkl",
            nargs="+",
            help="The discriminator network(s) to plot.",
        )

    def add_argument_genob_model(self):
        self.parser.add_argument(
            "genob_model",
            metavar="model.py",
            type=pathlib.Path,
            help=_GENOB_MODEL_HELP,
        )

    def add_argument_genob_model_optional(self):
        self.parser.add_argument(
            "-m",
            "--model",
            metavar="model.py",
            type=pathlib.Path,
            help="Model file from which the parameter truth values with be taken.",
        )

    def get_abc_dataset(self, args):
        """Load data file and filter by top-n or by probability."""
        data = dinf.load_results(args.data_file)
        shape = data["_Pr"].shape
        if len(shape) != 1:
            # Bail for MCMC-GAN datasets.
            raise ValueError(f"I don't understand GAN datasets (shape={shape}).")

        assert None in (args.top_n, args.probability_threshold)
        if args.top_n is not None:
            idx = np.flip(np.argsort(data["_Pr"]))
            data = data[idx[: args.top_n]]
        elif args.probability_threshold is not None:
            data = data[np.where(data["_Pr"] > args.probability_threshold)]

        return data


class _Features(_SubCommand):
    """
    Plot a feature matrix or matrices as heatmaps.

    By default, one simulation will be performed with the generator to obtain
    a set of features for plotting. To instead extract features from the
    target dataset, use the --target option.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "features")
        self.add_argument_output_file()
        self.add_argument_seed()
        self.parser.add_argument(
            "--target",
            action="store_true",
            help="Extract feature(s) from the target dataset.",
        )
        self.add_argument_genob_model()

    def __call__(self, args: argparse.Namespace):
        genobuilder = dinf.Genobuilder.from_file(args.genob_model)

        if args.target:
            assert genobuilder.target_func is not None
            mats = genobuilder.target_func(args.seed)
        else:
            rng = np.random.default_rng(args.seed)
            thetas = genobuilder.parameters.draw_prior(1, rng=rng)
            mats = genobuilder.generator_func(
                (rng.integers(low=0, high=2**31), thetas[0])
            )

        if not isinstance(mats, dict):
            mats_dict = {"": mats}
        else:
            mats_dict = mats

        fig, axs = features(mats_dict, subplots_kw=dict(figsize=plt.figaspect(9 / 16)))
        if args.output_file is None:
            plt.show()
        else:
            fig.savefig(args.output_file)


class _Metrics(_SubCommand):
    """
    Plot loss and accuracy of discriminator(s).

    Each metric is plotted as a function of the training epoch,
    and the resulting multipanel plot shows:
      - training loss,
      - training accuracy,
      - test loss, and
      - test accuracy.

    If multiple discriminator files are provided, the training metrics for
    each file are indicated by a different colour. The legend shows the
    corresponding filename.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "metrics")
        self.add_argument_output_file()
        self.add_argument_discriminators()

    def __call__(self, args: argparse.Namespace):
        discriminators = {
            pathlib.Path(d).name: dinf.Discriminator.from_file(d)
            for d in args.discriminators
        }
        fig, axs = metrics(
            networks=discriminators,
            subplot_mosaic_kw=dict(figsize=plt.figaspect(9 / 16)),
        )
        if args.output_file is None:
            plt.show()
        else:
            fig.savefig(args.output_file)


class _Hist2d(_SubCommand):
    """
    Plot 2d marginal posterior densities.

    One plot is produced for each unique pair of parameters.
    As this may lead to a large number of plots (particularly
    for interactive use!), the choice of which parameters to
    plot can be specified using the -x and -y options.

    The resulting figure is a 2d histogram, with darker squares
    indicating higher densities. If the data correspond to a
    simulation-only model, then the parameters' truth values will
    be indicated by red lines. By default, all values in the
    data file contribute equally to the histogram. For parameter
    values drawn from the prior distribution, this will therefore
    show the prior distribution. A more informative plot can be
    obtained by weighting parameter values by the discriminator
    probabilities using the -W option. Alternately, the data file
    can be filtered to obtain a posterior sample using the -n
    or -t options.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "hist2d")
        self.add_argument_output_file()
        self.add_argument_abc_thresholds()
        self.add_argument_weighted()
        self.parser.add_argument(
            "-x",
            "--x-param",
            type=str,
            action="append",
            help="Name of parameter to plot on horizontal axis.",
        )
        self.parser.add_argument(
            "-y",
            "--y-param",
            type=str,
            action="append",
            help="Name of parameter to plot on vertical axis.",
        )
        self.add_argument_genob_model_optional()
        self.add_argument_data_file()

    def __call__(self, args: argparse.Namespace):
        parameters = None
        if args.model is not None:
            parameters = dinf.Genobuilder.from_file(args.model).parameters
        data = self.get_abc_dataset(args)
        param_names = data.dtype.names[1:]

        if args.x_param is None:
            args.x_param = param_names
        if len(set(args.x_param)) != len(args.x_param):
            raise ValueError(f"--x-param values are not unique: {args.x_param}")
        if args.y_param is None:
            args.y_param = param_names
        if len(set(args.y_param)) != len(args.y_param):
            raise ValueError(f"--y-param values are not unique: {args.y_param}")

        for param in args.x_param + args.y_param:
            if param not in param_names:
                raise ValueError(f"{args.data_file}: parameter {param} not found")

        if args.output_file:
            cm = lambda: PdfPages(args.output_file)  # noqa: E731
        else:
            # Do-nothing context manager. Yields None in an 'as' statement.
            cm = contextlib.nullcontext

        hist2d_kw = {}
        if args.weighted:
            hist2d_kw["weights"] = data["_Pr"]

        done = set()
        with cm() as pdf:
            for x_param in args.x_param:
                x = data[x_param]
                x_truth = None
                if parameters is not None:
                    if x_param not in parameters:
                        raise ValueError(
                            f"{args.model}: expected parameter `{x_param}'"
                        )
                    x_truth = parameters[x_param].truth
                for y_param in args.y_param:
                    if x_param == y_param or (y_param, x_param) in done:
                        continue
                    done.add((x_param, y_param))

                    y = data[y_param]
                    y_truth = None
                    if parameters is not None:
                        if y_param not in parameters:
                            raise ValueError(
                                f"{args.model}: expected parameter `{y_param}'"
                            )
                        y_truth = parameters[y_param].truth

                    fig, ax = plt.subplots(
                        figsize=plt.figaspect(9 / 16), constrained_layout=True
                    )
                    hist2d(
                        x,
                        y,
                        ax=ax,
                        x_label=x_param,
                        y_label=y_param,
                        x_truth=x_truth,
                        y_truth=y_truth,
                        hist2d_kw=hist2d_kw,
                    )
                    if pdf is None:
                        plt.show()
                    else:
                        pdf.savefig(fig)
                    plt.close(fig)


class _Hist(_SubCommand):
    """
    Plot marginal posterior densities.

    One plot is produced for the discriminator probabilities,
    plus one plot for each model parameter. The choice of which
    parameter to plot can be specified using the -x option,
    with the special value "_Pr" indicating the discriminator
    probabilities.

    The resulting figure is a histogram. If the data correspond
    to a simulation-only model (provided via the -m option),
    then the parameter's truth value will be shown as a vertical
    red line. A 95% credible interval is shown at the bottom of
    the figure. By default, all values in the data file contribute
    equally to the histogram. For parameter values drawn from the
    prior distribution, this will therefore show the prior
    distribution. A more informative plot can be obtained by
    weighting parameter values by the discriminator probabilities
    using the -W option. Alternately, the data file can be filtered
    to obtain a posterior sample using the -n or -t options.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "hist")

        self.add_argument_output_file()
        self.add_argument_abc_thresholds()
        self.add_argument_weighted()
        self.parser.add_argument(
            "-c",
            "--cumulative",
            action="store_true",
            help="Plot cumulative distribution.",
        )
        self.parser.add_argument(
            "-x",
            "--x-param",
            type=str,
            action="append",
            help=(
                "Name of parameter to plot. "
                'The special name "_Pr" is recognised to plot the probabilities '
                "obtained from the discriminator."
            ),
        )
        self.add_argument_genob_model_optional()
        self.add_argument_data_file()

    def __call__(self, args: argparse.Namespace):
        parameters = None
        if args.model is not None:
            parameters = dinf.Genobuilder.from_file(args.model).parameters
        data = self.get_abc_dataset(args)

        if args.x_param is None:
            args.x_param = data.dtype.names
        if len(set(args.x_param)) != len(args.x_param):
            raise ValueError(f"--x-param values are not unique: {args.x_param}")

        for param in args.x_param:
            if param not in data.dtype.names:
                raise ValueError(f"{args.data_file}: parameter {param} not found")

        if args.output_file:
            cm = lambda: PdfPages(args.output_file)  # noqa: E731
        else:
            # Do-nothing context manager. Yields None in an 'as' statement.
            cm = contextlib.nullcontext

        with cm() as pdf:
            for x_param in args.x_param:
                fig, ax = plt.subplots(
                    figsize=plt.figaspect(9 / 16), constrained_layout=True
                )
                truth = None
                hist_kw = dict(cumulative=args.cumulative)
                if x_param == "_Pr":
                    # Plot discriminator probabilities.
                    ax.set_xlabel("Pr")
                    hist_kw["log"] = True
                    ci = False
                else:
                    # Plot parameter.
                    if parameters is not None:
                        px = parameters.get(x_param)
                        if px is None:
                            raise ValueError(f"{args.model}: couldn't find `{x_param}`")
                        truth = px.truth
                    x = data[x_param]
                    ax.set_xlabel(x_param)
                    if args.weighted:
                        hist_kw["weights"] = data["_Pr"]
                    ci = True

                x = data[x_param]
                hist(x, ax=ax, ci=ci, truth=truth, hist_kw=hist_kw)

                if args.cumulative:
                    ax.set_ylabel("cumulative density")
                else:
                    ax.set_ylabel("density")

                if args.output_file is None:
                    plt.show()
                else:
                    pdf.savefig(fig)
                plt.close(fig)


def main(args_list=None):
    top_parser = argparse.ArgumentParser(
        prog="dinf.plot", description="Dinf plotting tools."
    )

    subparsers = top_parser.add_subparsers(dest="subcommand")
    _Features(subparsers)
    _Metrics(subparsers)
    _Hist(subparsers)
    _Hist2d(subparsers)

    args = top_parser.parse_args(args_list)
    if args.subcommand is None:
        top_parser.print_help()
        exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
