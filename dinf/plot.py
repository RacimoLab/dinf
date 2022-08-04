from __future__ import annotations
import argparse
import logging
import pathlib
from typing import Any, Dict, Tuple

from adjustText import adjust_text
from cycler import cycler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import scipy.stats

from .cli import _SubCommand, _set_loglevel
from .misc import quantile
import dinf

logger = logging.getLogger(__name__)


class _MultiPage:
    """
    PdfPages-like context manager that also handles non-pdfs.

    For pdf output, PdfPages is used to create a multi-page pdf with
    one page per figure. For non-pdf output, separate files are created
    for each figure, where the filenames are constructed by inserting
    the figure number into the filename.
    """

    def __init__(self, filename, n_figures):
        self.filename = None
        self.pdf = None
        if filename is not None:
            if filename.endswith(".pdf"):
                self.pdf = PdfPages(filename)
            self.filename = pathlib.Path(filename)
        self.n_figures = n_figures

    def __enter__(self):
        if self.pdf is not None:
            self.pdf.__enter__()
        return self

    def __exit__(self, *args):
        if self.pdf is not None:
            return self.pdf.__exit__(*args)

    def savefig(self, fig, *, hint):
        if self.filename is None:
            plt.show()
            return

        if self.pdf is not None:
            logger.debug("%s: saving figure (%s).", self.filename, hint)
            self.pdf.savefig(fig)
        else:
            filename = self.filename
            if self.n_figures > 1:
                # Insert the hint into the filename.
                filename = filename.with_stem(f"{self.filename.stem}_{hint}")
            logger.debug("%s: saving figure (%s).", filename, hint)
            fig.savefig(filename)


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
    metrics_collection: Dict[str, Dict[str, Any]],
    /,
    *,
    legend_title: str | None = None,
    subplot_mosaic_kw: dict | None = None,
):
    """
    Plot training metrics from a discriminator neural network.

    :param metrics_collection:
        A dictionary mapping labels to metrics dictionaries.
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

    mkeys = ("train_loss", "test_loss", "train_accuracy", "test_accuracy")
    for label, metrics in metrics_collection.items():
        for metric in mkeys:
            y = metrics[metric]
            epoch = range(1, len(y) + 1)
            axs[metric].plot(epoch, y, label=label)
            axs[metric].set_title(metric.replace("_", " "))

    axs["train_loss"].set_ylabel("loss")
    axs["train_accuracy"].set_ylabel("accuracy")
    axs["train_accuracy"].set_xlabel("epoch")
    axs["test_accuracy"].set_xlabel("epoch")

    if len(metrics_collection) > 1:
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
    line_kw = dict(c=colour, ls="-", alpha=0.7)
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
        ax.axvline(truth, c="red", ls="-", zorder=3, alpha=0.7)

    if ci:
        # Get median and 95% credible interval.
        q = [0.025, 0.5, 0.975]
        xq = quantile(x, q=q, weights=kw.get("weights"))

        # Show interval as a horizontal line with whiskers.
        # It would be easier to use hlines() and vlines() here,
        # but blended transforms are broken. So we use plot() instead.
        # https://github.com/matplotlib/matplotlib/issues/23171
        ys = 0.1 + np.array([0.0, 0.04, 0.05, 0.06, 0.07])
        line_kw = dict(c="k", lw=3, zorder=3, transform=ax.get_xaxis_transform())
        # hline
        ax.plot([xq[0], xq[2]], [ys[2], ys[2]], **line_kw)
        # vlines
        for j in range(len(xq)):
            ax.plot([xq[j], xq[j]], [ys[1], ys[3]], **line_kw)

        # Annotate with the median and CI values.
        txt_kw = dict(
            ha="center",
            va="center",
            zorder=3,
            bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.6, pad=0),
            transform=ax.get_xaxis_transform(),
        )
        texts = []
        for j in range(len(xq)):
            t1 = ax.text(xq[j], ys[0], f"{100*q[j]:g}%", **txt_kw)
            t2 = ax.text(xq[j], ys[4], _num2str(xq[j]), **txt_kw)
            texts.extend([t1, t2])

        def adjust(event_=None):
            # Fix text items from overlapping. Maybe!
            adjust_text(
                texts,
                ax=ax,
                only_move=dict(points="", objects="", text="x"),
                avoid_points=False,
                avoid_self=False,
            )

        adjust()
        ax.figure.canvas.mpl_connect("resize_event", adjust)

    return ax.figure, ax


class _DinfPlotSubCommand(_SubCommand):
    """
    Base class for `dinf-plot` subcommands.
    """

    def add_argument_output_file(self):
        self.parser.add_argument(
            "-o",
            "--output-file",
            type=str,
            metavar="output.pdf",
            help=(
                "Output file for the figure. "
                "The file extension determines the filetype, which can be "
                "any format supported by Matplotlib (e.g. pdf, svg, png)."
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
            "--top",
            metavar="N",
            type=int,
            help="Accept only the N top samples, ranked by probability.",
        )

    def add_argument_weighted(self):
        self.parser.add_argument(
            "-W",
            "--weighted",
            action="store_true",
            help="Weight the parameter contributions by their probability.",
        )

    def add_argument_data_file(self, nargs):
        self.parser.add_argument(
            "data_files",
            metavar="data.npz",
            type=pathlib.Path,
            nargs=nargs,
            help="Data file containing discriminator predictions.",
        )

    def add_argument_discriminators(self):
        self.parser.add_argument(
            "discriminators",
            metavar="discriminator.nn",
            nargs="+",
            help="The discriminator network(s) to plot.",
        )

    def add_argument_gan_folder(self):
        self.parser.add_argument(
            "gan_folder",
            type=pathlib.Path,
            help="Folder containing results from a GAN run.",
        )

    def get_abc_datasets(self, data_files, top_n):
        """Load data file and filter by top-n or by probability."""
        datasets = []
        for filename in data_files:
            data = dinf.load_results(filename)
            # Flatten MCMC datasets with multiple chains.
            data = data.reshape(-1)

            if top_n is not None:
                k = len(data) - top_n
                data = np.partition(data, k, order="_Pr")[k:]

            datasets.append(data)

        return datasets

    def get_gan_datasets(self, gan_folder):
        discriminators = []
        data_files = []
        store = dinf.Store(gan_folder, create=False)
        dataset_type = None
        for j, path in enumerate(store):
            if (path / "discriminator.nn").exists():
                discriminators.append(path / "discriminator.nn")
            for prefix in ("abc", "mcmc", "pg-gan-proposals"):
                if (path / f"{prefix}.npz").exists():
                    if dataset_type is None:
                        dataset_type = prefix
                    else:
                        assert dataset_type == prefix, (j, dataset_type, prefix)
                    data_files.append(path / f"{prefix}.npz")
        return discriminators, data_files


class _Demes(_DinfPlotSubCommand):
    """
    Plot a demes-as-tubes demographic model using DemesDraw.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "demes")
        self.add_argument_output_file()
        self.add_argument_model()

    def __call__(self, args: argparse.Namespace):
        parameters = dinf.DinfModel.from_file(args.model).parameters

        # _dinf_user_module is the name given to the args.model module in
        # DinfModel.from_file(), which gets cached in sys.modules.
        # As a side-effect, we can now import it with this name to look for
        # a demography function.
        import _dinf_user_module  # type: ignore
        import inspect
        import demesdraw

        demography = getattr(_dinf_user_module, "demography", None)
        if not inspect.isfunction(demography):
            raise AttributeError(f"{args.model}: demography() function not found.")

        sig = inspect.signature(demography)
        if inspect.Parameter.VAR_KEYWORD in {v.kind for v in sig.parameters.values()}:
            # The function uses **kwargs, so we don't know what parameters
            # it expects. Just guess and pass all the dinf_model.parameters.
            demog_params = set(parameters)
        else:
            demog_params = set(sig.parameters)

        # If the parameter has a truth value, use that. Otherwise, use the
        # mid point of the parameter's range.
        param_kwargs = {
            name: p.truth if p.truth is not None else (p.high + p.low) / 2
            for name, p in parameters.items()
            if name in demog_params
        }

        graph = demography(**param_kwargs)
        fig, ax = plt.subplots(figsize=plt.figaspect(9 / 16))
        demesdraw.tubes(graph, ax=ax)

        if args.output_file is None:
            plt.show()
        else:
            fig.savefig(args.output_file)


class _Features(_DinfPlotSubCommand):
    """
    Plot feature matrices as heatmaps.

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
        self.add_argument_model()

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)

        if args.target:
            assert dinf_model.target_func is not None
            mats = dinf_model.target_func(args.seed)
        else:
            ss = np.random.SeedSequence(args.seed)
            ss_generator, ss_thetas = ss.spawn(2)
            rng = np.random.default_rng(ss_thetas)
            thetas = dinf_model.parameters.sample_prior(size=1, rng=rng)
            mats = dinf_model.generator_func_v((ss_generator, thetas[0]))

        if not isinstance(mats, dict):
            mats_dict = {"": mats}
        else:
            mats_dict = mats

        fig, axs = features(mats_dict, subplots_kw=dict(figsize=plt.figaspect(9 / 16)))
        if args.output_file is None:
            plt.show()
        else:
            fig.savefig(args.output_file)


class _Metrics(_DinfPlotSubCommand):
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
        metrics_collection = {
            pathlib.Path(d).name: dinf.Discriminator.from_file(d).metrics
            for d in args.discriminators
        }
        fig, axs = metrics(
            metrics_collection,
            subplot_mosaic_kw=dict(figsize=plt.figaspect(9 / 16)),
        )
        if args.output_file is None:
            plt.show()
        else:
            fig.savefig(args.output_file)


class _Hist2d(_DinfPlotSubCommand):
    """
    Plot 2d marginal posterior densities.

    One plot is produced for each unique pair of parameters.
    As this may lead to a large number of plots (particularly
    for interactive use!), the choice of which parameters to
    plot can be specified using the -x and -y options.

    If a pdf requested with the -o option, a multipage pdf is
    created. If another format is requested, then one file is
    created for each figure (the requested filename will be
    modified to include the parameter names).

    The resulting figure is a 2d histogram, with darker squares
    indicating higher densities. If the data correspond to a
    simulation-only model, then the parameters' truth values will
    be indicated by red lines. By default, all values in the
    data file contribute equally to the histogram. For parameter
    values drawn from the prior distribution, this will therefore
    show the prior distribution. A posterior sample can be obtained
    by weighting parameter values by the discriminator probabilities
    using the -W option, and/or rejection sampling the prior sample
    using the --top option.
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
        self.add_argument_model(required=False)
        self.add_argument_data_file(nargs=1)

    def __call__(self, args: argparse.Namespace):
        parameters = None
        if args.model is not None:
            parameters = dinf.DinfModel.from_file(args.model).parameters

        datasets = self.get_abc_datasets(args.data_files, args.top)
        assert len(datasets) == 1
        data = datasets[0]
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
                raise ValueError(f"{args.data_files[0]}: parameter {param} not found")

        hist2d_kw = {}
        if args.weighted:
            hist2d_kw["weights"] = data["_Pr"]

        # Count the number of figures.
        n_figures = 0
        done: set = set()
        for x_param in args.x_param:
            for y_param in args.y_param:
                if x_param == y_param or (y_param, x_param) in done:
                    continue
                    done.add((x_param, y_param))
                    n_figures += 1

        done = set()
        with _MultiPage(args.output_file, n_figures) as pages:
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
                    pages.savefig(fig, hint=f"{x_param}_{y_param}")
                    plt.close(fig)


def _kde1d_reflect(
    x: np.ndarray,
    /,
    *,
    weights: np.ndarray | None = None,
    left: float | None = None,
    right: float | None = None,
    n_points: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kernel density estimate, with reflection to mitigate edge effects.

    Note that the reflection is only implemented in 1d.

    :param x:
        The data values.
    :param weights:
        Weights for each data value.
    :param left:
        The left bound for the data.
        If None, the minimum value will be used.
    :param right:
        The right bound for the data.
        If None, the maximum value will be used.
    :param n_points:
        Number of points at which to evaluate the KDE.
        The points will be evenly spaced between the left and right edges.
    :return:
        A 2-tuple of (xrange, pdf), where ``xrange`` are the points at which
        the KDE is evaluated, and ``pdf`` are the KDE for those points.
    """
    if left is None:
        left = x.min()
    if right is None:
        right = x.max()
    kde = scipy.stats.gaussian_kde(x, bw_method="scott", weights=weights)
    # KDEs for left-relected and right-reflected x's that are beyond the
    # bounds of the data/parameters. Adding these avoids the usual drop in
    # density at the edges of the domain (but may introduce other artifacts).
    bw = kde.factor
    kde_left = scipy.stats.gaussian_kde(2 * left - x, bw_method=bw, weights=weights)
    kde_right = scipy.stats.gaussian_kde(2 * right - x, bw_method=bw, weights=weights)

    xrange = np.linspace(left, right, n_points)
    pdf = kde_left(xrange) + kde(xrange) + kde_right(xrange)

    return xrange, pdf


class _Hist(_DinfPlotSubCommand):
    """
    Plot marginal posterior densities.

    One plot is produced for the discriminator probabilities,
    plus one plot for each model parameter. The choice of which
    parameter to plot can be specified using the -x option,
    with the special value "_Pr" indicating the discriminator
    probabilities.

    If a pdf requested with the -o option, a multipage pdf is
    created. If another format is requested, then one file is
    created for each figure (the requested filename will be
    modified to include the parameter name).

    The resulting figure is a histogram. If the data correspond
    to a simulation-only model (provided via the -m option),
    then the parameter's truth value will be shown as a vertical
    red line. A 95% credible interval is shown at the bottom of
    the figure. By default, all values in the data file contribute
    equally to the histogram. For parameter values drawn from the
    prior distribution, this will therefore show the prior
    distribution. A posterior sample can be obtained by weighting
    parameter values by the discriminator probabilities using the
    -W option, and/or rejection sampling the prior sample using
    the --top option.
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
        self.parser.add_argument(
            "--resample",
            action="store_true",
            help=(
                "Resample parameter values to obtain a smoother density estimate. "
                "This matches the n-dimensional KDE sampling used with the ABC-GAN."
            ),
        )
        self.parser.add_argument(
            "--kde",
            action="store_true",
            help="Also draw a 1-dimensional marginal kernel density estimate.",
        )
        self.add_argument_model(required=False)
        self.add_argument_data_file(nargs="+")

    def __call__(self, args: argparse.Namespace):
        parameters = None
        if args.model is not None:
            parameters = dinf.DinfModel.from_file(args.model).parameters
        datasets = self.get_abc_datasets(args.data_files, args.top)

        if args.x_param is None:
            args.x_param = datasets[0].dtype.names
        if len(set(args.x_param)) != len(args.x_param):
            raise ValueError(f"--x-param values are not unique: {args.x_param}")

        for j, data in enumerate(datasets):
            for param in args.x_param:
                if param not in data.dtype.names:
                    raise ValueError(
                        f"{args.data_files[j]}: parameter {param} not found"
                    )

        datasets_resampled = []
        if args.resample:
            if parameters is None:
                raise ValueError(
                    "When --resample is requested, must also specify --model"
                )
            rng = np.random.default_rng(123)
            for data in datasets:
                names = list(data.dtype.names)[1:]
                probs = data["_Pr"]
                thetas = structured_to_unstructured(data[names])
                X = parameters.sample_kde(
                    thetas,
                    probs=probs,
                    size=1_000_000,
                    rng=rng,
                )
                X_dict = {name: X[..., j] for j, name in enumerate(names)}
                datasets_resampled.append(X_dict)

        with _MultiPage(args.output_file, len(args.x_param)) as pages:
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
                    ax.set_xlabel(x_param)
                    if args.weighted:
                        hist_kw["weights"] = data["_Pr"]
                    ci = not args.cumulative

                for j, (data, path) in enumerate(zip(datasets, args.data_files)):
                    x = data[x_param]
                    hist_kw["label"] = path.name
                    if args.resample and x_param != "_Pr":
                        if "weights" in hist_kw:
                            del hist_kw["weights"]
                        x = datasets_resampled[j][x_param]
                        hist_kw["bins"] = 100
                    hist(x, ax=ax, ci=ci, truth=truth, hist_kw=hist_kw)
                    if args.kde:
                        left = right = None
                        xlim = ax.get_xlim()
                        if parameters is not None and x_param != "_Pr":
                            left = max(parameters[x_param].low, xlim[0])
                            right = min(parameters[x_param].high, xlim[1])
                        xrange, pdf = _kde1d_reflect(
                            data[x_param], weights=data["_Pr"], left=left, right=right
                        )
                        if args.cumulative:
                            cdf = np.cumsum(pdf)
                            cdf /= cdf[-1]
                            y = cdf
                        else:
                            y = pdf
                        ax.plot(xrange, y, c="cyan", alpha=0.7)

                if args.cumulative:
                    ax.set_ylabel("cumulative density")
                else:
                    ax.set_ylabel("density")

                if len(datasets) > 1:
                    ax.legend()

                pages.savefig(fig, hint="Pr" if x_param == "_Pr" else x_param)
                plt.close(fig)


class _Gan(_DinfPlotSubCommand):
    """
    Plot GAN things.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "gan")

        self.add_argument_output_file()
        self.add_argument_abc_thresholds()
        self.add_argument_weighted()
        """
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
        """
        self.add_argument_model(required=False)
        self.add_argument_gan_folder()

    def __call__(self, args: argparse.Namespace):
        discriminators, data_files = self.get_gan_datasets(args.gan_folder)

        parameters = None
        if args.model is not None:
            parameters = dinf.DinfModel.from_file(args.model).parameters

        datasets = self.get_abc_datasets(data_files, args.top)
        x_params = datasets[0].dtype.names
        for data in datasets:
            assert data.dtype.names == x_params

        metrics_collection = {
            f"Iteration {j}": dinf.Discriminator.from_file(d).metrics
            for j, d in enumerate(discriminators)
        }

        cmap = matplotlib.cm.get_cmap("gnuplot")
        cycle = cycler(
            color=[cmap(i / len(discriminators)) for i in range(len(discriminators))]
        )
        saved_prop_cycle = matplotlib.rcParams["axes.prop_cycle"]
        matplotlib.rcParams["axes.prop_cycle"] = cycle

        fig, axs = metrics(
            metrics_collection,
            subplot_mosaic_kw=dict(figsize=plt.figaspect(9 / 16)),
        )
        matplotlib.rcParams["axes.prop_cycle"] = saved_prop_cycle

        axs["test_loss"].get_legend().remove()
        fig.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=0, vmax=len(discriminators) - 1),
                cmap=cmap,
            ),
            ax=axs["test_loss"],
            pad=0.02,
            fraction=0.04,
            label="iteration",
        )

        with _MultiPage(args.output_file, 1 + len(x_params)) as pages:
            pages.savefig(fig, hint="metrics")
            plt.close(fig)

            for x_param in x_params:
                fig, ax = plt.subplots(
                    figsize=plt.figaspect(9 / 16), constrained_layout=True
                )
                ax.violinplot(
                    [data[x_param] for data in datasets],
                    quantiles=[[0.025, 0.5, 0.975]] * len(datasets),
                    showextrema=False,
                )
                ax.set_ylabel(x_param)
                ax.set_xlabel("Iteration")

                if parameters is not None and x_param != "_Pr":
                    truth = parameters[x_param].truth
                    ax.axhline(truth, c="red", ls="-", alpha=0.7)

                fig.suptitle(x_param)
                pages.savefig(fig, hint=x_param)
                plt.close(fig)


def main(args_list=None):
    top_parser = argparse.ArgumentParser(
        prog="dinf.plot", description="Dinf plotting tools."
    )
    top_parser.add_argument(
        "-V", "--version", action="version", version=dinf.__version__
    )

    subparsers = top_parser.add_subparsers(dest="subcommand")
    _Demes(subparsers)
    _Features(subparsers)
    _Metrics(subparsers)
    _Hist(subparsers)
    _Hist2d(subparsers)
    _Gan(subparsers)

    args = top_parser.parse_args(args_list)
    if args.subcommand is None:
        top_parser.print_help()
        exit(1)

    if hasattr(args, "output_file") and args.output_file is None:
        # Interactive figure. Default dpi is 100, which makes the scale
        # of interactive figures quite different to that of saved files.
        # Bumping the dpi to 140 provides greater parity.
        plt.rc("figure", dpi=140)

    _set_loglevel(args.quiet, args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
