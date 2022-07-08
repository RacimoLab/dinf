from __future__ import annotations
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt

import dinf


_in_notebook = False
try:
    get_ipython()
except NameError:
    _in_notebook = True

if _in_notebook:
    # Output SVG.
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("svg")


def metrics(
    *,
    networks: Dict[str, dinf.Discriminator],
    legend_title: str | None = None,
    subplot_mosaic_kw={},
):
    """
    Plots training metrics for a discriminator neural network.

    :param networks:
        A dictionary mapping labels to Discriminator objects.
    :param legend_title:
        A title for the legend.
    :param subplot_mosaic_kw:
        Additional keyword arguments that will be passed to matplotlib's
        ``subplot_mosiac()`` function.
    :rtype: tuple[Figure, dict[label, Axes]]
    :return:
        A (figure, axes_dict) tuple, where axes_dict is a dictionary mapping
        metrics to the ``matplotlib.axes.Axes`` objects.
        (As obtained from matplotlib's ``subplot_mosaic()``).
    """
    fig, axs = plt.subplot_mosaic(
        [
            ["train_loss", "test_loss"],
            ["train_accuracy", "test_accuracy"],
        ],
        sharex=True,
        constrained_layout=True,
        **subplot_mosaic_kw,
    )
    axs["train_loss"].get_shared_y_axes().join(axs["train_loss"], axs["test_loss"])
    axs["train_accuracy"].get_shared_y_axes().join(
        axs["train_accuracy"], axs["test_accuracy"]
    )
    if not fig.get_tight_layout():
        fig.set_constrained_layout(True)

    metrics = ("train_loss", "test_loss", "train_accuracy", "test_accuracy")
    for label, network in networks.items():
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


def feature(mat, /, *, ax, channel=0, cmap=None, cb=True, cb_label, vmax=None):
    """
    Plot a feature matrix as an image.

    :param mat:
        The feature matrix
    :param ax:
        Matplotlib axes onto which the plot will be drawn.
    :param channel:
        Channel number of the feature matrix to draw.
    :param cmap:
        Matplotlib colour map.
    :param cb:
        If True, add a colour bar.
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


def features(mats, /, *, subplots_kw={}):
    """
    Plot multiple feature matrices.

    :param mats:
        The multiple feature matrices.
    :rtype: tuple[Figure, Axes]
    :return:
        A (figure, axes) tuple.
        (As obtained from matplotlib's ``subplots()``).
    """
    assert isinstance(mats, dict)

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


import argparse
import pathlib
import textwrap

import numpy as np

from .cli import ADRDFormatter, _GENOB_MODEL_HELP


class _Metrics:
    """
    Plot training metrics for discriminator networks.
    """

    def __init__(self, subparsers):
        parser = subparsers.add_parser(
            "metrics",
            help="Plot training metrics for discriminator networks.",
            description=textwrap.dedent(self.__doc__),
            formatter_class=ADRDFormatter,
        )
        parser.set_defaults(func=self)
        parser.add_argument(
            "out_file",
            metavar="output.pdf",
            help="The output file for the figure.",
        )
        parser.add_argument(
            "discriminators",
            metavar="discriminator.pkl",
            nargs="*",
            help="The discriminator network(s) to plot.",
        )

    def __call__(self, args: argparse.Namespace):
        discriminators = {
            pathlib.Path(d).name: dinf.Discriminator.from_file(d)
            for d in args.discriminators
        }
        fig, axs = metrics(
            networks=discriminators,
            subplot_mosaic_kw=dict(figsize=plt.figaspect(9 / 16)),
        )
        fig.savefig(args.out_file)


class _Features:
    """
    Plot feature matrix or matrices from a Dinf model.

    By default, features are extracted from the generator for plotting.
    To instead extract features from the target dataset,
    use the --target option.
    """

    def __init__(self, subparsers):
        parser = subparsers.add_parser(
            "features",
            help="Plot feature matrix or matrices from a Dinf model.",
            description=textwrap.dedent(self.__doc__),
            formatter_class=ADRDFormatter,
        )
        parser.set_defaults(func=self)
        parser.add_argument(
            "--target",
            action="store_true",
            help="Plot feature(s) from the target dataset.",
        )
        parser.add_argument(
            "-S",
            "--seed",
            type=int,
            help="Seed for the random number generator",
        )
        parser.add_argument(
            "out_file",
            metavar="output.pdf",
            help="The output file for the figure.",
        )
        parser.add_argument(
            "genob_model",
            metavar="user_model.py",
            type=pathlib.Path,
            help=_GENOB_MODEL_HELP,
        )

    def __call__(self, args: argparse.Namespace):
        genobuilder = dinf.Genobuilder.from_file(args.genob_model)

        if args.target:
            mats = genobuilder.target_func(args.seed)
        else:
            rng = np.random.default_rng(args.seed)
            thetas = genobuilder.parameters.draw_prior(num_replicates=1, rng=rng)
            mats = genobuilder.generator_func(
                (rng.integers(low=0, high=2**31), thetas[0])
            )

        if not isinstance(mats, dict):
            mats_dict = {"": mats}
        else:
            mats_dict = mats

        fig, axs = features(mats_dict, subplots_kw=dict(figsize=plt.figaspect(9 / 16)))
        fig.set_constrained_layout(True)
        fig.savefig(args.out_file)


def _main(args_list=None):
    top_parser = argparse.ArgumentParser(
        prog="dinf.plot",
        description="Dinf plotting tools.",
    )

    subparsers = top_parser.add_subparsers(dest="subcommand", metavar="{metrics}")
    _Metrics(subparsers)
    _Features(subparsers)

    args = top_parser.parse_args(args_list)
    if args.subcommand is None:
        top_parser.print_help()
        exit(1)
    args.func(args)


if __name__ == "__main__":
    _main()
