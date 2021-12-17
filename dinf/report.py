from __future__ import annotations
import pathlib

import arviz as az
import corner
import emcee
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .genobuilder import Genobuilder
from .parameters import Parameters
from .store import Store


def _get_truths(parameters: Parameters):
    truths = None
    if all(p.truth is not None for p in parameters.values()):
        truths = {k: p.truth for k, p in parameters.items()}
    return truths


def _fig_from_arviz_axes(axes):
    if hasattr(axes, "flat"):
        # axes is an ndarray of matplotlib.axes.Axes
        fig = axes.flat[0].figure
    else:
        # axes is a matplotlib.axes.Axes
        fig = axes.figure
    return fig


def _pair_plot(dataset, truths, figsize):
    # Arviz pair plots look crap.
    """
    axes = az.plot_pair(
        dataset,
        figsize=figsize,
        kind=["scatter", "kde"],
        marginals=True,
        scatter_kwargs=dict(rasterized=True),
        reference_values=truths,
        reference_values_kwargs=dict(
            markeredgecolor="red", marker="x", markersize=20, markeredgewidth=3
        ),
    )
    fig = _fig_from_arviz_axes(axes)
    """

    ndim = len(dataset.posterior.data_vars.keys())
    fig, ax = plt.subplots(ndim, ndim, figsize=figsize)
    fig = corner.corner(
        dataset,
        truths=truths,
        # bins=50,
        truth_color="red",
        fig=fig,
        # quantiles=[0.025, 0.5, 0.975],
    )

    # Why is it so hard to get a single value out of an arviz dataset?
    acceptance_rate = float(dataset.sample_stats["acceptance_rate"].as_numpy()[0][0])

    fig.suptitle(f"pair plot / acceptance_rate = {acceptance_rate:.3g}")
    return fig


def _autocorr(x):
    n_t, n_w = x.shape
    acf = np.zeros(n_t)
    for k in range(n_w):
        acf += emcee.autocorr.function_1d(x[:, k])
    acf /= n_w
    return acf


def _plot_autocorr(dataset, figsize):

    # Arviz produces one plot per chain per variable.
    # Completely incomprehensible.
    """
    axes = az.plot_autocorr(
        dataset,
        figsize=figsize,
        combined=False,
    )
    fig = _fig_from_arviz_axes(axes)
    """

    var_names = list(dataset.posterior.data_vars.keys())
    ndim = len(var_names)
    fig, axes = plt.subplots(1, ndim, figsize=figsize)
    # dataset is (ndim, walkers, samples)
    # chain is (samples, walkers, ndim), as emcee expects
    chain = np.array(dataset.posterior.to_array()).swapaxes(0, 2)
    # taus = emcee.autocorr.integrated_time(chain, quiet=True)
    for j, ax in enumerate(axes.flat):
        acf = _autocorr(chain[:, :, j])
        x = np.arange(len(chain))
        ax.plot(x, acf)
        ax.set_title(f"{var_names[j]}")
        # ax.vlines(taus[j], 0, 1, color="red")

    fig.suptitle("autocorrelation")
    return fig


def report(
    genobuilder: Genobuilder,
    working_directory: None | str | pathlib.Path = None,
    aspect=9 / 16,
    scale=1,
):
    if working_directory is None:
        working_directory = "."

    store = Store(working_directory)
    output_filename = pathlib.Path(working_directory) / "report.pdf"
    truths = _get_truths(genobuilder.parameters)

    figsize = scale * plt.figaspect(aspect)
    dpi = 200

    with PdfPages(output_filename) as pdf:
        for j, path in enumerate(store):
            dataset = az.from_netcdf(path / "mcmc.ncf")

            fig = _pair_plot(dataset, truths, figsize)
            fig.suptitle(f"iteration {j} / {fig._suptitle.get_text()}")

            pdf.savefig(figure=fig, dpi=dpi)
            plt.close(fig)

            fig = _plot_autocorr(dataset, figsize)
            fig.suptitle(f"iteration {j} / {fig._suptitle.get_text()}")

            pdf.savefig(figure=fig, dpi=dpi)
            plt.close(fig)
