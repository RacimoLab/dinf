from __future__ import annotations
import itertools
import pathlib

import arviz as az
import corner
import emcee
import matplotlib
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


def _pair_plot(dataset, truths, figsize, var_names=None, filter_vars=None):
    if var_names is not None:
        ndim = len(var_names)
    else:
        ndim = len(dataset.posterior.data_vars.keys())
    # fig, ax = plt.subplots(ndim, ndim, figsize=figsize)

    # Arviz pair plots look crap.
    axes = az.plot_pair(
        dataset,
        var_names=var_names,
        filter_vars=filter_vars,
        #ax=ax,
        figsize=figsize,
        kind="hexbin",
        marginals=True,
        scatter_kwargs=dict(rasterized=True),
        reference_values=truths,
        reference_values_kwargs=dict(
            markeredgecolor="red", marker="x", markersize=20, markeredgewidth=3
        ),
    )
    fig = _fig_from_arviz_axes(axes)
    """

    fig = corner.corner(
        dataset,
        var_names=var_names,
        truths=truths,
        # bins=50,
        truth_color="red",
        fig=fig,
        # quantiles=[0.025, 0.5, 0.975],
    )
    """

    #fmt = matplotlib.ticker.ScalarFormatter(useOffset=False, useMathText=True)
    #fmt.set_scientific(True)
    #for ax in fig.axes:
    #    ax.xaxis.set_major_formatter(fmt)
    #    ax.yaxis.set_major_formatter(fmt)

    # Why is it so hard to get a single value out of an arviz dataset?
    acceptance_rate = float(dataset.sample_stats["acceptance_rate"].as_numpy()[0][0])

    fig.suptitle(f"pair plot / acceptance_rate = {acceptance_rate:.3g}")
    fig.set_tight_layout(True)
    return fig

def _all_pair_plots(dataset, truths, figsize):
    var_names = list(dataset.posterior.data_vars.keys())
    for var_name1, var_name2 in itertools.permutations(var_names, 2):
        fig = _pair_plot(dataset, truths, figsize, var_names=[var_name1, var_name2])
        yield fig

def _grouped_pair_plots(dataset, truths, figsize):
    for var_names, filter_vars in [
        ("^N_", "regex"),
        ("^m_", "regex"),
        ("^dT_", "regex"),
    ]:
        fig = _pair_plot(dataset, truths, figsize, var_names=var_names, filter_vars=filter_vars)
        yield fig

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
    ncols = int(np.ceil(np.sqrt(ndim)))
    nrows = int(np.ceil(ndim / ncols))
    fig, axes = plt.subplots(ncols, nrows, figsize=figsize, tight_layout=True)
    # dataset is (ndim, walkers, samples)
    # chain is (samples, walkers, ndim), as emcee expects
    chain = np.array(dataset.posterior.to_array()).swapaxes(0, 2)
    taus = emcee.autocorr.integrated_time(chain, quiet=True)
    for j, ax in enumerate(axes.flat):
        if j >= ndim:
            ax.set_axis_off()
            continue
        acf = _autocorr(chain[:, :, j])
        x = np.arange(len(chain))
        ax.plot(x, acf)
        ax.set_title(f"{var_names[j]}")
        # ax.vlines(taus[j], 0, 1, color="red")
        # print(var_names[j], "autocorrelation", taus[j])

    fig.suptitle("autocorrelation")
    return fig


def report(
    genobuilder: Genobuilder,
    working_directory: None | str | pathlib.Path = None,
    aspect=9 / 16,
    scale=2,
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
            # discard burn-in
            #num_draws = len(dataset.posterior.draw)
            #dataset = dataset.isel(draw=slice(num_draws // 2, None, 100))
            acceptance_rate = float(dataset.sample_stats.acceptance_rate.iloc(chain=0, draw=0))

            """
            for fig in _all_pair_plots(dataset, truths, figsize):
                fig.suptitle(f"iteration {j} / {fig._suptitle.get_text()}")
                pdf.savefig(figure=fig, dpi=dpi)
                plt.close(fig)
            """
            for fig in _grouped_pair_plots(dataset, truths, figsize):
                fig.suptitle(f"iteration {j} / {fig._suptitle.get_text()}")
                pdf.savefig(figure=fig, dpi=dpi)
                plt.close(fig)

            fig = _plot_autocorr(dataset, figsize)
            fig.suptitle(f"iteration {j} / {fig._suptitle.get_text()}")

            pdf.savefig(figure=fig, dpi=dpi)
            plt.close(fig)
