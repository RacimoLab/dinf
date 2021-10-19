import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import arviz as az


def _subset_quantile(dataset: az.InferenceData, q) -> az.InferenceData:
    """Get q'th quantile of the dataset by discriminator probability."""
    idx = np.flip(np.argsort(dataset.posterior["D"].values[0]))
    top_n = int(q * len(dataset.posterior.draw))
    datadict = {
        p: dataset.posterior[p].values[:, idx[:top_n]] for p in dataset.posterior.keys()
    }
    return az.convert_to_inference_data(datadict)


def _subset_threshold(dataset: az.InferenceData, x: float) -> az.InferenceData:
    """Get subset of the dataset with discriminator probability > x."""
    idx = [j for j, D in enumerate(dataset.posterior["D"].values[0]) if D > x]
    datadict = {
        p: dataset.posterior[p].values[:, idx] for p in dataset.posterior.keys()
    }
    return az.convert_to_inference_data(datadict)

def _subset_dropburnin(dataset: az.InferenceData, n: int) -> az.InferenceData:
    """Drop the first n data points in each chain."""
    datadict = {
        p: dataset.posterior[p].values[:, n:] for p in dataset.posterior.keys()
    }
    return az.convert_to_inference_data(datadict)

def iter_chains(dataset: az.InferenceData):
    num_chains = len(dataset.posterior.chain)
    for j in range(num_chains):
        datadict = {
            p: dataset.posterior[p].values[j] for p in dataset.posterior.keys()
        }
        yield az.convert_to_inference_data(datadict)


def plot_pair(dataset, aspect=9 / 16, scale=1):
    var_names = [p for p in dataset.posterior.keys() if p != "D"]
    fig, ax = plt.subplots(
        len(var_names),
        len(var_names),
        figsize=scale * plt.figaspect(aspect),
        tight_layout=True,
        squeeze=False,
    )
    az.plot_pair(
        dataset,
        var_names=var_names,
        ax=ax,
        kind=["scatter", "kde"],
        marginals=True,
        scatter_kwargs=dict(rasterized=True),
    )
    ax[1, 0].scatter(10000, 200, c="red", marker="x", zorder=5)

    return fig


def plot_abc_quantile(dataset):
    dataset = _subset_quantile(dataset, 0.001)
    fig = plot_pair(dataset)
    fig.suptitle("0.1% largest D(x)")
    return fig


def plot_abc_threshold(dataset):
    dataset = _subset_threshold(dataset, 0.99)
    fig = plot_pair(dataset)
    fig.suptitle("D(x) > 0.99")
    return fig


def plot_abc_D(dataset, aspect=9 / 16, scale=1):
    fig, ax = plt.subplots(
        figsize=scale * plt.figaspect(aspect),
        tight_layout=True,
    )
    N0 = np.concatenate(dataset.posterior["N0"])
    N1 = np.concatenate(dataset.posterior["N1"])
    D = np.concatenate(dataset.posterior["D"])
    sc = ax.scatter(N0, N1, c=D, vmin=0, vmax=1, rasterized=True, s=2)
    fig.colorbar(sc)
    ax.set_title("D(x)")
    ax.set_xlabel("N0")
    ax.set_ylabel("N1")

    ax.scatter(10000, 200, c="red", marker="x")

    return fig


def plot_mcmc_trace(dataset, kind, aspect=9 / 16, scale=1):
    figsize = scale * plt.figaspect(aspect)
    axes = az.plot_trace(
        dataset,
        figsize=figsize,
        kind=kind,
        combined=False,
        compact=False,
        trace_kwargs=dict(rasterized=True),
    )
    fig = axes.reshape(-1)[0].figure
    fig.set_constrained_layout(False)
    fig.set_tight_layout(True)
    return fig


def plot_mcmc_autocorr(dataset, aspect=9 / 16, scale=1):
    figsize = scale * plt.figaspect(aspect)
    axes = az.plot_autocorr(dataset, figsize=figsize, combined=False)
    fig = axes.reshape(-1)[0].figure
    fig.set_constrained_layout(False)
    fig.set_tight_layout(True)
    return fig


def plot_mcmc_ess(dataset, kind, aspect=9 / 16, scale=1):
    figsize = scale * plt.figaspect(aspect)
    axes = az.plot_ess(dataset, kind=kind, figsize=figsize)
    fig = axes.reshape(-1)[0].figure
    fig.set_constrained_layout(False)
    fig.set_tight_layout(True)
    return fig


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} {{abc|mcmc}} data.ncf report.pdf")
        exit(1)

    subcommand = sys.argv[1]
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]

    if subcommand not in ("abc", "mcmc"):
        raise RuntimeError(f"Not a valid subcommand: '{subcommand}'")

    dataset = az.from_netcdf(input_filename)

    with PdfPages(output_filename) as pdf:
        if subcommand == "abc":
            funcs = [plot_abc_D, plot_abc_threshold, plot_abc_quantile]
        elif subcommand == "mcmc":
            funcs = [
                functools.partial(plot_mcmc_ess, kind="quantile"),
                functools.partial(plot_mcmc_ess, kind="evolution"),
                plot_mcmc_autocorr,
                plot_pair,
                functools.partial(plot_mcmc_trace, kind="trace"),
                #functools.partial(plot_mcmc_trace, kind="rank_bars"),
            ]
        for func in funcs:
            fig = func(dataset)
            pdf.savefig(figure=fig, dpi=200)
            plt.close(fig)
        if subcommand == "mcmc":
            for chaindataset in iter_chains(dataset):
                fig = plot_mcmc_trace(chaindataset, kind="trace")
                pdf.savefig(figure=fig, dpi=200)
                plt.close(fig)
