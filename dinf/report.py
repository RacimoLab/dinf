import functools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import arviz as az
import natsort

from autocorr import AutoCorrTime


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


def _subset_slice(dataset: az.InferenceData, slice_: slice) -> az.InferenceData:
    """Select a slice of the data points in each chain."""
    datadict = {
        p: dataset.posterior[p].values[:, slice_] for p in dataset.posterior.keys()
    }
    return az.convert_to_inference_data(datadict)


def _fig_from_arviz_axes(axes):
    if hasattr(axes, "flat"):
        # axes is an ndarray of matplotlib.axes.Axes
        fig = axes.flat[0].figure
    else:
        # axes is a matplotlib.axes.Axes
        fig = axes.figure
    return fig


def iter_chains(dataset: az.InferenceData):
    num_chains = len(dataset.posterior.chain)
    for j in range(num_chains):
        datadict = {p: dataset.posterior[p].values[j] for p in dataset.posterior.keys()}
        yield az.convert_to_inference_data(datadict)


def plot_pair(dataset, aspect=9 / 16, scale=1):
    var_names = [k for k in dataset.posterior.data_vars.keys() if k != "D"]
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
        var_names=("[^D]"),
        filter_vars="regex",
        figsize=figsize,
        kind=kind,
        combined=False,
        compact=False,
        trace_kwargs=dict(rasterized=True),
    )
    fig = _fig_from_arviz_axes(axes)
    fig.set_constrained_layout(False)
    fig.set_tight_layout(True)
    return fig


def plot_mcmc_autocorr(dataset, aspect=9 / 16, scale=1):
    figsize = scale * plt.figaspect(aspect)
    axes = az.plot_autocorr(
        dataset,
        figsize=figsize,
        combined=False,
        var_names=("[^D]"),
        filter_vars="regex",
    )
    fig = _fig_from_arviz_axes(axes)
    fig.set_constrained_layout(False)
    fig.set_tight_layout(True)
    return fig


def plot_mcmc_ess(dataset, kind, aspect=9 / 16, scale=1):
    figsize = scale * plt.figaspect(aspect)
    axes = az.plot_ess(
        dataset,
        kind=kind,
        figsize=figsize,
        var_names=("[^D]"),
        filter_vars="regex",
    )
    fig = _fig_from_arviz_axes(axes)
    fig.set_constrained_layout(False)
    fig.set_tight_layout(True)
    return fig


def plot_mcmc_iat(dataset, aspect=9 / 16, scale=1):
    var_names = [k for k in dataset.posterior.data_vars.keys() if k != "D"]
    samples = np.array([dataset.posterior[k] for k in var_names])
    samples = samples.swapaxes(0, 2)
    nsteps, nwalkers, nvars = samples.shape
    # assert nsteps >= 100

    var_names = [k for k in dataset.posterior.data_vars.keys() if k != "D"]
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(var_names),
        figsize=scale * plt.figaspect(aspect),
        tight_layout=True,
        squeeze=False,
    )

    N = np.arange(100, nsteps, 100, dtype=int)
    dfm = np.empty((len(N), len(var_names)))
    gw = np.empty((len(N), len(var_names)))
    mk = np.empty((len(N), len(var_names)))
    for k, n in enumerate(N):
        dfm[k] = AutoCorrTime(samples[:n], method="dfm")
        gw[k] = AutoCorrTime(samples[:n], method="gw")
        mk[k] = AutoCorrTime(samples[:n], method="mk")

    for j, var_name in enumerate(var_names):
        ax = axes.flat[j]
        ax.plot(N, dfm[:, j], label="dfm")
        ax.plot(N, gw[:, j], label="gw")
        ax.plot(N, mk[:, j], label="mk")
        ax.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
        ax.legend()
        ax.set_title(f"Integrated autocorrelation time ({var_name})")
        ax.set_xlabel("N")
        ax.set_ylabel(r"IAT")

    dfm_max = np.max(dfm[-1])
    print("IAT", dfm_max)

    return fig


def iat_max(dataset):
    """
    Max integrated autocorrelation time, over all variables.
    """
    var_names = [k for k in dataset.posterior.data_vars.keys() if k != "D"]
    samples = np.array([dataset.posterior[k] for k in var_names])
    samples = samples.swapaxes(0, 2)
    # nsteps, nwalkers, nvars = samples.shape
    dfm = AutoCorrTime(samples, method="dfm")
    # max over all vars
    dfm_max = np.max(dfm)
    return int(np.ceil(dfm_max))


def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def plot_gan_ecdf(datasets, aspect=9 / 16, scale=1):
    var_names = [k for k in datasets[0].posterior.data_vars.keys()]
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(var_names) + 1,
        gridspec_kw=dict(width_ratios=[1] * len(var_names) + [0.02 * len(var_names)]),
        figsize=scale * plt.figaspect(aspect),
        tight_layout=True,
        squeeze=False,
    )
    cmap = matplotlib.cm.get_cmap("coolwarm")
    kx = 0  # len(datasets) // 2
    norm = matplotlib.colors.Normalize(vmin=kx, vmax=len(datasets) - 1)

    posteriors = {var: list() for var in var_names}
    for j, dataset in enumerate(datasets[kx:], kx):
        for var, ax in zip(var_names, axes.flat):
            posterior = dataset.posterior[var].values.reshape(-1)
            posteriors[var].append(posterior)
            xs, ys = ecdf(posterior)
            ax.plot(xs, ys, alpha=0.5, c=cmap(norm(j)))

    for var, ax in zip(var_names, axes.flat):
        posterior = np.array(posteriors[var]).reshape(-1)
        xs, ys = ecdf(posterior)
        ax.plot(xs, ys, "k")

        ax.set_title(var)
        ax.set_xlabel(var)
        ax.set_ylabel(r"Pr($x<X$)")

    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=axes.flat[-1],
        label="iteration",
    )

    return fig


def partial(f, *args, **kwargs):
    """
    Apply functools.update_wrapper to the functools.partial.
    """
    return functools.update_wrapper(functools.partial(f, *args, **kwargs), f)


def load_ncf_multi(filenames):
    filenames = natsort.natsorted(filenames)
    return [az.from_netcdf(filename) for filename in filenames]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            f"usage: {sys.argv[0]} {{abc|mcmc|gan}} report.pdf data.ncf [... dataN.ncf]"
        )
        exit(1)

    subcommand = sys.argv[1]
    if subcommand not in ("abc", "mcmc", "gan"):
        raise RuntimeError(f"Not a valid subcommand: '{subcommand}'")

    output_filename = sys.argv[2]
    if subcommand == "gan":
        input_filenames = sys.argv[3:]
        dataset = load_ncf_multi(input_filenames)
    else:
        input_filename = sys.argv[3]
        dataset = az.from_netcdf(input_filename)

    with PdfPages(output_filename) as pdf:
        if subcommand == "abc":
            funcs = [plot_abc_D, plot_abc_threshold, plot_abc_quantile]
        elif subcommand == "mcmc":
            funcs = [
                # partial(plot_mcmc_ess, kind="quantile"),
                # partial(plot_mcmc_ess, kind="evolution"),
                plot_mcmc_iat,
                plot_mcmc_autocorr,
                plot_pair,
                partial(plot_mcmc_trace, kind="trace"),
                # partial(plot_mcmc_trace, kind="rank_bars"),
            ]
        elif subcommand == "gan":
            funcs = [
                plot_gan_ecdf,
            ]
            pass

        for func in funcs:
            # print("pre", func.__name__)
            fig = func(dataset)
            # print("post", func.__name__)
            pdf.savefig(figure=fig, dpi=200)
            plt.close(fig)

        if subcommand == "mcmc":
            for chaindataset in iter_chains(dataset):
                fig = plot_mcmc_trace(chaindataset, kind="trace")
                pdf.savefig(figure=fig, dpi=200)
                plt.close(fig)
