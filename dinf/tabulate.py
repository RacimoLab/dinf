from __future__ import annotations
import argparse
import contextlib
import logging
import pathlib
import sys

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from .cli import _SubCommand, _set_loglevel
from .misc import quantile
import dinf

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _open_default(file, mode="w", default=sys.stdout):
    """
    Context manager for opening a file, defaulting to stdout if file is None.
    """
    if file is None:
        yield default
    else:
        with open(file, mode) as f:
            yield f


class _DinfTabulateSubCommand(_SubCommand):
    """
    Base class for `dinf-tabulate` subcommands.
    """

    def add_argument_output_file(self):
        self.parser.add_argument(
            "-o",
            "--output-file",
            type=str,
            metavar="output.txt",
            help=(
                "Output file for the tabulated data. "
                "If not specified, output will be printed to stdout."
            ),
        )

    def add_argument_separator(self):
        self.parser.add_argument(
            "--separator",
            type=str,
            default="\t",
            help="The string that separates columns.",
        )

    def add_argument_format(self):
        self.parser.add_argument(
            "--format",
            type=str,
            help="Printf-style format specifier for float values.",
        )

    def add_argument_top(self):
        self.parser.add_argument(
            "--top",
            metavar="N",
            type=int,
            help="Filter data to retain top N samples, ranked by probability.",
        )

    def add_argument_weighted(self):
        self.parser.add_argument(
            "-W",
            "--weighted",
            action="store_true",
            help="Weight the parameter contributions by their probability.",
        )

    def add_argument_quantiles(self):
        self.parser.add_argument(
            "--quantiles",
            type=str,
            default="0.025,0.5,0.975",
            help="Comma separated list of quantiles to calculate.",
        )

    def add_argument_discriminators(self):
        self.parser.add_argument(
            "discriminators",
            metavar="discriminator.nn",
            type=pathlib.Path,
            nargs="+",
            help="The discriminator network(s) from which to tabulate metrics.",
        )

    def add_argument_data(self):
        self.parser.add_argument(
            "data",
            metavar="data.npz",
            type=pathlib.Path,
            help="Data file in numpy .npz format.",
        )


class _Metrics(_DinfTabulateSubCommand):
    """
    Print discriminator metrics.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "metrics")

        self.add_argument_output_file()
        self.add_argument_separator()
        self.add_argument_format()
        self.add_argument_discriminators()

    def __call__(self, args: argparse.Namespace):
        metrics_list = []
        for filename in args.discriminators:
            discr = dinf.Discriminator.from_file(filename)
            metrics_list.append(discr.metrics)

        assert len(metrics_list) > 0

        mkeys = list(metrics_list[0])
        for filename, metric in zip(args.discriminators, metrics_list):
            if metric.keys() != metrics_list[0].keys():
                raise ValueError(
                    f"{filename}: metrics ({list(metric.keys())}) "
                    f"don't match {args.discriminators[0]} ({mkeys})"
                )

        assert len(mkeys) > 0

        header = ["epoch"] + mkeys
        if len(metrics_list) > 1:
            header = ["file"] + header

        with _open_default(args.output_file) as f:
            print(*header, sep=args.separator, file=f)

            for filename, metric in zip(args.discriminators, metrics_list):
                n_epochs = len(list(metric.values())[0])
                for epoch in range(n_epochs):
                    line = []
                    if len(metrics_list) > 1:
                        line.append(filename)
                    line.append(epoch)
                    line.extend([metric[k][epoch] for k in mkeys])
                    if args.format is not None:
                        line = [args.format % v for v in line]
                    print(*line, sep=args.separator, file=f)


class _Data(_DinfTabulateSubCommand):
    """
    Print .npz data---predictions from a discriminator.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "data")

        self.add_argument_output_file()
        self.add_argument_separator()
        self.add_argument_format()
        self.add_argument_data()

    def __call__(self, args: argparse.Namespace):
        data = dinf.load_results(args.data)
        if args.format is None:
            # Numpy default.
            args.format = "%.18e"

        with _open_default(args.output_file) as f:
            print(*data.dtype.names, sep=args.separator, file=f)
            np.savetxt(f, data, delimiter=args.separator, fmt=args.format)


class _Quantiles(_DinfTabulateSubCommand):
    """
    Calculate quantiles of the data.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "quantiles")

        self.add_argument_output_file()
        self.add_argument_separator()
        self.add_argument_format()
        self.add_argument_top()
        self.add_argument_weighted()
        self.add_argument_quantiles()
        self.add_argument_data()

    def __call__(self, args: argparse.Namespace):
        data = dinf.load_results(args.data)

        names = list(data.dtype.names)
        probs = data["_Pr"]
        thetas = structured_to_unstructured(data[names[1:]])

        if args.top is not None:
            thetas, probs = dinf.Parameters.top_n(thetas, probs=probs, n=args.top)

        q = [float(qj) for qj in args.quantiles.split(",")]
        weights = probs if args.weighted else None

        with _open_default(args.output_file) as f:
            print("Param", *q, sep=args.separator, file=f)
            for j, name in enumerate(names[1:]):
                xq = quantile(thetas[..., j], q=q, weights=weights).tolist()
                if args.format is not None:
                    xq = [args.format % v for v in xq]
                print(name, *xq, sep=args.separator, file=f)


def main(args_list=None):
    top_parser = argparse.ArgumentParser(
        prog="dinf.tabulate", description="Tabulate Dinf output."
    )
    top_parser.add_argument(
        "-V", "--version", action="version", version=dinf.__version__
    )

    subparsers = top_parser.add_subparsers(dest="subcommand")
    _Metrics(subparsers)
    _Data(subparsers)
    _Quantiles(subparsers)

    args = top_parser.parse_args(args_list)
    if args.subcommand is None:
        top_parser.print_help()
        exit(1)

    _set_loglevel(args.quiet, args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
