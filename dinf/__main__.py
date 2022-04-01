from __future__ import annotations
import argparse
import pathlib
import textwrap

import numpy as np

import dinf


def _add_common_parser_group(parser):
    group = parser.add_argument_group(title="common arguments")
    group.add_argument(
        "-S", "--seed", type=int, help="Seed for the random number generator."
    )
    group.add_argument(
        "-j",
        "--parallelism",
        type=int,
        help=(
            "Number of processes to use for parallelising calls to the "
            "Genobuilder's generator_func and target_func."
        ),
    )


def _add_train_parser_group(parser):
    group = parser.add_argument_group(title="training arguments")
    group.add_argument(
        "-r",
        "--training-replicates",
        type=int,
        default=1_000_000,
        help=(
            "Size of the dataset used to train the discriminator. "
            "This dataset is constructed once each GAN iteration."
        ),
    )
    group.add_argument(
        "-R",
        "--test-replicates",
        type=int,
        default=1_000,
        help=(
            "Size of the test dataset used to evaluate the discriminator "
            "after each training epoch."
        ),
    )
    group.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1,
        help=(
            "Number of full passes over the training dataset when training "
            "the discriminator."
        ),
    )


_GENOB_MODEL_HELP = (
    'Python script from which to import the variable "genobuilder". '
    "This is a dinf.Genobuilder object that describes the GAN. "
    "See the examples/ folder for example models."
)


def _add_gan_parser_group(parser):
    group = parser.add_argument_group(title="GAN arguments")
    group.add_argument(
        "-i", "--iterations", type=int, default=1, help="Number of GAN iterations."
    )
    group.add_argument(
        "-d",
        "--working-directory",
        type=str,
        help=(
            "Folder to output results. If not specified, the current "
            "directory will be used."
        ),
    )
    group.add_argument(
        "genob_model",
        metavar="user_model.py",
        type=pathlib.Path,
        help=_GENOB_MODEL_HELP,
    )


class ADRDFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


class AbcGan:
    """
    Run the ABC GAN.

    Each iteration of the GAN can be conceptually divided into:
    - constructing train/test datasets for the discriminator,
    - training the discriminator for a certain number of epochs,
    - running the ABC.

    In the first iteration, the parameter values given to the generator
    to produce the test/train datasets are drawn from the parameters' prior
    distribution. In subsequent iterations, the parameter values are drawn
    by sampling with replacement from the previous iteration's ABC posterior.
    """

    def __init__(self, subparsers):
        parser = subparsers.add_parser(
            "abc-gan",
            help="Run the ABC GAN",
            description=textwrap.dedent(self.__doc__),
            formatter_class=ADRDFormatter,
        )
        parser.set_defaults(func=self)

        _add_common_parser_group(parser)
        _add_train_parser_group(parser)

        group = parser.add_argument_group("ABC arguments")
        group.add_argument(
            "-p",
            "--proposals",
            type=int,
            default=1_000_000,
            help="Number of ABC sample draws.",
        )
        group.add_argument(
            "-P",
            "--posteriors",
            type=int,
            default=1000,
            help="Number of top-ranked ABC sample draws to keep.",
        )

        _add_gan_parser_group(parser)

    def __call__(self, args: argparse.Namespace):
        rng = np.random.default_rng(args.seed)
        genobuilder = dinf.Genobuilder._from_file(args.genob_model)
        dinf.dinf.abc_gan(
            genobuilder=genobuilder,
            iterations=args.iterations,
            training_replicates=args.training_replicates,
            test_replicates=args.test_replicates,
            epochs=args.epochs,
            proposals=args.proposals,
            posteriors=args.posteriors,
            working_directory=args.working_directory,
            parallelism=args.parallelism,
            rng=rng,
        )


class AlfiMcmcGan:
    """
    Run the ALFI MCMC GAN.

    This is an MCMC GAN with a surrogate network, as described in
    Kim et al. 2020, https://arxiv.org/abs/2004.05803v1
    """

    def __init__(self, subparsers):
        parser = subparsers.add_parser(
            "alfi-mcmc-gan",
            help="Run the ALFI MCMC GAN",
            description=textwrap.dedent(self.__doc__),
            formatter_class=ADRDFormatter,
        )
        parser.set_defaults(func=self)

        _add_common_parser_group(parser)
        _add_train_parser_group(parser)

        group = parser.add_argument_group("MCMC arguments")
        group.add_argument(
            "-w",
            "--walkers",
            type=int,
            default=64,
            help="Number of independent MCMC chains.",
        )
        group.add_argument(
            "-s",
            "--steps",
            type=int,
            default=1000,
            help="The chain length for each MCMC walker.",
        )

        _add_gan_parser_group(parser)

    def __call__(self, args: argparse.Namespace):
        rng = np.random.default_rng(args.seed)
        genobuilder = dinf.Genobuilder._from_file(args.genob_model)
        dinf.alfi_mcmc_gan(
            genobuilder=genobuilder,
            iterations=args.iterations,
            training_replicates=args.training_replicates,
            test_replicates=args.test_replicates,
            epochs=args.epochs,
            walkers=args.walkers,
            steps=args.steps,
            working_directory=args.working_directory,
            parallelism=args.parallelism,
            rng=rng,
        )


class McmcGan:
    """
    Run the MCMC GAN.

    Each iteration of the GAN can be conceptually divided into:
    - constructing train/test datasets for the discriminator,
    - training the discriminator for a certain number of epochs,
    - running the MCMC.

    In the first iteration, the parameter values given to the generator
    to produce the test/train datasets are drawn from the parameters' prior
    distribution. In subsequent iterations, the parameter values are drawn
    by sampling with replacement from the previous iteration's MCMC chains.
    """

    def __init__(self, subparsers):
        parser = subparsers.add_parser(
            "mcmc-gan",
            help="Run the MCMC GAN",
            description=textwrap.dedent(self.__doc__),
            formatter_class=ADRDFormatter,
        )
        parser.set_defaults(func=self)

        _add_common_parser_group(parser)
        _add_train_parser_group(parser)

        group = parser.add_argument_group("MCMC arguments")
        group.add_argument(
            "-w",
            "--walkers",
            type=int,
            default=64,
            help="Number of independent MCMC chains.",
        )
        group.add_argument(
            "-s",
            "--steps",
            type=int,
            default=1000,
            help="The chain length for each MCMC walker.",
        )
        group.add_argument(
            "--Dx-replicates",
            type=int,
            default=64,
            help="Number of generator replicates for approximating E[D(x)|θ].",
        )

        _add_gan_parser_group(parser)

    def __call__(self, args: argparse.Namespace):
        rng = np.random.default_rng(args.seed)
        genobuilder = dinf.Genobuilder._from_file(args.genob_model)
        dinf.mcmc_gan(
            genobuilder=genobuilder,
            iterations=args.iterations,
            training_replicates=args.training_replicates,
            test_replicates=args.test_replicates,
            epochs=args.epochs,
            walkers=args.walkers,
            steps=args.steps,
            Dx_replicates=args.Dx_replicates,
            working_directory=args.working_directory,
            parallelism=args.parallelism,
            rng=rng,
        )


class PgGan:
    """
    Run PG-GAN style simulated annealing.
    """

    def __init__(self, subparsers):
        parser = subparsers.add_parser(
            "pg-gan",
            help="Run PG-GAN style simulated annealing",
            description=textwrap.dedent(self.__doc__),
            formatter_class=ADRDFormatter,
        )
        parser.set_defaults(func=self)

        _add_common_parser_group(parser)
        _add_train_parser_group(parser)

        group = parser.add_argument_group("PG-GAN arguments")
        group.add_argument(
            "--Dx-replicates",
            type=int,
            default=64,
            help="Number of generator replicates for approximating E[D(x)|θ].",
        )
        group.add_argument(
            "--num-proposals",
            type=int,
            default=10,
            help="Number of proposals for each parameter in a given iteration.",
        )
        group.add_argument(
            "--max-pretraining-iterations",
            type=int,
            default=100,
            help="Maximum number of pretraining rounds.",
        )

        _add_gan_parser_group(parser)

    def __call__(self, args: argparse.Namespace):
        rng = np.random.default_rng(args.seed)
        genobuilder = dinf.Genobuilder._from_file(args.genob_model)
        dinf.pg_gan(
            genobuilder=genobuilder,
            iterations=args.iterations,
            training_replicates=args.training_replicates,
            test_replicates=args.test_replicates,
            epochs=args.epochs,
            Dx_replicates=args.Dx_replicates,
            num_proposals=args.num_proposals,
            max_pretraining_iterations=args.max_pretraining_iterations,
            working_directory=args.working_directory,
            parallelism=args.parallelism,
            rng=rng,
        )


class Check:
    """
    Check a genobuilder object by calling the target and generator functions.
    """

    def __init__(self, subparsers):
        parser = subparsers.add_parser(
            "check",
            help="Basic genobuilder health checks",
            description=textwrap.dedent(self.__doc__),
            formatter_class=ADRDFormatter,
        )
        parser.set_defaults(func=self)

        parser.add_argument(
            "genob_model",
            metavar="user_model.py",
            type=pathlib.Path,
            help=_GENOB_MODEL_HELP,
        )

    def __call__(self, args: argparse.Namespace):
        genobuilder = dinf.Genobuilder._from_file(args.genob_model)
        genobuilder.check()


def main(args_list=None):
    top_parser = argparse.ArgumentParser(
        prog="dinf",
        description="Discriminator-based inference of population parameters.",
    )
    top_parser.add_argument("--version", action="version", version=dinf.__version__)

    subparsers = top_parser.add_subparsers(
        dest="subcommand", metavar="{check,abc-gan,alfi-mcmc-gan,mcmc-gan,pg-gan}"
    )
    Check(subparsers)
    AbcGan(subparsers)
    AlfiMcmcGan(subparsers)
    McmcGan(subparsers)
    PgGan(subparsers)

    args = top_parser.parse_args(args_list)
    if args.subcommand is None:
        top_parser.print_help()
        exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
