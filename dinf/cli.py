from __future__ import annotations
import argparse
import inspect
import os
import pathlib

import dinf


def check_output_file(path):
    """
    Check the file is writable and doesn't already exist.

    We do this check to ensure the user doesn't spend a lot of compute
    time on simulations, etc., only to find their results couldn't be
    saved due to trivial problems like a typo in the filename.

    Writing to the file could fail for lots of reasons, e.g. permission
    denied, path is a directory, parent directory doesn't exist, etc.
    Checking each of the possibilities is not practical, so we just
    try writing to the file. Other problems could occur later when writing
    data to the file, such as a full disk, or network filesystem
    inaccessibility, but we should catch the most common problems early.
    """
    if path.exists():
        if path.samefile(os.devnull) or path.is_fifo():
            # No problem!
            return
        raise ValueError(f"{path}: output file already exists, refusing to overwrite")
    path.touch()
    path.unlink()


_DINF_MODEL_HELP = (
    'Python script from which to import the variable "dinf_model". '
    "This is a dinf.DinfModel object that describes the model components. "
    "See the examples/ folder of the git repository for example models. "
    "https://github.com/RacimoLab/dinf"
)


class ADRDFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


class _SubCommand:
    """
    Base class for subcommands.
    """

    def __init__(self, subparsers, command):
        docstring = inspect.getdoc(self)
        self.parser = subparsers.add_parser(
            command,
            help=docstring.splitlines()[0],
            description=docstring,
            formatter_class=ADRDFormatter,
        )
        self.parser.set_defaults(func=self)

    def add_common_parser_group(self):
        group = self.parser.add_argument_group(title="common arguments")
        group.add_argument(
            "-S",
            "--seed",
            type=int,
            help=(
                "Seed for the random number generator. "
                "CPU-based training is expected to produce deterministic results. "
                "Results may differ between CPU and GPU trained networks for the "
                "same seed value. Also note that operations on a GPU are not "
                "fully determinstic, so training or applying a neural network "
                "twice with the same seed value will not produce identical results."
            ),
        )
        group.add_argument(
            "-j",
            "--parallelism",
            type=int,
            help=(
                "Number of processes to use for parallelising calls to the "
                "DinfModel's generator_func and target_func. "
                "If not specified, all CPU cores will be used. "
                "The number of cores used for CPU-based neural networks "
                "is not set with this parameter---instead use the"
                "`taskset` command. See "
                "https://github.com/google/jax/issues/1539"
            ),
        )

    def add_train_parser_group(self):
        group = self.parser.add_argument_group(title="training arguments")
        group.add_argument(
            "-r",
            "--training-replicates",
            type=int,
            default=1000,
            help=("Size of the dataset used to train the discriminator."),
        )
        group.add_argument(
            "-R",
            "--test-replicates",
            type=int,
            default=1000,
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

    def add_gan_parser_group(self):
        group = self.parser.add_argument_group(title="GAN arguments")
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
            "model",
            metavar="model.py",
            type=pathlib.Path,
            help=_DINF_MODEL_HELP,
        )


class AbcGan(_SubCommand):
    """
    Adversarial Abstract Bayesian Computation.

    Conceptually, the GAN takes the following steps for iteration j:

      - sample train/test datasets from the prior[j] distribution,
      - train the discriminator,
      - make predictions with the discriminator,
      - construct a posterior[j] sample from the test dataset,
      - set prior[j+1] = posterior[j].

    In the first iteration, the parameter values given to the generator
    to produce the test/train datasets are drawn from the parameters'
    prior distribution. In subsequent iterations, the parameter values
    are drawn from a posterior ABC sample. The posterior is obtained by
    weighting the prior distribution by the discriminator predictions,
    followed by gaussian smoothing.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "abc-gan")
        self.add_common_parser_group()
        self.add_train_parser_group()

        group = self.parser.add_argument_group("ABC arguments")
        group.add_argument(
            "-n",
            "--top-n",
            metavar="N",
            type=int,
            help=(
                "In each iteraction, accept only the N top samples, "
                "ranked by probability."
            ),
        )

        self.add_gan_parser_group()

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)
        dinf.dinf.abc_gan(
            dinf_model=dinf_model,
            iterations=args.iterations,
            training_replicates=args.training_replicates,
            test_replicates=args.test_replicates,
            epochs=args.epochs,
            working_directory=args.working_directory,
            parallelism=args.parallelism,
            seed=args.seed,
        )


class AlfiMcmcGan(_SubCommand):
    """
    Run the ALFI MCMC GAN.

    This is an MCMC GAN with a surrogate network, as described in
    Kim et al. 2020, https://arxiv.org/abs/2004.05803v1
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "alfi-mcmc-gan")

        self.add_common_parser_group()
        self.add_train_parser_group()

        group = self.parser.add_argument_group("MCMC arguments")
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

        self.add_gan_parser_group()

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)
        dinf.alfi_mcmc_gan(
            dinf_model=dinf_model,
            iterations=args.iterations,
            training_replicates=args.training_replicates,
            test_replicates=args.test_replicates,
            epochs=args.epochs,
            walkers=args.walkers,
            steps=args.steps,
            working_directory=args.working_directory,
            parallelism=args.parallelism,
            seed=args.seed,
        )


class McmcGan(_SubCommand):
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
        super().__init__(subparsers, "mcmc-gan")

        self.add_common_parser_group()
        self.add_train_parser_group()

        group = self.parser.add_argument_group("MCMC arguments")
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

        self.add_gan_parser_group()

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)
        dinf.mcmc_gan(
            dinf_model=dinf_model,
            iterations=args.iterations,
            training_replicates=args.training_replicates,
            test_replicates=args.test_replicates,
            epochs=args.epochs,
            walkers=args.walkers,
            steps=args.steps,
            Dx_replicates=args.Dx_replicates,
            working_directory=args.working_directory,
            parallelism=args.parallelism,
            seed=args.seed,
        )


class PgGan(_SubCommand):
    """
    Run PG-GAN style simulated annealing.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "pg-gan")

        self.add_common_parser_group()
        self.add_train_parser_group()

        group = self.parser.add_argument_group("PG-GAN arguments")
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

        self.add_gan_parser_group()

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)
        dinf.pg_gan(
            dinf_model=dinf_model,
            iterations=args.iterations,
            training_replicates=args.training_replicates,
            test_replicates=args.test_replicates,
            epochs=args.epochs,
            Dx_replicates=args.Dx_replicates,
            num_proposals=args.num_proposals,
            max_pretraining_iterations=args.max_pretraining_iterations,
            working_directory=args.working_directory,
            parallelism=args.parallelism,
            seed=args.seed,
        )


class Train(_SubCommand):
    """
    Train a discriminator.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "train")

        self.add_common_parser_group()
        self.add_train_parser_group()

        group = self.parser.add_argument_group()
        group.add_argument(
            "model",
            metavar="model.py",
            type=pathlib.Path,
            help=_DINF_MODEL_HELP,
        )
        group.add_argument(
            "discriminator_file",
            metavar="discriminator.nn",
            type=pathlib.Path,
            help="Output file where the discriminator will be saved.",
        )

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)
        if args.epochs > 0:
            check_output_file(args.discriminator_file)
        discriminator = dinf.train(
            dinf_model=dinf_model,
            training_replicates=args.training_replicates,
            test_replicates=args.test_replicates,
            epochs=args.epochs,
            parallelism=args.parallelism,
            seed=args.seed,
        )
        if args.epochs > 0:
            discriminator.to_file(args.discriminator_file)


class Predict(_SubCommand):
    """
    Make predictions using a trained discriminator.

    By default, features will be obtained by sampling replicates from
    the generator (using parameters from the prior distribution).
    To instead sample features from the target dataset, use the
    --target option.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "predict")
        self.add_common_parser_group()

        group = self.parser.add_argument_group(title="predict arguments")
        group.add_argument(
            "-r",
            "--replicates",
            type=int,
            default=1000,
            help=(
                "Number of theta replicates to generate and predict with "
                "the discriminator. "
            ),
        )

        self.parser.add_argument(
            "--target",
            action="store_true",
            help="Sample features from the target dataset.",
        )

        group = self.parser.add_argument_group()
        group.add_argument(
            "model",
            metavar="model.py",
            type=pathlib.Path,
            help=_DINF_MODEL_HELP,
        )
        group.add_argument(
            "discriminator_file",
            metavar="discriminator.nn",
            type=pathlib.Path,
            help="Discriminator to use for predictions.",
        )
        group.add_argument(
            "output_file",
            metavar="output.npz",
            type=pathlib.Path,
            help="Output data, matching thetas to discriminator predictions.",
        )

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)
        discriminator = dinf.Discriminator(
            dinf_model.feature_shape, network=dinf_model.discriminator_network
        ).from_file(args.discriminator_file)
        check_output_file(args.output_file)
        thetas, probs = dinf.predict(
            discriminator=discriminator,
            dinf_model=dinf_model,
            replicates=args.replicates,
            sample_target=args.target,
            parallelism=args.parallelism,
            seed=args.seed,
        )
        dinf.save_results(
            args.output_file,
            thetas=thetas,
            probs=probs,
            parameters=dinf_model.parameters,
        )


class Check(_SubCommand):
    """
    Basic dinf_model health checks.

    Checks that the target and generator functions work and return the
    same feature shapes.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "check")

        self.parser.add_argument(
            "model",
            metavar="model.py",
            type=pathlib.Path,
            help=_DINF_MODEL_HELP,
        )

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)
        dinf_model.check()


def main(args_list=None):
    top_parser = argparse.ArgumentParser(
        prog="dinf",
        description="Discriminator-based inference of population parameters.",
    )
    top_parser.add_argument("--version", action="version", version=dinf.__version__)

    subparsers = top_parser.add_subparsers(
        dest="subcommand",  # metavar="{check,abc-gan,alfi-mcmc-gan,mcmc-gan,pg-gan}"
    )
    Check(subparsers)
    Train(subparsers)
    Predict(subparsers)

    AbcGan(subparsers)
    AlfiMcmcGan(subparsers)
    McmcGan(subparsers)
    PgGan(subparsers)

    args = top_parser.parse_args(args_list)
    if args.subcommand is None:
        top_parser.print_help()
        exit(1)

    args.func(args)
