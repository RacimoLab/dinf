import argparse
import pathlib

import numpy as np

from . import dinf, feature_extractor, models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discriminator-based inference of population parameters."
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    train_parser = subparsers.add_parser(
        "train",
        help="Train discriminator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    abc_parser = subparsers.add_parser(
        "abc", help="ABC", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opt_parser = subparsers.add_parser(
        "opt",
        help="gradient-free optimisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mcmc_parser = subparsers.add_parser(
        "mcmc", help="MCMC", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    for p in (train_parser, abc_parser, opt_parser, mcmc_parser):
        p.add_argument(
            "-s",
            "--seed",
            type=int,
            help="Seed for the random number generator",
        )
        p.add_argument(
            "-j",
            "--parallelism",
            type=int,
            help="Number of processes for simulation",
        )
        p.add_argument(
            "-n",
            "--num-samples",
            default=128,
            type=int,
            help="Number of samples (haplotypes)",
        )
        p.add_argument(
            "-m",
            # TODO rename this
            "--fixed-dimension",
            default=128,
            type=int,
            help="Number of haplotype bins to resize genotype matrix",
        )
        p.add_argument(
            "-l",
            "--sequence-length",
            default=1_000_000,
            type=int,
            help="Sequence length",
        )
        p.add_argument(
            "-a",
            "--maf-thresh",
            type=float,
            default=0.05,
            help="Ignore SNPs with minor allele frequency lower than this value.",
        )
        p.add_argument(
            "-o",
            "--output-directory",
            default=pathlib.Path("."),
            type=pathlib.Path,
            help="Directory for output (cache, results and reports)",
        )
        p.add_argument(
            "discriminator_filename",
            type=str,
            help="Filename for trained descriminator neural network",
        )

    # train
    train_parser.add_argument(
        "-e",
        "--training-epochs",
        default=1,
        type=int,
        help="Number of epochs to train the discriminator",
    )
    train_parser.add_argument(
        "-r",
        "--num-replicates",
        type=int,
        default=10_000,
        help=(
            "Number of sample replicates for each training class."
            "One class is produced by the generator using the prior "
            "on the parameters, the other class is drawn from observed data."
        ),
    )
    train_parser.add_argument(
        "-V",
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Proportion of replicates used for training validation.",
    )

    # abc
    abc_parser.add_argument(
        "-r",
        "--num-replicates",
        type=int,
        default=10_000,
        help="Number of replicate simulations from the prior distribution.",
    )

    # opt
    opt_parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for optimisation.",
    )
    opt_parser.add_argument(
        "--num-Dx-replicates",
        type=int,
        default=32,
        help="Number of simulation replicates to approximate E[D(x)|θ]",
    )

    # mcmc
    mcmc_parser.add_argument(
        "-w",
        "--walkers",
        type=int,
        default=16,
        help="Number of walkers. See zeus-mcmc documentation.",
    )
    mcmc_parser.add_argument(
        "-S",
        "--steps",
        type=int,
        default=1000,
        help="Number of steps for each walker. See zeus-mcmc documentation.",
    )
    mcmc_parser.add_argument(
        "--num-Dx-replicates",
        type=int,
        default=32,
        help="Number of simulation replicates to approximate E[D(x)|θ]",
    )

    return parser.parse_args()


def cli():
    args = parse_args()
    # logger_setup(args.verbose)
    rng = np.random.default_rng(args.seed)

    args.output_directory.mkdir(parents=True, exist_ok=True)

    bh_matrix = feature_extractor.BinnedHaplotypeMatrix(
        num_samples=args.num_samples,
        fixed_dimension=args.fixed_dimension,
        maf_thresh=args.maf_thresh,
    )

    generator = models.Bottleneck(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        feature_extractor=bh_matrix,
    )

    if args.subcommand == "train":
        dinf.train(
            generator=generator,
            discriminator_filename=args.discriminator_filename,
            num_replicates=args.num_replicates,
            parallelism=args.parallelism,
            validation_ratio=args.validation_ratio,
            training_epochs=args.training_epochs,
            rng=rng,
        )
    elif args.subcommand == "abc":
        dinf.abc(
            generator=generator,
            discriminator_filename=args.discriminator_filename,
            num_replicates=args.num_replicates,
            parallelism=args.parallelism,
            working_directory=args.output_directory,
            rng=rng,
        )
    elif args.subcommand == "opt":
        dinf.opt(
            generator=generator,
            discriminator_filename=args.discriminator_filename,
            parallelism=args.parallelism,
            working_directory=args.output_directory,
            iterations=args.iterations,
            num_Dx_replicates=args.num_Dx_replicates,
            rng=rng,
        )
    elif args.subcommand == "mcmc":
        dinf.mcmc(
            generator=generator,
            discriminator_filename=args.discriminator_filename,
            parallelism=args.parallelism,
            working_directory=args.output_directory,
            walkers=args.walkers,
            steps=args.steps,
            num_Dx_replicates=args.num_Dx_replicates,
            rng=rng,
        )
    else:
        raise AssertionError


if __name__ == "__main__":
    cli()
