from __future__ import annotations
import argparse
import inspect
import logging
import os
import pathlib

import rich.logging
import rich.progress

import dinf


def _check_output_file(path):
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


class _ADRDFormatter(
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
            formatter_class=_ADRDFormatter,
        )
        self.parser.set_defaults(func=self)

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help=(
                "Increase verbosity. Specify once for INFO messages and "
                "twice for DEBUG messages."
            ),
        )
        group.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            help="Disable output. Only ERROR and CRITICAL messages are printed.",
        )

    def add_argument_model(self, *, parser=None, required=True):
        if parser is None:
            parser = self.parser
        parser.add_argument(
            "-m",
            "--model",
            metavar="model.py",
            required=required,
            type=pathlib.Path,
            help=_DINF_MODEL_HELP,
        )


class _DinfSubCommand(_SubCommand):
    """
    Base class for `dinf` subcommands.
    """

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
            "-o",
            "--output-folder",
            type=str,
            help=(
                "Folder to output results. If not specified, the current "
                "directory will be used."
            ),
        )
        self.add_argument_model(parser=group)

    def add_argument_discriminator(self, *, parser=None, help=None):
        if parser is None:
            parser = self.parser
        if help is None:
            help = "File containing discriminator network weights."
        parser.add_argument(
            "-d",
            "--discriminator",
            metavar="discriminator.nn",
            required=True,
            type=pathlib.Path,
            help=help,
        )


class _AbcGan(_DinfSubCommand):
    """
    Adversarial Abstract Bayesian Computation / Sequential Monte Carlo.

    Conceptually, the GAN takes the following steps for iteration j:

      - sample training and proposal datasets from the prior[j] distribution,
      - train the discriminator,
      - make predictions with the discriminator on the proposal dataset,
      - construct a posterior[j] sample from the proposal dataset,
      - set prior[j+1] = posterior[j].

    In the first iteration, the parameter values given to the generator
    to produce the train/proposal datasets are drawn from the parameters'
    prior distribution. In subsequent iterations, the parameter values
    are drawn from a posterior ABC sample. The posterior is obtained by
    rejection sampling the proposal distribution and weighting the posterior
    by the discriminator predictions, followed by gaussian smoothing.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "abc-gan")
        self.add_common_parser_group()
        self.add_train_parser_group()

        group = self.parser.add_argument_group("ABC arguments")
        group.add_argument(
            "--top",
            metavar="N",
            type=int,
            help=(
                "In each iteration, accept only the N top proposals, "
                "ranked by probability."
            ),
        )
        group.add_argument(
            "-P",
            "--proposal-replicates",
            type=int,
            default=1000,
            help="Number of replicates for ABC proposals.",
        )

        self.add_gan_parser_group()

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)

        progress = rich.progress.Progress(
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TextColumn(
                "[progress.percentage]{task.completed}/{task.total}"
            ),
            rich.progress.TimeRemainingColumn(),
            rich.progress.TextColumn("{task.fields[_loss]}"),
            rich.progress.TextColumn("{task.fields[_accuracy]}"),
        )

        def cb_counter(description):
            task_id = task_ids[description]

            def cb(n):
                progress.update(task_id, visible=True)
                if n < progress.tasks[task_id].completed:
                    progress.reset(task_id)
                if n > 0:
                    progress.advance(task_id)

            return cb

        def cb_iteration(description):
            task_id = task_ids[description]

            def cb(n):
                if not progress.tasks[task_id].visible:
                    # Set total iterations in the 'resume' case.
                    progress.reset(task_id, total=n + args.iterations)
                progress.update(task_id, completed=n, visible=True)

            return cb

        def cb_batch(description):
            task_id = task_ids[description]

            def cb(n, loss, accuracy):
                if n < progress.tasks[task_id].completed:
                    progress.reset(task_id)
                progress.update(
                    task_id,
                    completed=n,
                    visible=True,
                    _loss=f"loss {loss:.4f}",
                    _accuracy=f"accuracy {accuracy:.4f}",
                )

            return cb

        def cb_predict_batch(description):
            task_id = task_ids[description]

            def cb(n):
                if n < progress.tasks[task_id].completed:
                    progress.reset(task_id)
                progress.update(
                    task_id,
                    completed=n,
                    visible=True,
                )

            return cb

        n_test = args.test_replicates // 2
        n_train = args.training_replicates // 2
        task_ids = {}
        callbacks = {}

        for description, total, cb_name, cb_func in [
            ("Generator/test", n_test, "test/generator/feature", cb_counter),
            ("Target/test", n_test, "test/target/feature", cb_counter),
            ("Iteration", args.iterations, "iteration", cb_iteration),
            (" Generator/train", n_train, "train/generator/feature", cb_counter),
            (" Target/train", n_train, "train/target/feature", cb_counter),
            (" Epoch", args.epochs, "fit/epoch", cb_counter),
            ("  Train", args.training_replicates, "fit/train_batch", cb_batch),
            ("  Test", args.test_replicates, "fit/test_batch", cb_batch),
            (
                " Generator/proposal",
                args.proposal_replicates,
                "proposal/feature",
                cb_counter,
            ),
            ("Predict", args.proposal_replicates, "predict/batch", cb_predict_batch),
        ]:
            task_ids[description] = progress.add_task(
                description,
                total=total,
                visible=False,
                _loss="",
                _accuracy="",
            )
            callbacks[cb_name] = cb_func(description)

        if args.quiet:
            callbacks = {}

        with progress:
            dinf.dinf.abc_gan(
                dinf_model=dinf_model,
                iterations=args.iterations,
                training_replicates=args.training_replicates,
                test_replicates=args.test_replicates,
                proposal_replicates=args.proposal_replicates,
                epochs=args.epochs,
                top_n=args.top,
                output_folder=args.output_folder,
                parallelism=args.parallelism,
                seed=args.seed,
                callbacks=callbacks,
            )


class _McmcGan(_DinfSubCommand):
    """
    Run the MCMC GAN.

    Conceptually, the GAN takes the following steps for iteration j:

      - sample training dataset from the prior[j] distribution,
      - train the discriminator,
      - run the MCMC,
      - obtain posterior[j] as weighted KDE of MCMC sample,
      - set prior[j+1] = posterior[j].

    In the first iteration, the parameter values given to the generator
    to produce the training dataset are drawn from the parameters' prior
    distribution. In subsequent iterations, the parameter values are drawn
    from a weighted gaussian KDE of the previous iteration's MCMC chains.
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
            default=32,
            help="Number of generator replicates for approximating E[D(x)|θ].",
        )

        self.add_gan_parser_group()

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)

        progress = rich.progress.Progress(
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TextColumn(
                "[progress.percentage]{task.completed}/{task.total}"
            ),
            rich.progress.TimeRemainingColumn(),
            rich.progress.TextColumn("{task.fields[_loss]}"),
            rich.progress.TextColumn("{task.fields[_accuracy]}"),
        )

        def cb_counter(description):
            task_id = task_ids[description]

            def cb(n):
                progress.update(task_id, visible=True)
                if n < progress.tasks[task_id].completed:
                    progress.reset(task_id)
                if n > 0:
                    progress.advance(task_id)

            return cb

        def cb_iteration(description):
            task_id = task_ids[description]

            def cb(n):
                if not progress.tasks[task_id].visible:
                    # Set total iterations in the 'resume' case.
                    progress.reset(task_id, total=n + args.iterations)
                progress.update(task_id, completed=n, visible=True)

            return cb

        def cb_batch(description):
            task_id = task_ids[description]

            def cb(n, loss, accuracy):
                if n < progress.tasks[task_id].completed:
                    progress.reset(task_id)
                progress.update(
                    task_id,
                    completed=n,
                    visible=True,
                    _loss=f"loss {loss:.4f}",
                    _accuracy=f"accuracy {accuracy:.4f}",
                )

            return cb

        n_test = args.test_replicates // 2
        n_train = args.training_replicates // 2
        task_ids = {}
        callbacks = {}

        for description, total, cb_name, cb_func in [
            ("Generator/test", n_test, "test/generator/feature", cb_counter),
            ("Target/test", n_test, "test/target/feature", cb_counter),
            ("Iteration", args.iterations, "iteration", cb_iteration),
            (" Generator/train", n_train, "train/generator/feature", cb_counter),
            (" Target/train", n_train, "train/target/feature", cb_counter),
            (" Epoch", args.epochs, "fit/epoch", cb_counter),
            ("  Train", args.training_replicates, "fit/train_batch", cb_batch),
            ("  Test", args.test_replicates, "fit/test_batch", cb_batch),
            (" MCMC", args.steps, "mcmc", cb_counter),
        ]:
            task_ids[description] = progress.add_task(
                description,
                total=total,
                visible=False,
                _loss="",
                _accuracy="",
            )
            callbacks[cb_name] = cb_func(description)

        if args.quiet:
            callbacks = {}

        with progress:
            dinf.mcmc_gan(
                dinf_model=dinf_model,
                iterations=args.iterations,
                training_replicates=args.training_replicates,
                test_replicates=args.test_replicates,
                epochs=args.epochs,
                walkers=args.walkers,
                steps=args.steps,
                Dx_replicates=args.Dx_replicates,
                output_folder=args.output_folder,
                parallelism=args.parallelism,
                seed=args.seed,
                callbacks=callbacks,
            )


class _PgGan(_DinfSubCommand):
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
            default=32,
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
            output_folder=args.output_folder,
            parallelism=args.parallelism,
            seed=args.seed,
        )


class _Train(_DinfSubCommand):
    """
    Train a discriminator.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "train")

        self.add_common_parser_group()
        self.add_train_parser_group()

        group = self.parser.add_argument_group()
        self.add_argument_model(parser=group)
        self.add_argument_discriminator(
            parser=group,
            help="Output file where the discriminator will be saved.",
        )

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)
        if args.epochs > 0:
            _check_output_file(args.discriminator)

        progress = rich.progress.Progress(
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TextColumn(
                "[progress.percentage]{task.completed}/{task.total}"
            ),
            rich.progress.TimeRemainingColumn(),
            rich.progress.TextColumn("{task.fields[_loss]}"),
            rich.progress.TextColumn("{task.fields[_accuracy]}"),
        )

        task_ids = {
            description: progress.add_task(
                description,
                total=total,
                visible=False,
                _loss="",
                _accuracy="",
            )
            for description, total in [
                ("Generator/train", args.training_replicates // 2),
                ("Target/train", args.training_replicates // 2),
                ("Generator/test", args.test_replicates // 2),
                ("Target/test", args.test_replicates // 2),
                ("Epoch", args.epochs),
                (" Train", args.training_replicates),
                (" Test", args.test_replicates),
            ]
        }

        def cb_counter(description):
            task_id = task_ids[description]

            def cb(n):
                progress.update(task_id, visible=True)
                if n > 0:
                    progress.advance(task_id)

            return cb

        def cb_batch(description):
            task_id = task_ids[description]

            def cb(n, loss, accuracy):
                if n < progress.tasks[task_id].completed:
                    progress.reset(task_id)
                progress.update(
                    task_id,
                    completed=n,
                    visible=True,
                    _loss=f"loss {loss:.4f}",
                    _accuracy=f"accuracy {accuracy:.4f}",
                )

            return cb

        callbacks = {
            "train/generator/feature": cb_counter("Generator/train"),
            "train/target/feature": cb_counter("Target/train"),
            "test/generator/feature": cb_counter("Generator/test"),
            "test/target/feature": cb_counter("Target/test"),
            "discriminator/fit/epoch": cb_counter("Epoch"),
            "discriminator/fit/train_batch": cb_batch(" Train"),
            "discriminator/fit/test_batch": cb_batch(" Test"),
        }

        if args.quiet:
            callbacks = {}

        with progress:
            discriminator = dinf.train(
                dinf_model=dinf_model,
                training_replicates=args.training_replicates,
                test_replicates=args.test_replicates,
                epochs=args.epochs,
                parallelism=args.parallelism,
                seed=args.seed,
                callbacks=callbacks,
            )

        if args.epochs > 0:
            discriminator.to_file(args.discriminator)


class _Predict(_DinfSubCommand):
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
        self.add_argument_model(parser=group)
        self.add_argument_discriminator(parser=group)
        group.add_argument(
            "-o",
            "--output-file",
            metavar="output.npz",
            type=pathlib.Path,
            help="Output data, matching thetas to discriminator predictions.",
        )

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)
        discriminator = dinf.Discriminator.from_file(
            args.discriminator, network=dinf_model.discriminator_network
        )
        _check_output_file(args.output_file)

        progress = rich.progress.Progress(
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TextColumn(
                "[progress.percentage]{task.completed}/{task.total}"
            ),
            rich.progress.TimeRemainingColumn(),
        )

        task_ids = {
            description: progress.add_task(description, total=total, visible=False)
            for description, total in [
                ("Generator", args.replicates),
                ("Target", args.replicates),
                ("Predict", args.replicates),
            ]
        }

        def cb_counter(description):
            task_id = task_ids[description]

            def cb(n):
                progress.update(task_id, visible=True)
                if n > 0:
                    progress.advance(task_id)

            return cb

        def cb_batch(description):
            task_id = task_ids[description]

            def cb(n):
                if n < progress.tasks[task_id].completed:
                    progress.reset(task_id)
                progress.update(task_id, completed=n, visible=True)

            return cb

        callbacks = {
            "predict/generator/feature": cb_counter("Generator"),
            "predict/target/feature": cb_counter("Target"),
            "discriminator/predict/batch": cb_batch("Predict"),
        }

        if args.quiet:
            callbacks = {}

        with progress:
            thetas, probs = dinf.predict(
                discriminator=discriminator,
                dinf_model=dinf_model,
                replicates=args.replicates,
                sample_target=args.target,
                parallelism=args.parallelism,
                seed=args.seed,
                callbacks=callbacks,
            )

        dinf.save_results(
            args.output_file,
            thetas=thetas,
            probs=probs,
            parameters=dinf_model.parameters,
        )


class _Check(_DinfSubCommand):
    """
    Basic dinf_model health checks.

    Checks that the target and generator functions work and return the
    same feature shapes and dtypes.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "check")
        self.add_argument_model()

    def __call__(self, args: argparse.Namespace):
        dinf_model = dinf.DinfModel.from_file(args.model)
        dinf_model.check()


def _set_loglevel(quiet, verbose):
    # Set root logger's level to WARNING (the default),
    # or ERROR if --quiet is specified.
    level = "WARNING"
    if quiet:
        level = "ERROR"
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich.logging.RichHandler()],
        # Yes, really set the root logging configuration!
        force=True,
    )

    # If --verbose is specified, we increase the log level for dinf code.
    # The root logger's level remains at WARNING so that we don't get
    # additional messages from third-party packages.
    assert not (verbose and quiet)
    if verbose == 1:
        level = "INFO"
    elif verbose >= 2:
        level = "DEBUG"
    logging.getLogger(dinf.__name__).setLevel(level)


def main(args_list=None):
    top_parser = argparse.ArgumentParser(
        prog="dinf",
        description="Discriminator-based inference of population parameters.",
    )
    top_parser.add_argument(
        "-V", "--version", action="version", version=dinf.__version__
    )

    subparsers = top_parser.add_subparsers(dest="subcommand")
    _Check(subparsers)
    _Train(subparsers)
    _Predict(subparsers)
    _AbcGan(subparsers)
    _McmcGan(subparsers)
    _PgGan(subparsers)

    args = top_parser.parse_args(args_list)
    if args.subcommand is None:
        top_parser.print_help()
        exit(1)

    _set_loglevel(args.quiet, args.verbose)
    args.func(args)
