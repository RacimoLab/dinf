# Changelog

## 0.5.0 - 2023-04-26

* Renamed smc to `mc` and mcmc-gan to `mcmc`, in both the API and CLI.
  We're not really doing SMC, and the MCMC isn't really a GAN.
* Used KDE reflection in the violin plots, to ameliorate the reduction
  in density near the edges of the support. This was already implemented
  for the `hist` plots.
* Fixed iteration numbering in plots to start at zero.
* Fixed broken pipe when piping `dinf-tabulate` output to, e.g., head.
* Updated the tutorial in the docs.

## 0.4.0 - 2023-03-15

* Removed abc-gan CLI subcommand, and added `smc` subcommand.
* Removed the `--resample` option for `dinf-plot hist`.
* Implement KDE variance shrinkage.
* Added entropy line plot.
* Bring dependencies up to date, e.g. fix deprecations.
* Lazily import scipy, to improve the CLI lag.
* Added `dinf tabulate` subcommand.


## 0.3.0 - 2022-08-08

* Changed how the CLI is used. Now the model and discriminator
  are specified with the -m/--model and -d/--discriminator
  options, even when they are required.
* Removed DinfModel.feature_shape.
* Simplified Discriminator API.
* Add progress bars and do logging with `rich`.
* Fix KDE bandwidth for abc-gan.
* Separate the train/test datasets from abc-gan proposals.
* Reduce memory use for jobs with millions of simulations.
* Remove alfi-mcmc-gan.


## 0.2.0 - 2022-07-20

Initial release, featuring:

* `dinf` and `dinf-plot` CLIs,
* train, predict, abc-gan, mcmc-gan, alfi-mcmc-gan, pg-gan,
* two flavours of feature matrices, dinf-style `BinnedHaplotypeMatrix`
  and pg-gan-style `HaplotypeMatrix`,
* two flavours of discriminator networks, dinf-style `ExchangeableCNN`
  and pg-gan-style `ExchangeablePGGAN`, plus multi-population versions
  of each.


## 0.1.0 - 2021-12-08

Dummy release to reserve name on Pypi.
