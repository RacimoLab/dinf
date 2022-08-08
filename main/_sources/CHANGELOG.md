# Changelog

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
