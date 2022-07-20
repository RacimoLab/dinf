(sec_introduction)=
# Introduction

Dinf is discriminator-based inference for population genetics.
It uses a neural network to discriminate between a target dataset
and a simulated dataset.
Inference is done by finding simulation parameters that produce
data closely matching the target dataset.
Dinf provides a [Python API](sec_api) for creating simulation models,
and a [CLI](sec_cli) for discriminator training, inference, and plotting.


## Standing on the shoulders of giants

Dinf uses and takes inspiration from the following projects.

- [JAX](https://jax.readthedocs.io/),
  [flax](https://flax.readthedocs.io/),
  and [optax](https://optax.readthedocs.io/)
  for training neural networks.
- [msprime](https://tskit.dev/msprime/docs/)
  and the [tskit ecosystem](https://tskit.dev/) for simulations.
- [cyvcf2](https://github.com/brentp/cyvcf2) for reading vcf and bcf files
  (which itself uses [htslib](https://github.com/samtools/htslib)).
- [matplotlib](https://matplotlib.org/) for creating plots.
- [pg-gan](https://github.com/mathiesonlab/pg-gan)
  does discriminator-based inference using simulated annealing
  in a GAN (generative adversarial network).
  Dinf implements and extends many ideas from pg-gan.
