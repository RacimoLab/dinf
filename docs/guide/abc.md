(sec_guide_abc)=
# An ABC analysis

This page shows how to do a rejection ABC analysis,
where the discriminator output is used as a measure of similarity
to the target dataset.
Dinf saves the discriminator output using the {doc}`arviz <arviz:index>`
package, which we'll use to load the data back into a Python terminal.

## Training the discriminator

Using the `train` subcommand of the command line interface, we'll train the
discriminator on a large number of replicates.
Below, we use the discriminator on the two-parameter
bottleneck model from the {ref}`sec_guide_creating_a_dinf_model` section.
We'll use 1 million training replicates and 20,000 test replicates,
and also set the seed for the random number generator to obtain
reproducible results. This will train the discriminator on 500,000
samples from the generator (with parameter values sampled from the prior
distribution) and 500,000 samples from the target dataset
(using the `truth` parameter values, as this is a simulation-only model).

```
dinf train \
  --seed 1 \
  --training-replicates 1000000 \
  --test-replicates 20000 \
  examples/bottleneck/model.py \
  bottleneck-discriminator.pkl
```

The output `.pkl` file is the trained discriminator, which is a regular Python
{mod}`python:pickle` that contains the trained network's weights.
Using 80 Xeon 6248 CPU cores for simulations, and a Tesla T4 GPU for training
the neural network, this completes in about 6 minutes.

```
[epoch 1|1000000] train loss 0.2063, accuracy 0.9202; test loss 0.1548, accuracy 0.9433
```

## Sampling from the prior and measuring similarity

Next, we'll use the `predict` subcommand to obtain discriminator predications
for 1 million replicates from the generator (with parameter values sampled
from the prior).
We'll use a different seed, to ensure that the new simulations are different
from the simulations on which the discriminator was trained.

```
dinf predict \
  --seed 2 \
  --replicates 1000000 \
  examples/bottleneck/model.py \
  bottleneck-discriminator.pkl
  bottleneck-prior.ncf
```

The output `.ncf` file is an {doc}`arviz <arviz:index>`
{doc}`netcdf <arviz:schema/schema>` file.  This file contains
the parameter values for each replicate, and the predictions
made by the discriminator.

Using 80 Xeon 6248 CPU cores for simulations, and a Tesla T4 GPU for
discriminator predictions, this completes in about 4 minutes.

## Choosing the posterior distribution

Recalling that the discriminator outputs $Pr(t)$, the probability that
a given input feature is from the target distribution,
two simple ways of choosing the posterior distribution are
- choose the $n$ samples with the highest discriminator score, or
- choose samples with a score $>x$.

In practice, we may not trust the absolute $Pr(t)$ values
(e.g. when there is {ref}`model misspecification <sec_guide_misspecification>`),
so choosing the $n$ highest values is preferred. Below, we'll take the top
1000 parameters and output the median value of this posterior sample
(output is prefixed with `##`).

```python
import arviz as az
import numpy as np

dataset = az.from_netcdf("bottleneck-prior.ncf")
# log probability from discriminator
lp = np.array(dataset.sample_stats.lp[0])
# get the indices that sort the log-probs in descending order
idx = np.flip(np.argsort(lp))
# get the posterior sample for parameters N0 and N1
dataset2 = dataset.isel(draw=idx[:1_000])
N0 = np.array(dataset2.posterior["N0"])
N1 = np.array(dataset2.posterior["N1"])
# get median values and 95% credible intervals
np.quantile(N0, [0.025, 0.5, 0.975])
## array([ 2725.32983797, 11779.47948882, 26170.06932507])
np.quantile(N1, [0.025, 0.5, 0.975])
## array([   33.05656911,   126.12728086, 21988.30256896])
```

While the 95% credible intervals are quite wide, the median values
are close to the `truth` values used in the model (`N0=10000` and `N1=200`).
Note that this is a toy model and the accuracy of the discriminator can
yet be improved (see {ref}`sec_guide_accuracy`).
