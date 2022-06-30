(sec_guide_testing_a_dinf_model)=
# Testing a Dinf model

This page explains how to test a Dinf model.
See the section on {ref}`creating a Dinf model <sec_guide_creating_a_dinf_model>`
for how to write a model file.

## Dinf model files

When Dinf reads a model file (a `.py` file), it looks for a `genobuilder`
variable which must be an instance of the {class}`.Genobuilder` class.
The genobuilder contains all the information needed to train a discriminator
network.

## Checking a model file

The Dinf command line interface can be used to do basic checks of the model.

```
dinf check examples/bottleneck/model.py
```

This will sample parameters from the prior distribution,
call the `generator_func` and `target_func` functions,
and confirm that their output matches the specified `feature_shape`.
If the model is a simulation-only model (i.e. the `target_func` is `None`),
then the parameters will be checked to ensure they each have a `truth` value.

## Training a discriminator

As an additional check of the model, it is useful to train a discriminator
with a modest number of replicates to confirm that the discriminator
can learn from the training data.

```
dinf train \
    --epochs 10 \
    --training-replicates 1000 \
    --test-replicates 1000 \
    examples/bottleneck/model.py \
    /tmp/discriminator.pkl
```

Msprime simulations are quite fast, and on an 8-core i7-8665U laptop with
CPU-only training, this completes in about 40 seconds.
The output indicates that the test loss is decreasing over time, and the
test accuracy is increasing. This suggests that the discriminator is capable
of learning from the model.

```
[epoch 1|1000] train loss 0.4213, accuracy 0.7710; test loss 1.0608, accuracy 0.4990
[epoch 2|1000] train loss 0.3079, accuracy 0.8310; test loss 0.7033, accuracy 0.5080
[epoch 3|1000] train loss 0.2651, accuracy 0.8800; test loss 0.6397, accuracy 0.5490
[epoch 4|1000] train loss 0.2268, accuracy 0.9460; test loss 0.6157, accuracy 0.6350
[epoch 5|1000] train loss 0.1947, accuracy 0.9680; test loss 0.5933, accuracy 0.6830
[epoch 6|1000] train loss 0.1703, accuracy 0.9700; test loss 0.5763, accuracy 0.7070
[epoch 7|1000] train loss 0.1439, accuracy 0.9830; test loss 0.5415, accuracy 0.7620
[epoch 8|1000] train loss 0.1331, accuracy 0.9840; test loss 0.5123, accuracy 0.7970
[epoch 9|1000] train loss 0.1087, accuracy 0.9920; test loss 0.4876, accuracy 0.8160
[epoch 10|1000] train loss 0.0930, accuracy 0.9870; test loss 0.4353, accuracy 0.8420
```

To obtain more impressive accuracy, additional replicates will be needed.
Other ways to improve the accuracy are discussed in the
{ref}`sec_guide_accuracy` section.
