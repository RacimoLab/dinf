(sec_guide_accuracy)=
# Improving discriminator accuracy

```{todo}
Edit and expand this section
```

## Training replicates

- More training replicates!
- Limit the number of training epochs. Beyond a few epochs, the discriminator
  can overfit (test loss goes up, test accuracy goes down).
  But this can vary depending on the network architecture and capacity.

## Information content of features

- Bigger feature arrays hold more information.
- Does the feature matrix constitute a sufficient statistic
  for the parameter(s) in question?

## Discriminator network capacity

Assuming the features contain sufficient information,
the discriminator network needs to be able to extract this.
The capacity of the network can be increased by increasing
the number of trainable neural network parameters
- Deeper network
- More filters in convolution layers
- More neurons in fully connected (dense) layers

Increased network capacity comes at a cost.
- Need to train for longer (more training replicates! more epochs?).
- Can overfit.
