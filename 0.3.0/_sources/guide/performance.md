(sec_guide_performance)=
# Performance characteristics

Each replicate from the `generator_func` and `target_func` are independent,
and Dinf will sample replicates in parallel using all CPU cores by default.
Most subcommands have a `-j/--parallelism` flag to reduce the number of
simultaneous processes that will be used for simulation.

## CPU

During training, CPU-time of simulations scales roughly
- linearly with the number of replicates,

During network training and/or prediction, CPU-time scales roughly
- linearly with the number of replicates,
- linearly with the number of individuals, and
- linearly with the number of loci.

```{todo}
check these
```

Therefore it's useful to use small values for `num_individuals` and `num_loci`
during model development.

## GPU

Only single-GPU training is supported.
In most cases this wont be a problem because we train with small batch sizes
and the network size is too small to obtain much advantage from multiple GPUs.
Furthermore, when using a GPU, run time is dominated by sampling features
from the `generator_func` and `target_func` which run on the CPU(s).

## Memory

Max memory scales
- linearly with number of replicates,
- linearly with the number of individuals, and
- linearly with the number of loci.

```{todo}
check this
```

All simulated feature matrices are kept in RAM. Online training by streaming
features from the `target_func` and `generator_func` to the discriminator
is possible, but this is not implemented (yet?).
