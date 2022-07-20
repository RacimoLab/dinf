(sec_api)=
# API reference

(sec_api_summary)=
## Summary

```{eval-rst}
.. currentmodule:: dinf
```

### Defining a Dinf model

```{eval-rst}
.. autosummary::
    DinfModel
```

### Specifying parameters

```{eval-rst}
.. autosummary::
    Parameters
    Param
```

### Feature extraction

```{eval-rst}
.. autosummary::
    HaplotypeMatrix
    MultipleHaplotypeMatrices
    BinnedHaplotypeMatrix
    MultipleBinnedHaplotypeMatrices
    BagOfVcf
```

### Training

```{eval-rst}
.. autosummary::
    train
```

### Inference

```{eval-rst}
.. autosummary::
    predict
    abc_gan
    mcmc_gan
    alfi_mcmc_gan
    pg_gan
```

### Classification

```{eval-rst}
.. autosummary::
    Discriminator
```

### Discriminator networks

```{eval-rst}
.. autosummary::
    ExchangeableCNN
    ExchangeablePGGAN
    Symmetric
```

### Miscellaneous

```{eval-rst}
.. autosummary::
    save_results
    load_results
    ts_individuals
    sample_smooth
    get_samples_from_1kgp_metadata
    get_contig_lengths
    Store
```

### Plotting

```{eval-rst}
.. currentmodule:: dinf.plot

.. autosummary::
    feature
    features
    metrics
    hist
    hist2d
```

## Reference

(sec_api_defining_a_dinf_model)=
### Defining a Dinf model

```{eval-rst}
.. autoclass:: dinf.DinfModel
   :members:
```

(sec_api_parameters)=
### Specifying parameters

```{eval-rst}
.. autoclass:: dinf.Parameters
   :members:

.. autoclass:: dinf.Param
   :members:
```

(sec_api_feature_extraction)=
### Feature extraction

```{eval-rst}
.. autoclass:: dinf.HaplotypeMatrix
   :members:

   .. automethod:: from_ts
   .. automethod:: from_vcf

.. autoclass:: dinf.MultipleHaplotypeMatrices
   :members:

   .. automethod:: from_ts
   .. automethod:: from_vcf
   .. autoproperty:: shape

.. autoclass:: dinf.BinnedHaplotypeMatrix
   :members:

   .. automethod:: from_ts
   .. automethod:: from_vcf

.. autoclass:: dinf.MultipleBinnedHaplotypeMatrices
   :members:

   .. automethod:: from_ts
   .. automethod:: from_vcf
   .. autoproperty:: shape


.. autoclass:: dinf.BagOfVcf
   :members:
```

(sec_api_train)=
### Training

```{eval-rst}
.. autofunction:: dinf.train
```

(sec_api_inference)=
### Inference

```{eval-rst}
.. autofunction:: dinf.predict
.. autofunction:: dinf.abc_gan
.. autofunction:: dinf.mcmc_gan
.. autofunction:: dinf.alfi_mcmc_gan
.. autofunction:: dinf.pg_gan
```

(sec_api_classification)=
### Classification

```{eval-rst}
.. autoclass:: dinf.Discriminator

    .. automethod:: init
    .. automethod:: from_file
    .. automethod:: to_file
    .. automethod:: fit
    .. automethod:: predict
    .. automethod:: summary
```

(sec_api_discriminator_networks)=
### Discriminator networks

```{eval-rst}
.. autoclass:: dinf.ExchangeableCNN
.. autoclass:: dinf.ExchangeablePGGAN
.. autoclass:: dinf.Symmetric
```

(sec_api_misc)=
### Miscellaneous

```{eval-rst}

.. autofunction:: dinf.save_results
.. autofunction:: dinf.load_results

.. autofunction:: dinf.ts_individuals
.. autofunction:: dinf.sample_smooth

.. autofunction:: dinf.get_samples_from_1kgp_metadata
.. autofunction:: dinf.get_contig_lengths

.. autoclass:: dinf.Store
   :members:

   .. automethod:: __len__
   .. automethod:: __getitem__
```

(sec_api_plotting)=
### Plotting

```{eval-rst}
.. automodule:: dinf.plot
    :members:
```
