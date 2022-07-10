(sec_api)=
# API reference

(sec_api_defining_a_dinf_model)=
## Defining a Dinf model

```{eval-rst}
.. autoclass:: dinf.Genobuilder
   :members:
```

(sec_api_parameters)=
## Specifying parameters

```{eval-rst}
.. autoclass:: dinf.Parameters
   :members:

.. autoclass:: dinf.Param
   :members:
```

(sec_api_feature_extraction)=
## Feature extraction

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
## Training

```{eval-rst}
.. autofunction:: dinf.train
```

(sec_api_inference)=
## Inference

```{eval-rst}
.. autofunction:: dinf.predict
.. autofunction:: dinf.abc_gan
.. autofunction:: dinf.alfi_mcmc_gan
.. autofunction:: dinf.mcmc_gan
.. autofunction:: dinf.pg_gan
```

(sec_api_classification)=
## Classification

```{eval-rst}
.. autoclass:: dinf.Discriminator

    .. automethod:: from_input_shape
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
## Miscellaneous

```{eval-rst}

.. autofunction:: dinf.misc.ts_individuals
.. autofunction:: dinf.get_samples_from_1kgp_metadata
.. autofunction:: dinf.get_contig_lengths

.. autoclass:: dinf.Store
   :members:

   .. automethod:: __len__
   .. automethod:: __getitem__
```

(sec_api_plotting)=
## Plotting

```{eval-rst}
.. automodule:: dinf.plot
    :members:
```
