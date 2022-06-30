(sec_api)=
# API reference

(sec_api_defining_a_dinf_model)=
## Defining a Dinf model

```{eval-rst}
.. autoclass:: dinf.Genobuilder
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

   .. automethod:: __iter__
   .. automethod:: __getitem__
   .. automethod:: __len__
```

(sec_api_parameters)=
## Specifying parameters

```{eval-rst}
.. autoclass:: dinf.Parameters
   :members:

.. autoclass:: dinf.Param
   :members:
```

(sec_api_inference)=
## Inference

```{eval-rst}
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
```

(sec_api_misc)=
## Miscellaneous

```{eval-rst}

.. automodule:: dinf.misc
   :members: ts_individuals

.. autoclass:: dinf.Store
   :members:

   .. automethod:: __len__
   .. automethod:: __getitem__
```
