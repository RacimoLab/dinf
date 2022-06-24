# API reference

## Defining a Dinf model

```{eval-rst}
.. autoclass:: dinf.Genobuilder
   :members:
```

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

## Specifying parameters

```{eval-rst}
.. autoclass:: dinf.Parameters
   :members:

.. autoclass:: dinf.Param
   :members:
```

## Inference

```{eval-rst}
.. autofunction:: dinf.abc_gan

.. autofunction:: dinf.alfi_mcmc_gan

.. autofunction:: dinf.mcmc_gan

.. autofunction:: dinf.pg_gan
```

## Miscellaneous

```{eval-rst}
.. autoclass:: dinf.Discriminator
   :members:

.. autoclass:: dinf.Surrogate
   :members:

.. autoclass:: dinf.Store
   :members:

   .. automethod:: __len__
   .. automethod:: __getitem__
```
