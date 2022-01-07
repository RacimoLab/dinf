# API reference

## Defining a Dinf model

```{eval-rst}
.. autoclass:: dinf.Genobuilder
   :members:
```

## Feature extraction

```{eval-rst}
.. autoclass:: dinf.BinnedHaplotypeMatrix
   :members:
```


```{eval-rst}
.. autoclass:: dinf.MultipleBinnedHaplotypeMatrices
   :members:
```

```{eval-rst}
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
```

```{eval-rst}
.. autoclass:: dinf.Param
   :members:
```

## Inference

```{eval-rst}
.. autofunction:: dinf.mcmc_gan
```

## Miscellaneous

```{eval-rst}
.. autoclass:: dinf.Discriminator
   :members:
```

```{eval-rst}
.. autoclass:: dinf.Store
   :members:

   .. automethod:: __len__
   .. automethod:: __getitem__
```
