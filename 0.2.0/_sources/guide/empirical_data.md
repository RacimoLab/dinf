(sec_guide_empirical_data)=
# Empirical data

```{todo}
write me
```

## VCF/BCF files

Please use BCF rather than VCF.

See {class}`.BagOfVcf`.

(sec_guide_misspecification)=
## Model misspecification

```{todo}
move this to its own page
```

There might be differences between the features from the `target_func`
and features from the `generator_func` that are not accounted for by
the model. If this is the case, the discriminator may be able to
distinguish between the two data sources very easily.


Differences can be identified by first obtaining an estimate of the parameters
from the empirical data, then using these estimates for the `truth` values,
setting `target_func=None`, and training a new discriminator.
If the discriminatory power is much smaller from this simulation study
(i.e. test accuracy is much lower), this likely indicates unmodelled
differences between the empirical data and the model.

The following things may help to close the gap:
- Increase the model complexity. E.g. include additional
  population structure in the demographic model,
  or sample recombination rates from an empirically-derived
  recombination map.
- Filter the empirical data to remove genotyping errors.
  E.g. remove low-coverage individuals and set a non-zero `maf_thresh`
  for the feature extraction.
- Check any other data characteristics or filtering steps,
  to identify how the empirical data may be different to simulations.
  E.g. SNP ascertainment.

## Complete example

```{literalinclude} ../../examples/gutenkunst2009/model.py
:language: python
```
