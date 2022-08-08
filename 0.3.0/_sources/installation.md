(sec_installation)=
# Installation

Dinf requires Python >= 3.8, and can be installed with `pip`
or with `conda` (via the *bioconda* channel).

## Pip installation

Installation is as simple as `pip install dinf`,
but we recommend installation inside a virtual environment.

```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install dinf
```

### GPU training

To train models using a GPU, an appropriate version of `jaxlib`
needs to be installed. This can be done after installing `dinf`.
See the
[`jax` documentation](https://github.com/google/jax/#installation)
for instructions. E.g. on Linux try

```
pip install "jax[cuda]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Check that your GPU device(s) are recognised by jax.
```
$ python
Python 3.9.13 | packaged by conda-forge | (main, May 27 2022, 16:56:21)
[GCC 10.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> jax.devices()
[GpuDevice(id=0, process_index=0), GpuDevice(id=1, process_index=0)]
```

## Conda/mamba installation

First ensure that your conda configuration includes the *bioconda* channel
following the [bioconda instructions](https://bioconda.github.io/),
then create a fresh Dinf environment with the commands below.
[Mamba](https://github.com/mamba-org/mamba) is a faster implementation
of conda, but substitute `conda` if you don't want to use `mamba`.

```
mamba create -n dinf dinf
mamba activate dinf
```

The conda-forge `jaxlib` packages are GPU-enabled by default,
so GPU support should just work. See GPU instructions above to confirm
that your GPU device(s) are recognised.
