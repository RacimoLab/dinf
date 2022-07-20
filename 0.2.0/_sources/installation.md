(sec_installation)=
# Installation

Dinf can be installed via `pip` or via `conda` (or `mamba`).

## Pip installation

Create a virtual environment, activate it, then install.
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install git+https://github.com/RacimoLab/dinf.git
```

### GPU training

To train models using a GPU, an appropriate version of jaxlib
needs to be installed. See the JAX documentation
https://github.com/google/jax/#installation

E.g.
```
pip install "jax[cuda11_cudnn82]" \
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

[Mamba](https://github.com/mamba-org/mamba) is a faster implementation
of conda. Substitute `conda` below if you don't have mamba.
Clone the repository, create a conda virtual environment, activate, then install.
```
git clone https://github.com/RacimoLab/dinf.git
cd dinf
mamba env create -n dinf --file environment.yml
mamba activate dinf
pip install .
```

The conda-forge jaxlib packages are cuda-enabled by default,
so GPU support should just work. See pip instructions for checking
that your GPU device(s) are recognised.
