__version__ = "undefined"
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    pass

import os

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    # Mute tensorflow/xla.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if "KMP_AFFINITY" not in os.environ:
    # Pin threads to cpus. This can improve blas performance.
    os.environ["KMP_AFFINITY"] = "granularity=fine,noverbose,compact,1,0"

# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
# if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
#    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# if "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ:
#    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


from .dinf import (
    abc_gan,
    alfi_mcmc_gan,
    mcmc_gan,
    pg_gan,
    predict,
    train,
    save_results,
    load_results,
)
from .discriminator import (
    Discriminator,
    Surrogate,
    ExchangeableCNN,
    ExchangeablePGGAN,
    Symmetric,
)
from .feature_extractor import (
    HaplotypeMatrix,
    MultipleHaplotypeMatrices,
    BinnedHaplotypeMatrix,
    MultipleBinnedHaplotypeMatrices,
)
from .dinf_model import DinfModel
from .parameters import Param, Parameters
from .store import Store
from .vcf import BagOfVcf, get_contig_lengths, get_samples_from_1kgp_metadata

__all__ = [
    "__version__",
    "BagOfVcf",
    "BinnedHaplotypeMatrix",
    "Discriminator",
    "ExchangeableCNN",
    "ExchangeablePGGAN",
    "MultipleBinnedHaplotypeMatrices",
    "MultipleHaplotypeMatrices",
    "DinfModel",
    "HaplotypeMatrix",
    "Param",
    "Parameters",
    "Surrogate",
    "Store",
    "Symmetric",
    "get_contig_lengths",
    "get_samples_from_1kgp_metadata",
    "abc_gan",
    "alfi_mcmc_gan",
    "mcmc_gan",
    "load_results",
    "pg_gan",
    "predict",
    "save_results",
    "train",
]
