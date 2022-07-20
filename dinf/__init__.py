__version__ = "undefined"
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    pass

from .misc import ts_individuals
from .store import Store as Store
from .parameters import Param, Parameters
from .dinf_model import DinfModel
from .vcf import (
    BagOfVcf,
    get_contig_lengths,
    get_samples_from_1kgp_metadata,
)
from .feature_extractor import (
    HaplotypeMatrix,
    MultipleHaplotypeMatrices,
    BinnedHaplotypeMatrix,
    MultipleBinnedHaplotypeMatrices,
)
from .discriminator import (
    Discriminator,
    Surrogate,
    ExchangeableCNN,
    ExchangeablePGGAN,
    Symmetric,
)
from .dinf import (
    abc_gan,
    alfi_mcmc_gan,
    mcmc_gan,
    pg_gan,
    predict,
    train,
    save_results,
    load_results,
    sample_smooth,
)
