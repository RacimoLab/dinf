__version__ = "undefined"
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    pass

"""
from .feature_extractor import (
    FeatureExtractor,
    BinnedHaplotypeMatrix,
)
from .generator import (
    Generator,
    Parameter,
    MsprimeHudsonSimulator,
)
from .dinf import train, abc, opt, mcmc
"""
