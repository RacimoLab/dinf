__version__ = "undefined"
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    pass

import os

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    # Mute tensorflow.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if "KMP_AFFINITY" not in os.environ:
    # See also https://github.com/tensorflow/tensorflow/issues/29354
    os.environ["KMP_AFFINITY"] = "granularity=fine,noverbose,compact,1,0"

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
