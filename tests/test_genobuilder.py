import numpy as np
import pytest

import dinf

# minimal genobuilder.


def _generator_func(seed, a):
    rng = np.random.default_rng(seed)
    return np.array(rng.uniform(low=0, high=a))


def _target_func(seed):
    return _generator_func(seed, 10)


_parameters = dinf.Parameters(a=dinf.Param(low=0, high=100))


class TestGenobuilder:
    def test_basic(self):
        g = dinf.Genobuilder(
            target_func=_target_func,
            generator_func=_generator_func,
            parameters=_parameters,
            feature_shape=(),
        )
        g.check()

    def test_zero_parameters(self):
        with pytest.raises(ValueError, match="Must define one or more parameters"):
            dinf.Genobuilder(
                target_func=_target_func,
                generator_func=_generator_func,
                parameters=dinf.Parameters(),
                feature_shape=(),
            )

    def test_wrong_generator_shape(self):
        def generator(seed, a):
            return np.array([1.0, 2.0])

        g = dinf.Genobuilder(
            target_func=_target_func,
            generator_func=generator,
            parameters=_parameters,
            feature_shape=(),
        )
        with pytest.raises(ValueError, match="generator_func has shape"):
            g.check()

    def test_wrong_target_shape(self):
        def target(seed):
            return np.array([1.0, 2.0])

        g = dinf.Genobuilder(
            target_func=target,
            generator_func=_generator_func,
            parameters=_parameters,
            feature_shape=(),
        )
        with pytest.raises(ValueError, match="target_func has shape"):
            g.check()
