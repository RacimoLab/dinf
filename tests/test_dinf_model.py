import numpy as np
import pytest

import dinf

# minimal dinf_model.


def _generator_func(seed, my_param):
    rng = np.random.default_rng(seed)
    return {"a": np.array(rng.uniform(size=10, low=0, high=my_param))}


def _target_func(seed):
    return _generator_func(seed, 10)


_parameters = dinf.Parameters(my_param=dinf.Param(low=0, high=100))


class TestDinfModel:
    def test_basic(self):
        g = dinf.DinfModel(
            target_func=_target_func,
            generator_func=_generator_func,
            parameters=_parameters,
        )
        g.check()

    def test_zero_parameters(self):
        with pytest.raises(ValueError, match="Must define one or more parameters"):
            dinf.DinfModel(
                target_func=_target_func,
                generator_func=_generator_func,
                parameters=dinf.Parameters(),
            )

    def test_mismatched_shape(self):
        def generator(seed, my_param):
            return {"a": np.array([1.0, 2.0])}

        g = dinf.DinfModel(
            target_func=_target_func,
            generator_func=generator,
            parameters=_parameters,
        )
        with pytest.raises(ValueError, match="generator_func .* shape"):
            g.check()

    def test_mismatched_dtype(self):
        def target(seed):
            return {"a": np.zeros(10, dtype=int)}

        g = dinf.DinfModel(
            target_func=target,
            generator_func=_generator_func,
            parameters=_parameters,
        )
        with pytest.raises(ValueError, match="generator_func .* dtype"):
            g.check()

    def test_bad_custom_param(self):
        class TriangleParam(dinf.Param):
            def sample_prior(self, size, rng):
                mid = (self.low + self.high) / 2
                # Accidentally forget to use 'size'.
                return rng.triangular(left=self.low, mode=mid, right=self.high)

        parameters = dinf.Parameters(a=TriangleParam(low=0, high=100))
        g = dinf.DinfModel(
            target_func=_target_func,
            generator_func=_generator_func,
            parameters=parameters,
        )
        with pytest.raises(ValueError, match="parameters.sample_prior.* shape"):
            g.check()

    def test_missing_truth_values(self):
        with pytest.raises(ValueError, match="Truth values missing.*my_param"):
            dinf.DinfModel(
                target_func=None,
                generator_func=_generator_func,
                parameters=_parameters,
            )

    def test_from_file_no_spec(self):
        with pytest.raises(ImportError, match=r"nonexistent.txt"):
            dinf.DinfModel.from_file("nonexistent.txt")

    def test_from_file_file_not_found(self):
        with pytest.raises(FileNotFoundError, match=r"nonexistent.py"):
            dinf.DinfModel.from_file("nonexistent.py")

    @pytest.mark.usefixtures("tmp_path")
    def test_from_file_obj_not_found(self, tmp_path):
        filename = tmp_path / "model.py"
        with open(filename, "w") as f:
            f.write("geeenobilder = {}\n")
        with pytest.raises(AttributeError, match="variable 'dinf_model' not found"):
            dinf.DinfModel.from_file(filename)

    @pytest.mark.usefixtures("tmp_path")
    def test_from_file_obj_wrong_type(self, tmp_path):
        filename = tmp_path / "model.py"
        with open(filename, "w") as f:
            f.write("dinf_model = {}\n")
        with pytest.raises(TypeError, match="not a .*DinfModel"):
            dinf.DinfModel.from_file(filename)
