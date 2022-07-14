from typing import Callable

import numpy as np
import jax
import chex
import flax
import pytest

from dinf import discriminator
from dinf.misc import leading_dim_size, tree_cons, tree_car, tree_cdr


def random_dataset(size=None, shape=None, seed=1234):
    """Make up a test dataset."""
    rng = np.random.default_rng(seed)
    if shape is None:
        assert size is not None
        shape = {"x": (size, 32, 32, 1)}
    else:
        assert size is None
        size = jax.tree_flatten(tree_car(shape))[0][0]
    x = jax.tree_map(
        lambda s: rng.integers(low=0, high=128, size=s, dtype=np.int8),
        shape,
        is_leaf=lambda x: isinstance(x, tuple),
    )
    y = rng.integers(low=0, high=2, size=(size,), dtype=np.int8)
    input_shape = tree_cdr(shape)
    return x, y, input_shape


class TestSymmetric:
    @pytest.mark.parametrize("axis", [1, 2])
    @pytest.mark.parametrize(
        "func,k,squash",
        [
            ("max", None, 1),
            ("mean", None, 1),
            ("sum", None, 1),
            ("var", None, 1),
            ("moments", 2, 2),
            ("moments", 3, 3),
            ("moments", 4, 4),
            ("central-moments", 2, 2),
            ("central-moments", 3, 3),
            ("central-moments", 4, 4),
        ],
    )
    def test_shape(self, func, k, squash, axis):
        shape = (4, 4, 4, 4)
        sym = discriminator.Symmetric(func=func, k=k)
        x, _, _ = random_dataset(shape=shape)
        variables = sym.init(jax.random.PRNGKey(0), x, axis=axis)
        y = sym.apply(variables, x, axis=axis)

        expected_shape = list(shape)
        expected_shape[axis] = squash
        assert y.shape == tuple(expected_shape)

    @pytest.mark.parametrize("axis", [1, 2])
    def test_max(self, axis):
        shape = (4, 4, 4, 1)
        sym = discriminator.Symmetric(func="max")
        x, _, _ = random_dataset(shape=shape)
        variables = sym.init(jax.random.PRNGKey(0), x, axis=axis)
        y = sym.apply(variables, x, axis=axis)
        np.testing.assert_allclose(y, np.max(x, axis=axis, keepdims=True))

    @pytest.mark.parametrize("axis", [1, 2])
    def test_mean(self, axis):
        shape = (4, 4, 4, 1)
        sym = discriminator.Symmetric(func="mean")
        x, _, _ = random_dataset(shape=shape)
        variables = sym.init(jax.random.PRNGKey(0), x, axis=axis)
        y = sym.apply(variables, x, axis=axis)
        np.testing.assert_allclose(y, np.mean(x, axis=axis, keepdims=True))

    @pytest.mark.parametrize("axis", [1, 2])
    def test_sum(self, axis):
        shape = (4, 4, 4, 1)
        sym = discriminator.Symmetric(func="sum")
        x, _, _ = random_dataset(shape=shape)
        variables = sym.init(jax.random.PRNGKey(0), x, axis=axis)
        y = sym.apply(variables, x, axis=axis)
        np.testing.assert_allclose(y, np.sum(x, axis=axis, keepdims=True))

    @pytest.mark.parametrize("axis", [1, 2])
    def test_var(self, axis):
        shape = (4, 4, 4, 1)
        sym = discriminator.Symmetric(func="var")
        x, _, _ = random_dataset(shape=shape)
        variables = sym.init(jax.random.PRNGKey(0), x, axis=axis)
        y = sym.apply(variables, x, axis=axis)
        np.testing.assert_allclose(y, np.var(x, axis=axis, keepdims=True))

    @pytest.mark.parametrize("axis", [1, 2])
    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_moments(self, axis, k):
        shape = (4, 4, 4, 1)
        sym = discriminator.Symmetric(func="moments", k=k)
        x, _, _ = random_dataset(shape=shape)
        variables = sym.init(jax.random.PRNGKey(0), x, axis=axis)
        y = sym.apply(variables, x, axis=axis)

        m1 = np.mean(x, axis=axis, keepdims=True)
        m2 = np.mean(x**2, axis=axis, keepdims=True)
        m3 = np.mean(x**3, axis=axis, keepdims=True)
        m4 = np.mean(x**4, axis=axis, keepdims=True)
        expected = np.concatenate([m1, m2, m3, m4][:k], axis=axis)
        np.testing.assert_allclose(y, expected)

    @pytest.mark.parametrize("axis", [1, 2])
    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_central_moments(self, axis, k):
        shape = (4, 4, 4, 1)
        sym = discriminator.Symmetric(func="central-moments", k=k)
        x, _, _ = random_dataset(shape=shape)
        variables = sym.init(jax.random.PRNGKey(0), x, axis=axis)
        y = sym.apply(variables, x, axis=axis)

        m1 = np.mean(x, axis=axis, keepdims=True)
        m2 = np.var(x, axis=axis, keepdims=True)
        m3 = np.mean((x - m1) ** 3, axis=axis, keepdims=True)
        m4 = np.mean((x - m1) ** 4, axis=axis, keepdims=True)
        expected = np.concatenate([m1, m2, m3, m4][:k], axis=axis)
        np.testing.assert_allclose(y, expected, rtol=1e-6)

    @pytest.mark.parametrize("func", [None, "not-a-func"])
    def test_bad_func(self, func):
        sym = discriminator.Symmetric(func=func)
        x, _, _ = random_dataset(50)
        with pytest.raises(ValueError, match="Unexpected func"):
            sym.init(jax.random.PRNGKey(0), x)

    @pytest.mark.parametrize("k", [-1, 0, 1])
    @pytest.mark.parametrize("func", ["moments", "central-moments"])
    def test_bad_k(self, func, k):
        sym = discriminator.Symmetric(func=func, k=k)
        x, _, _ = random_dataset(50)
        with pytest.raises(ValueError, match="Must have k >= 2"):
            sym.init(jax.random.PRNGKey(0), x)


class _TestCNN:
    cnn: Callable  # nn.Module class

    @pytest.mark.parametrize("train", [True, False])
    def test_cnn(self, train: bool):
        cnn = self.cnn()
        x, _, _ = random_dataset(50)
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        y, new_variables = cnn.apply(
            variables,
            x,
            train=train,
            mutable=["batch_stats"] if train else [],
            rngs={"dropout": jax.random.PRNGKey(0)} if train else {},
        )
        assert np.shape(y) == (50,)

    @pytest.mark.parametrize("seed", (1, 2, 3, 4))
    def test_individual_exchangeability(self, seed):
        cnn = self.cnn()
        x1, _, _ = random_dataset(50, seed=seed)
        variables = cnn.init(jax.random.PRNGKey(0), x1, train=False)
        y1 = cnn.apply(variables, x1, train=False)

        # permute rows
        rng = np.random.default_rng(seed + 100)
        for _ in range(10):
            x2 = jax.tree_map(lambda a: rng.permutation(a, axis=1), x1)
            y2 = cnn.apply(variables, x2, train=False)
            # Set a generous tolerance here, as the CNN is using 32 bit floats.
            np.testing.assert_allclose(y1, y2, rtol=1e-2)

    @pytest.mark.parametrize("seed", (1, 2, 3, 4))
    def test_batch_dimension_exchangeability(self, seed):
        cnn = self.cnn()
        size = 50
        x1, _, _ = random_dataset(size, seed=seed)
        variables = cnn.init(jax.random.PRNGKey(0), x1, train=False)
        y1 = cnn.apply(variables, x1, train=False)

        # permute along batch dimension
        rng = np.random.default_rng(seed + 100)
        for _ in range(10):
            idx = rng.permutation(size)
            x2 = jax.tree_map(lambda a: a[idx], x1)
            y2 = cnn.apply(variables, x2, train=False)
            # Set a generous tolerance here, as the CNN is using 32 bit floats.
            np.testing.assert_allclose(y1[idx], y2, rtol=1e-2)


class TestExchangeableCNN(_TestCNN):
    cnn = discriminator.ExchangeableCNN

    def test_sizes(self):
        x, _, _ = random_dataset(50)

        cnn = self.cnn(sizes1=(1,), sizes2=(1,))
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        _ = cnn.apply(variables, x, train=False)
        assert sum(1 for k in variables["params"].keys() if "Conv" in k) == 2
        assert sum(1 for k in variables["params"].keys() if "Dense" in k) == 1

        cnn = self.cnn(sizes1=(1, 1), sizes2=())
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        _ = cnn.apply(variables, x, train=False)
        assert sum(1 for k in variables["params"].keys() if "Conv" in k) == 2
        assert sum(1 for k in variables["params"].keys() if "Dense" in k) == 1

        cnn = self.cnn(sizes1=(), sizes2=(1, 1))
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        _ = cnn.apply(variables, x, train=False)
        assert sum(1 for k in variables["params"].keys() if "Conv" in k) == 2
        assert sum(1 for k in variables["params"].keys() if "Dense" in k) == 1

        cnn = self.cnn(sizes1=(1, 2, 3), sizes2=(1, 2, 3, 4))
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        _ = cnn.apply(variables, x, train=False)
        assert sum(1 for k in variables["params"].keys() if "Conv" in k) == 7
        assert sum(1 for k in variables["params"].keys() if "Dense" in k) == 1

        # two labels
        x2, _, _ = random_dataset(shape={"x": (1, 10, 20, 1), "y": (1, 20, 10, 1)})
        cnn = self.cnn(sizes1=(1, 2, 3), sizes2=(1, 2, 3, 4))
        variables = cnn.init(jax.random.PRNGKey(0), x2, train=False)
        _ = cnn.apply(variables, x2, train=False)
        assert sum(1 for k in variables["params"].keys() if "Conv" in k) == 2 * 7
        assert sum(1 for k in variables["params"].keys() if "Dense" in k) == 1


class TestExchangeablePGGAN(_TestCNN):
    cnn = discriminator.ExchangeablePGGAN

    def test_sizes(self):
        x, _, _ = random_dataset(50)

        cnn = self.cnn(sizes1=(1,), sizes2=(1,))
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        _ = cnn.apply(variables, x, train=False)
        assert sum(1 for k in variables["params"].keys() if "Conv" in k) == 1
        assert sum(1 for k in variables["params"].keys() if "Dense" in k) == 1 + 1

        cnn = self.cnn(sizes1=(1, 1), sizes2=())
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        _ = cnn.apply(variables, x, train=False)
        assert sum(1 for k in variables["params"].keys() if "Conv" in k) == 2
        assert sum(1 for k in variables["params"].keys() if "Dense" in k) == 0 + 1

        cnn = self.cnn(sizes1=(), sizes2=(1, 1))
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        _ = cnn.apply(variables, x, train=False)
        assert sum(1 for k in variables["params"].keys() if "Conv" in k) == 0
        assert sum(1 for k in variables["params"].keys() if "Dense" in k) == 2 + 1

        cnn = self.cnn(sizes1=(1, 2, 3), sizes2=(1, 2, 3, 4))
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        _ = cnn.apply(variables, x, train=False)
        assert sum(1 for k in variables["params"].keys() if "Conv" in k) == 3
        assert sum(1 for k in variables["params"].keys() if "Dense" in k) == 4 + 1

        # two labels
        x2, _, _ = random_dataset(shape={"x": (1, 10, 20, 1), "y": (1, 20, 10, 1)})
        cnn = self.cnn(sizes1=(1, 2, 3), sizes2=(1, 2, 3, 4))
        variables = cnn.init(jax.random.PRNGKey(0), x2, train=False)
        _ = cnn.apply(variables, x2, train=False)
        assert sum(1 for k in variables["params"].keys() if "Conv" in k) == 3
        assert sum(1 for k in variables["params"].keys() if "Dense" in k) == 4 + 1


class TestDiscriminator:
    @pytest.mark.parametrize(
        "input_shape",
        [
            (32, 64, 1),
            (32, 64, 4),
            {"x": np.array((64, 32, 1))},
            {"x": (64, 32, 1), "y": {"z": (16, 8, 4)}},
        ],
    )
    def test_init(self, input_shape):
        rng = np.random.default_rng(1234)
        d = discriminator.Discriminator(input_shape)
        d = d.init(rng)
        assert d is not None
        assert len(d.state.params) > 0

    @pytest.mark.parametrize(
        "input_shape",
        [
            16,
            (16,),
            (16, 16),
            (32, 64, 10),
            (1, 64, 10),
            (64, 1, 10),
            {"x": np.array((64, 32, 10))},
            {"x": np.array((1, 32, 10))},
            {"x": np.array((64, 1, 10))},
            {"x": (64, 32, 1), "y": {"z": (16, 8, 10)}},
            {"x": (64, 32, 1), "y": {"z": (16, 1, 4)}},
            {"x": (64, 32, 1), "y": {"z": (1, 8, 4)}},
        ],
    )
    def test_bad_shape(self, input_shape):
        with pytest.raises(ValueError, match="features must each have shape"):
            discriminator.Discriminator(input_shape)

    @pytest.mark.parametrize(
        "shape",
        [
            (20, 8, 32, 1),
            {"a": (20, 8, 32, 1), "b": {"c": (20, 16, 8, 2)}},
        ],
    )
    def test_fit(self, shape):
        train_x, train_y, input_shape = random_dataset(shape=shape)
        val_x, val_y, _ = random_dataset(shape=tree_cons(10, tree_cdr(shape)))

        rng = np.random.default_rng(1234)
        d1 = discriminator.Discriminator(input_shape).init(rng)
        d1.fit(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, rng=rng)
        chex.assert_tree_all_finite(d1.state)

        # Results should be deterministic and not depend on validation.
        rng = np.random.default_rng(1234)
        d2 = discriminator.Discriminator(input_shape).init(rng)
        d2.fit(train_x=train_x, train_y=train_y, rng=rng)
        chex.assert_trees_all_close(
            flax.serialization.to_state_dict(d1.state),
            flax.serialization.to_state_dict(d2.state),
        )

    def test_fit_bad_shapes(self):
        rng = np.random.default_rng(1234)

        x, y, input_shape = random_dataset(shape=(30, 12, 34, 2))
        d = discriminator.Discriminator(input_shape).init(rng)
        d.fit(train_x=x, train_y=y, rng=rng)

        with pytest.raises(ValueError, match="Must specify both"):
            d.fit(train_x=x, train_y=y, val_x=x, rng=rng)
        with pytest.raises(ValueError, match="Must specify both"):
            d.fit(train_x=x, train_y=y, val_y=y, rng=rng)

        with pytest.raises(ValueError, match="Leading dimensions"):
            d.fit(train_x=x[:20, ...], train_y=y, rng=rng)
        with pytest.raises(ValueError, match="Leading dimensions"):
            d.fit(train_x=x, train_y=y[:20], rng=rng)
        with pytest.raises(ValueError, match="Leading dimensions"):
            d.fit(train_x=x, train_y=y, val_x=x[:20, ...], val_y=y, rng=rng)
        with pytest.raises(ValueError, match="Leading dimensions"):
            d.fit(train_x=x, train_y=y, val_x=x, val_y=y[:20], rng=rng)

        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(train_x=x[:, :8, :, :], train_y=y, rng=rng)
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(train_x=x[:, :, :8, :], train_y=y, rng=rng)
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(train_x=x[:, :, :, :1], train_y=y, rng=rng)
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(train_x=x, train_y=y, val_x=x[:, :8, :, :], val_y=y, rng=rng)
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(train_x=x, train_y=y, val_x=x[:, :, :8, :], val_y=y, rng=rng)
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(train_x=x, train_y=y, val_x=x[:, :, :, :1], val_y=y, rng=rng)

    @pytest.mark.usefixtures("capsys")
    def test_summary(self, capsys):
        rng = np.random.default_rng(1234)
        d = discriminator.Discriminator({"a": np.array((30, 40, 1))}).init(rng)
        d.summary()
        captured = capsys.readouterr()
        assert "params" in captured.out
        assert "batch_stats" in captured.out

    @pytest.mark.usefixtures("tmp_path")
    def test_load_save_roundtrip(self, tmp_path):
        rng = np.random.default_rng(1234)
        x, y, input_shape = random_dataset(50)
        d1 = discriminator.Discriminator(input_shape).init(rng)
        d1.fit(train_x=x, train_y=y, rng=rng)
        d1_y = d1.predict(x)
        filename = tmp_path / "discriminator.nn"
        d1.to_file(filename)
        d2 = discriminator.Discriminator(input_shape).from_file(filename)
        d2_y = d2.predict(x)
        np.testing.assert_allclose(d1_y, d2_y)
        d3 = discriminator.Discriminator(None).from_file(filename)
        d3_y = d3.predict(x)
        np.testing.assert_allclose(d1_y, d3_y)

    @pytest.mark.parametrize("discriminator_format", [0, "0.0.1"])
    def test_load_old_file(self, tmp_path, discriminator_format):
        rng = np.random.default_rng(1234)
        d1 = discriminator.Discriminator((30, 40, 1)).init(rng)
        d1.format_version = discriminator_format
        filename = tmp_path / "discriminator.nn"
        d1.to_file(filename)
        with pytest.raises(ValueError, match="network is not compatible"):
            discriminator.Discriminator((30, 40, 1)).from_file(filename)

    @pytest.mark.parametrize(
        "shape",
        [
            (20, 8, 32, 1),
            {"a": (20, 8, 32, 1), "b": (20, 16, 8, 2)},
        ],
    )
    def test_predict(self, shape):
        train_x, train_y, input_shape = random_dataset(shape=shape)
        val_x, val_y, _ = random_dataset(shape=tree_cons(10, tree_cdr(shape)))

        rng = np.random.default_rng(1234)
        d = discriminator.Discriminator(input_shape).init(rng)
        d.fit(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, rng=rng)

        y1 = d.predict(val_x)
        assert np.shape(y1) == (10,)
        assert all(y1 >= 0)
        assert all(y1 <= 1)

        # Should be deterministic.
        y2 = d.predict(val_x)
        chex.assert_trees_all_close(y1, y2)

    def test_predict_bad_shapes(self):
        rng = np.random.default_rng(1234)
        x, y, input_shape = random_dataset(shape=(30, 12, 34, 2))
        d = discriminator.Discriminator(input_shape).init(rng)
        d.fit(train_x=x, train_y=y, rng=rng)

        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.predict(x[:, :8, :, :])
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.predict(x[:, :, :16, :])
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.predict(x[:, :, :, :1])

    def test_predict_without_fit(self):
        rng = np.random.default_rng(1234)
        x, _, input_shape = random_dataset(shape=(30, 12, 34, 2))
        d = discriminator.Discriminator(input_shape).init(rng)
        with pytest.raises(ValueError, match="has not been trained"):
            d.predict(x)


class TestBatchify:
    def test_batchify(self):
        n = 128
        dataset = {"a": np.zeros(n), "b": {"c": np.ones(n)}}

        # batch_size = 1
        batches = list(discriminator.batchify(dataset, 1))
        assert len(batches) == n
        for batch in batches:
            chex.assert_trees_all_equal_structs(dataset, batch)
            assert leading_dim_size(batch) == 1

        # batch_size divides n
        batch_size = n // 4
        batches = list(discriminator.batchify(dataset, batch_size))
        assert len(batches) == n // batch_size
        for batch in batches:
            chex.assert_trees_all_equal_structs(dataset, batch)
            assert leading_dim_size(batch) == batch_size

        # batch_size > n
        batches = list(discriminator.batchify(dataset, n + 5))
        assert len(batches) == 1
        for batch in batches:
            chex.assert_trees_all_equal_structs(dataset, batch)
            assert leading_dim_size(batch) == n

        # batch_size doesn't divide n evenly
        batch_size = 13
        remainder = n % batch_size
        assert remainder > 0
        batches = list(discriminator.batchify(dataset, batch_size))
        assert len(batches) == n // batch_size + 1
        for batch in batches[:-1]:
            chex.assert_trees_all_equal_structs(dataset, batch)
            assert leading_dim_size(batch) == batch_size
        last_batch = batches[-1]
        chex.assert_trees_all_equal_structs(dataset, last_batch)
        assert leading_dim_size(last_batch) == remainder
