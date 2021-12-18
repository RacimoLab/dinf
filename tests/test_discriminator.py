import numpy as np
import jax
import chex
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


class TestExchangeableCNN:
    @pytest.mark.parametrize("train", [True, False])
    def test_cnn(self, train: bool):
        cnn = discriminator.ExchangeableCNN()
        x, _, _ = random_dataset(50)
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        y, new_variables = cnn.apply(
            variables, x, train=train, mutable=["batch_stats"] if train else []
        )
        assert np.shape(y) == (50,)


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
    def test_from_input_shape(self, input_shape):
        rng = np.random.default_rng(1234)
        discriminator.Discriminator.from_input_shape(input_shape, rng)

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
        rng = np.random.default_rng(1234)
        with pytest.raises(ValueError, match="features must each have shape"):
            discriminator.Discriminator.from_input_shape(input_shape, rng)

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
        d1 = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d1.fit(rng, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
        chex.assert_tree_all_finite(d1.variables)

        # Results should be deterministic and not depend on validation.
        rng = np.random.default_rng(1234)
        d2 = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d2.fit(rng, train_x=train_x, train_y=train_y)
        chex.assert_trees_all_close(d1.variables, d2.variables)

    def test_fit_bad_shapes(self):
        rng = np.random.default_rng(1234)

        x, y, input_shape = random_dataset(shape=(30, 12, 34, 2))
        d = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d.fit(rng, train_x=x, train_y=y)

        with pytest.raises(ValueError, match="Must specify both"):
            d.fit(rng, train_x=x, train_y=y, val_x=x)
        with pytest.raises(ValueError, match="Must specify both"):
            d.fit(rng, train_x=x, train_y=y, val_y=y)

        with pytest.raises(ValueError, match="Leading dimensions"):
            d.fit(rng, train_x=x[:20, ...], train_y=y)
        with pytest.raises(ValueError, match="Leading dimensions"):
            d.fit(rng, train_x=x, train_y=y[:20])
        with pytest.raises(ValueError, match="Leading dimensions"):
            d.fit(rng, train_x=x, train_y=y, val_x=x[:20, ...], val_y=y)
        with pytest.raises(ValueError, match="Leading dimensions"):
            d.fit(rng, train_x=x, train_y=y, val_x=x, val_y=y[:20])

        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(rng, train_x=x[:, :8, :, :], train_y=y)
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(rng, train_x=x[:, :, :8, :], train_y=y)
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(rng, train_x=x[:, :, :, :1], train_y=y)
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(rng, train_x=x, train_y=y, val_x=x[:, :8, :, :], val_y=y)
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(rng, train_x=x, train_y=y, val_x=x[:, :, :8, :], val_y=y)
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.fit(rng, train_x=x, train_y=y, val_x=x[:, :, :, :1], val_y=y)

    @pytest.mark.usefixtures("capsys")
    def test_summary(self, capsys):
        rng = np.random.default_rng(1234)
        d = discriminator.Discriminator.from_input_shape(
            {"a": np.array((30, 40, 1))}, rng
        )
        d.summary()
        captured = capsys.readouterr()
        assert "params" in captured.out
        assert "batch_stats" in captured.out

    @pytest.mark.usefixtures("tmp_path")
    def test_load_save_roundtrip(self, tmp_path):
        rng = np.random.default_rng(1234)
        x, y, input_shape = random_dataset(50)
        d1 = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d1.fit(rng, train_x=x, train_y=y)
        d1_y = d1.predict(x)
        filename = tmp_path / "discriminator.pkl"
        d1.to_file(filename)
        d2 = discriminator.Discriminator.from_file(filename)
        d2_y = d2.predict(x)
        np.testing.assert_allclose(d1_y, d2_y)

    @pytest.mark.parametrize("discriminator_format", [0, "0.0.1"])
    def test_load_old_file(self, tmp_path, discriminator_format):
        rng = np.random.default_rng(1234)
        d1 = discriminator.Discriminator.from_input_shape((30, 40, 1), rng)
        d1.discriminator_format = discriminator_format
        filename = tmp_path / "discriminator.pkl"
        d1.to_file(filename)
        with pytest.raises(ValueError, match="discriminator is not compatible"):
            discriminator.Discriminator.from_file(filename)

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
        d = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d.fit(rng, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)

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
        d = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d.fit(rng, train_x=x, train_y=y)

        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.predict(x[:, :8, :, :])
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.predict(x[:, :, :16, :])
        with pytest.raises(ValueError, match="Trailing dimensions"):
            d.predict(x[:, :, :, :1])

    def test_predict_without_fit(self):
        rng = np.random.default_rng(1234)
        x, _, input_shape = random_dataset(shape=(30, 12, 34, 2))
        d = discriminator.Discriminator.from_input_shape(input_shape, rng)
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
