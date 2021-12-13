import numpy as np
import jax
import chex
import pytest

from dinf import discriminator
from dinf.misc import tree_car, tree_shape


def random_dataset(size, seed=1234):
    """Make up a test dataset."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
    input_shape = (size, 32, 32, 1)
    x = {
        "x": jax.random.randint(
            key1, shape=input_shape, minval=0, maxval=128, dtype=np.int8
        )
    }
    y = jax.random.randint(key2, shape=(size,), minval=0, maxval=2, dtype=np.int8)
    x_shape = {"x": np.array(input_shape[1:])}
    return x, y, x_shape


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
    def test_from_input_shape(self):
        rng = np.random.default_rng(1234)
        input_shape = {"x": np.array((64, 64, 1))}
        discriminator.Discriminator.from_input_shape(input_shape, rng)

    def test_fit(self):
        train_x, train_y, input_shape = random_dataset(50)
        val_x, val_y, _ = random_dataset(40)

        rng = np.random.default_rng(1234)
        d1 = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d1.fit(rng, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
        chex.assert_tree_all_finite(d1.variables)

        # Should be deterministic.
        rng = np.random.default_rng(1234)
        d2 = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d2.fit(rng, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
        chex.assert_trees_all_close(d1.variables, d2.variables)

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
        d1 = discriminator.Discriminator.from_input_shape((30, 40, 1), rng)
        filename = tmp_path / "discriminator.pkl"
        d1.to_file(filename)
        d2 = discriminator.Discriminator.from_file(filename)
        chex.assert_trees_all_close(d1.variables, d2.variables)

    def test_predict(self):
        train_x, train_y, input_shape = random_dataset(50)
        val_x, val_y, _ = random_dataset(30)

        rng = np.random.default_rng(1234)
        d = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d.fit(rng, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)

        y1 = d.predict(val_x)
        assert np.shape(y1) == (30,)
        assert all(y1 >= 0)
        assert all(y1 <= 1)

        # Should be deterministic.
        y2 = d.predict(val_x)
        chex.assert_trees_all_close(y1, y2)


class TestBatchify:
    def test_batchify(self):
        n = 128
        dataset = {"a": np.zeros(n), "b": {"c": np.ones(n)}}

        def size(a):
            """Size of the leading dimension of a feature."""
            sz = np.array(jax.tree_flatten(tree_car(tree_shape(a)))[0])
            # All features should have the same size for the leading dimension.
            assert np.all(sz[0] == sz[1:])
            return sz[0]

        # batch_size = 1
        batches = list(discriminator.batchify(dataset, 1))
        assert len(batches) == n
        for batch in batches:
            chex.assert_trees_all_equal_structs(dataset, batch)
            assert size(batch) == 1

        # batch_size divides n
        batch_size = n // 4
        batches = list(discriminator.batchify(dataset, batch_size))
        assert len(batches) == n // batch_size
        for batch in batches:
            chex.assert_trees_all_equal_structs(dataset, batch)
            assert size(batch) == batch_size

        # batch_size > n
        batches = list(discriminator.batchify(dataset, n + 5))
        assert len(batches) == 1
        for batch in batches:
            chex.assert_trees_all_equal_structs(dataset, batch)
            assert size(batch) == n

        # batch_size doesn't divide n evenly
        batch_size = 13
        remainder = n % batch_size
        assert remainder > 0
        batches = list(discriminator.batchify(dataset, batch_size))
        assert len(batches) == n // batch_size + 1
        for batch in batches[:-1]:
            chex.assert_trees_all_equal_structs(dataset, batch)
            assert size(batch) == batch_size
        last_batch = batches[-1]
        chex.assert_trees_all_equal_structs(dataset, last_batch)
        assert size(last_batch) == remainder
