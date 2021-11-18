import numpy as np
import jax
import chex
import pytest

from dinf import discriminator


def random_dataset(size, seed=1234):
    """Make up a test dataset."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
    input_shape = (size, 32, 32, 1)
    x = jax.random.randint(key1, shape=input_shape, minval=0, maxval=128, dtype=np.int8)
    y = jax.random.randint(key2, shape=(size,), minval=0, maxval=2, dtype=np.int8)
    return x, y


class TestExchangeableCNN:
    @pytest.mark.parametrize("train", [True, False])
    def test_cnn(self, train: bool):
        cnn = discriminator.ExchangeableCNN()
        x, _ = random_dataset(50)
        variables = cnn.init(jax.random.PRNGKey(0), x, train=False)
        y, new_variables = cnn.apply(
            variables, x, train=train, mutable=["batch_stats"] if train else []
        )
        chex.assert_rank(y, 1)
        chex.assert_shape(y, (x.shape[0],))


class TestDiscriminator:
    def test_fit(self):
        train_x, train_y = random_dataset(50)
        val_x, val_y = random_dataset(50)
        input_shape = train_x.shape[1:]

        rng = np.random.default_rng(1234)
        d1 = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d1.fit(rng, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)

        rng = np.random.default_rng(1234)
        d2 = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d2.fit(rng, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)

        chex.assert_tree_all_finite(d1.variables)
        chex.assert_trees_all_close(d1.variables, d2.variables)

    @pytest.mark.usefixtures("capsys")
    def test_summary(self, capsys):
        rng = np.random.default_rng(1234)
        d = discriminator.Discriminator.from_input_shape((30, 40, 1), rng)
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
        train_x, train_y = random_dataset(50)
        val_x, val_y = random_dataset(50)
        input_shape = train_x.shape[1:]

        rng = np.random.default_rng(1234)
        d = discriminator.Discriminator.from_input_shape(input_shape, rng)
        d.fit(rng, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)

        y = d.predict(val_x)
        chex.assert_rank(y, 1)
        chex.assert_shape(y, (val_x.shape[0],))
        assert all(y >= 0)
        assert all(y <= 1)


class TestBatchify:
    def test_batchify(self):
        n = 128
        dataset = dict(a=np.zeros(n), b=np.ones(n))

        # batch_size = 1
        batches = list(discriminator.batchify(dataset, 1))
        assert len(batches) == n
        for batch in batches:
            assert set(["a", "b"]) == batch.keys()
            assert len(batch["a"]) == len(batch["b"]) == 1

        # batch_size divides n
        batch_size = n // 4
        batches = list(discriminator.batchify(dataset, batch_size))
        assert len(batches) == n // batch_size
        for batch in batches:
            assert set(["a", "b"]) == batch.keys()
            assert len(batch["a"]) == len(batch["b"]) == batch_size

        # batch_size > n
        batches = list(discriminator.batchify(dataset, n + 5))
        assert len(batches) == 1
        for batch in batches:
            assert set(["a", "b"]) == batch.keys()
            assert len(batch["a"]) == len(batch["b"]) == n

        # batch_size doesn't divide n evenly
        batch_size = 13
        remainder = n % batch_size
        assert remainder > 0
        batches = list(discriminator.batchify(dataset, batch_size))
        assert len(batches) == n // batch_size + 1
        for batch in batches[:-1]:
            assert set(["a", "b"]) == batch.keys()
            assert len(batch["a"]) == len(batch["b"]) == batch_size
        last_batch = batches[-1]
        assert set(["a", "b"]) == last_batch.keys()
        assert len(last_batch["a"]) == len(last_batch["b"]) == remainder
