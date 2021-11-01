import numpy as np
import jax.tree_util

from dinf import discriminator


def pytree_assert_equal(xs, ys):
    """assert xs == ys, where xs and ys are pytrees"""
    jax.tree_util.tree_all(
        jax.tree_util.tree_map(np.testing.assert_array_equal, xs, ys)
    )


def random_dataset(size):
    """Make up a test dataset."""
    rng = np.random.default_rng(1234)
    input_shape = (size, 32, 32, 1)
    x = rng.integers(low=0, high=128, size=input_shape, dtype=np.int8)
    y = rng.integers(low=0, high=1, size=size, dtype=np.int8, endpoint=True)
    return x, y


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

        pytree_assert_equal(d1.variables, d2.variables)


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
