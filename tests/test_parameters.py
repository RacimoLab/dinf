import numpy as np
import pytest

from dinf.parameters import Param, Parameters


class TestParam:
    def test_init(self):
        Param(low=0, high=10)
        Param(low=-1e6, high=1e6)
        Param(low=10_000, high=20_000, truth=15_000)

    @pytest.mark.parametrize(
        "low,high",
        [
            (-np.inf, 0),
            (0, np.inf),
            (-np.inf, np.inf),
            (np.nan, 0),
            (0, np.nan),
            (np.nan, np.nan),
        ],
    )
    def test_bad_bounds(self, low, high):
        with pytest.raises(ValueError, match="Bounds must be finite."):
            Param(low=low, high=high)

    @pytest.mark.parametrize("x", (10, 15, 20))
    def test_bounds_contain_true(self, x):
        assert Param(low=10, high=20).bounds_contain(x)

    @pytest.mark.parametrize("x", (9, 21, np.inf))
    def test_bounds_contain_false(self, x):
        assert not Param(low=10, high=20).bounds_contain(x)

    def test_bounds_contain_vector(self):
        p = Param(low=10, high=20)
        for xs, expected in [
            ([1, 2, 32], [False, False, False]),
            ([10, 2, 32], [True, False, False]),
            ([10, 20, 32], [True, True, False]),
            ([10, 20, 13], [True, True, True]),
            ([10, 2, 13], [True, False, True]),
            (np.array([1, 2, 13]), [False, False, True]),
        ]:
            in_bounds = p.bounds_contain(xs)
            assert len(xs) == len(in_bounds)
            assert np.all(in_bounds == expected)

    @pytest.mark.parametrize("truth", (15, -15))
    def test_truth_out_of_bounds(self, truth):
        with pytest.raises(ValueError, match="True value.*not in bounds"):
            Param(low=-10, high=10, truth=truth)

    @pytest.mark.parametrize("size", (1, 50))
    def test_draw_prior(self, size):
        rng = np.random.default_rng(1234)
        for p in [
            Param(low=0, high=10),
            Param(low=-1e6, high=1e6),
            Param(low=10_000, high=20_000, truth=15_000),
        ]:
            xs = p.draw_prior(size, rng)
            assert xs.shape == (size,)
            assert np.all(p.bounds_contain(x) for x in xs)


class TestParameters:
    def test_init(self):
        Parameters()
        Parameters(a=Param(low=0, high=10))
        Parameters(a=Param(low=0, high=10), b=Param(low=0, high=10))
        Parameters(a=Param(low=0, high=10, truth=5))
        Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )

    def test_getitem(self):
        a = Param(low=0, high=10)
        params = Parameters(a=a)
        assert params["a"] == a
        with pytest.raises(KeyError):
            params["b"]

    def test_iter(self):
        assert tuple(Parameters()) == ()
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        assert tuple(params) == tuple(params.keys())

    def test_len(self):
        assert len(Parameters()) == 0
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        assert len(params) == 3

    def test_bounds_contain_true(self):
        rng = np.random.default_rng(1234)

        params = Parameters(a=Param(low=0, high=10))
        xs = [[x] for x in params["a"].draw_prior(13, rng)]
        in_bounds = params.bounds_contain(xs)
        assert len(xs) == len(in_bounds)
        assert np.all(in_bounds)

        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        xs = np.array([p.draw_prior(13, rng) for p in params.values()]).T
        in_bounds = params.bounds_contain(xs)
        assert len(xs) == len(in_bounds)
        assert np.all(in_bounds)

    def test_bounds_contain_false(self):
        params = Parameters(a=Param(low=0, high=10))
        xs = [[x] for x in [-1, -10, 12, 100]]
        in_bounds = params.bounds_contain(xs)
        assert len(xs) == len(in_bounds)
        assert not np.any(in_bounds)

        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        xs = [
            [-1, 0, 15_000],
            [-1, -1e12, 15_000],
            [-1, -1e12, 115_000],
            [-1, 0, 115_000],
            [1, 0, 115_000],
        ]
        in_bounds = params.bounds_contain(xs)
        assert len(xs) == len(in_bounds)
        assert not np.any(in_bounds)

    def test_bounds_contain_mixed(self):
        params = Parameters(
            a=Param(low=0, high=5),
            b=Param(low=10, high=20),
        )
        for xs, expected in [
            (
                [[0, 15], [-1, -1], [6, 15]],
                [True, False, False],
            ),
            (
                [[0, 25], [1, 11], [1, 25]],
                [False, True, False],
            ),
            (
                [[0, 25], [-1, 9], [1, 20]],
                [False, False, True],
            ),
            (
                [[0, 15], [-1, 9], [1, 20]],
                [True, False, True],
            ),
        ]:
            in_bounds = params.bounds_contain(xs)
            assert len(xs) == len(in_bounds)
            assert np.all(in_bounds == expected)

    @pytest.mark.parametrize("size", (1, 50))
    def test_draw_prior(self, size):
        rng = np.random.default_rng(1234)
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        xs = params.draw_prior(size, rng)
        assert len(xs) == size
        in_bounds = params.bounds_contain(xs)
        assert len(in_bounds) == size
        assert np.all(in_bounds)

    def test_update_posterior(self):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        assert params._posterior is None
        rng = np.random.default_rng(1234)
        xs = params.draw_prior(10, rng)
        params.update_posterior(xs)
        assert np.all(params._posterior == xs)

    @pytest.mark.parametrize("size1,size2", [(1, 10), (50, 1), (50, 500)])
    def test_draw(self, size1, size2):
        rng = np.random.default_rng(1234)
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        # draw from the prior
        xs = params.draw(size1, rng)
        assert len(xs) == size1
        in_bounds = params.bounds_contain(xs)
        assert len(in_bounds) == size1
        assert np.all(in_bounds)

        params.update_posterior(xs)
        # draw from the posterior
        xs = params.draw(size2, rng)
        assert len(xs) == size2
        in_bounds = params.bounds_contain(xs)
        assert len(in_bounds) == size2
        assert np.all(in_bounds)
