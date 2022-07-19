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
        with pytest.raises(ValueError, match="truth.*not in bounds"):
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

    def test_name(self):
        # Param name should be set after construction.
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        for k, p in params.items():
            assert k == p.name

        # Param names shouldn't be altered by construction.
        params = Parameters(
            a=Param(low=0, high=10, name="a"),
            b=Param(low=-1e6, high=1e6, name="b"),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        for k, p in params.items():
            assert k == p.name

    def test_name_mismatch(self):
        with pytest.raises(ValueError, match="Name mismatch"):
            Parameters(a=Param(low=0, high=10, name="b"))

    def test_getitem(self):
        a = Param(low=0, high=10)
        params = Parameters(a=a)
        a2 = params["a"]
        assert a2.name == "a"
        assert a2.low == 0
        assert a2.high == 10
        assert a2.truth is None
        with pytest.raises(KeyError):
            params["b"]

    def test_iter(self):
        assert tuple(Parameters()) == ()
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        assert tuple(params) == ("a", "b", "c")
        assert tuple(params.keys()) == ("a", "b", "c")

    def test_len(self):
        assert len(Parameters()) == 0
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        assert len(params) == 3

    def test_mapping_protocol(self):
        parameters = Parameters(
            a=Param(low=0, high=1),
            b=Param(low=5, high=100),
            c=Param(low=1e-6, high=1e-3),
        )
        assert len(parameters) == 3
        # Iterate over parameters.
        assert list(parameters) == ["a", "b", "c"]
        assert [p.high for p in parameters.values()] == [1, 100, 1e-3]
        # Lookup a Param by name.
        a = parameters["a"]
        assert a.low == 0
        assert a.high == 1
        assert a.name == "a"

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

    def test_custom_param(self):
        class UnboundedParam(Param):
            def __post_init__(self):
                pass

            def bounds_contain(self, x):
                return np.ones_like(x, dtype=bool)

        Parameters(p=UnboundedParam(low=-np.inf, high=np.inf))

    def test_bad_custom_param(self):
        class MyParam:
            pass

        with pytest.raises(TypeError):
            Parameters(p=MyParam())

    def test_transform_preserves_order(self):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        size = 10_000
        rng = np.random.default_rng(1234)
        thetas = params.draw_prior(size, rng)
        thetas_unbounded = params.transform(thetas)
        for j in range(thetas.shape[-1]):
            idx1 = np.argsort(thetas[:, j])
            idx2 = np.argsort(thetas_unbounded[:, j])
            np.testing.assert_array_equal(idx1, idx2)

    def test_transform_itransform_inverses(self):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        size = 10_000
        rng = np.random.default_rng(1234)
        thetas = params.draw_prior(size, rng)
        thetas_unbounded = params.transform(thetas)
        thetas2 = params.itransform(thetas_unbounded)
        np.testing.assert_allclose(thetas, thetas2)

    def test_itransform_bounded(self):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        size = 10_000
        rng = np.random.default_rng(1234)
        for lo, hi in ((0, 1), (-1000, -50), (50, 1000), (-1e50, 1e50)):
            U = rng.uniform(low=lo, high=hi, size=(size, len(params)))
            thetas = params.itransform(U)
            assert np.all(params.bounds_contain(thetas))

    def test_truncate_bounded(self):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        size = 10_000
        rng = np.random.default_rng(1234)
        for lo, hi in ((0, 1), (-1000, -50), (50, 1000), (-1e50, 1e50)):
            U = rng.uniform(low=lo, high=hi, size=(size, len(params)))
            thetas = params.truncate(U)
            assert np.all(params.bounds_contain(thetas))
            # Truncation is idempotent.
            thetas2 = params.truncate(thetas)
            np.testing.assert_array_equal(thetas, thetas2)

    def test_truncate_doesnt_change_bounded_values(self):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        size = 10_000
        rng = np.random.default_rng(1234)
        thetas = params.draw_prior(size, rng)
        thetas2 = params.truncate(thetas)
        np.testing.assert_array_equal(thetas, thetas2)

    def test_truncate_examples(self):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        thetas = params.truncate(
            [
                (-1, -1e8, -1),
                (100, 1e8, 1e6),
                (-1, 0, 0),
                (100, 100, 100),
                (-1, 0, 1e6),
            ]
        )
        np.testing.assert_array_equal(
            thetas,
            [
                (0, -1e6, 10_000),
                (10, 1e6, 20_000),
                (0, 0, 10_000),
                (10, 100, 10_000),
                (0, 0, 20_000),
            ],
        )

    def test_reflect_bounded(self):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        size = 10_000
        rng = np.random.default_rng(1234)
        for lo, hi in ((0, 1), (-1000, -50), (50, 1000), (-1e50, 1e50)):
            U = rng.uniform(low=lo, high=hi, size=(size, len(params)))
            thetas = params.reflect(U)
            assert np.all(params.bounds_contain(thetas))
            # Reflection is idempotent.
            thetas2 = params.reflect(thetas)
            np.testing.assert_array_equal(thetas, thetas2)

    def test_reflect_doesnt_change_bounded_values(self):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        size = 10_000
        rng = np.random.default_rng(1234)
        thetas = params.draw_prior(size, rng)
        thetas2 = params.reflect(thetas)
        np.testing.assert_array_equal(thetas, thetas2)

    def test_reflect_examples(self):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        thetas = params.reflect(
            [
                (-1, -1.1e6, -1),
                (15, 1e8, 25_000),
                (-2, 0, 0),
                (100, 100, 100),
                (-3, 0, 1e6),
            ]
        )
        np.testing.assert_array_equal(
            thetas,
            [
                (1, -9e5, 20_000),
                (5, -1e6, 15_000),
                (2, 0, 20_000),
                (0, 100, 19_900),
                (3, 0, 10_000),
            ],
        )
