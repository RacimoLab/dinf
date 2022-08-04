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

    def test_bounds_contain_1d(self):
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

    def test_bounds_contain_2d(self):
        p = Param(low=10, high=20)
        for xs, expected in [
            ([[1, 2, 32], [10, 2, 32]], [[False, False, False], [True, False, False]]),
            ([[10, 20, 32], [10, 20, 13]], [[True, True, False], [True, True, True]]),
            (np.array([[1, 2], [13, 13]]), [[False, False], [True, True]]),
        ]:
            in_bounds = p.bounds_contain(xs)
            assert np.shape(xs) == in_bounds.shape
            assert np.all(in_bounds == expected)

    @pytest.mark.parametrize("truth", (15, -15))
    def test_truth_out_of_bounds(self, truth):
        with pytest.raises(ValueError, match="truth.*not in bounds"):
            Param(low=-10, high=10, truth=truth)

    @pytest.mark.parametrize("size", (1, 50, (2, 3)))
    def test_sample_prior(self, size):
        rng = np.random.default_rng(1234)
        for p in [
            Param(low=0, high=10),
            Param(low=-1e6, high=1e6),
            Param(low=10_000, high=20_000, truth=15_000),
        ]:
            xs = p.sample_prior(size=size, rng=rng)
            shape = size if isinstance(size, tuple) else (size,)
            assert xs.shape == shape
            assert np.all(p.bounds_contain(xs))


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

    @pytest.mark.parametrize("size", (1, 13, (2, 3)))
    def test_bounds_contain_true(self, size):
        rng = np.random.default_rng(1234)

        params = Parameters(a=Param(low=0, high=10))
        xs = np.moveaxis(
            np.array([[x] for x in params["a"].sample_prior(size=size, rng=rng)]), 1, -1
        )
        in_bounds = params.bounds_contain(xs)
        assert xs.shape[:-1] == in_bounds.shape
        assert np.all(in_bounds)

        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        xs = np.moveaxis(
            np.array([p.sample_prior(size=size, rng=rng) for p in params.values()]),
            0,
            -1,
        )
        in_bounds = params.bounds_contain(xs)
        assert xs.shape[:-1] == in_bounds.shape
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

    @pytest.mark.parametrize("size", (1, 50, (2, 3)))
    def test_sample_prior(self, size):
        rng = np.random.default_rng(1234)
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        xs = params.sample_prior(size=size, rng=rng)
        assert xs.shape[-1] == len(params)
        shape = size if isinstance(size, tuple) else (size,)
        assert xs.shape[:-1] == shape
        in_bounds = params.bounds_contain(xs)
        assert in_bounds.shape == shape
        assert np.all(in_bounds)

    @pytest.mark.parametrize("size", (1, 100_000))
    def test_sample_ball(self, size):
        cov_factor = 0.001**2
        rng = np.random.default_rng(1234)
        params = Parameters(
            a=Param(low=0, high=10, truth=5),
            b=Param(low=-1e6, high=1e6, truth=1000),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        truths = np.array([p.truth for p in params.values()])
        xs = params.sample_ball(truths, cov_factor=cov_factor, size=size, rng=rng)
        assert xs.shape[-1] == len(params)
        shape = size if isinstance(size, tuple) else (size,)
        assert xs.shape[:-1] == shape
        in_bounds = params.bounds_contain(xs)
        assert in_bounds.shape == shape
        assert np.all(in_bounds)

        if size > 1:
            mean = xs.mean(axis=0)
            np.testing.assert_allclose(truths, mean, rtol=0.01)
            var = xs.var(axis=0) / cov_factor
            var_expected = np.array([(p.high - p.low) ** 2 for p in params.values()])
            np.testing.assert_allclose(var_expected, var, rtol=0.01)

    @pytest.mark.parametrize("size", (2, 50))
    def test_sample_kde(self, size):
        rng = np.random.default_rng(1234)
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        xs = params.sample_prior(size=size, rng=rng)
        assert xs.shape[-1] == len(params)
        vs = params.sample_kde(xs, size=size, rng=rng, probs=np.ones(size))
        assert xs.shape == vs.shape
        assert np.all(params.bounds_contain(vs))

        vs2 = params.sample_kde(xs, size=10 * size, rng=rng, probs=np.ones(size))
        assert len(vs) * 10 == len(vs2)
        assert np.all(params.bounds_contain(vs2))

    def test_sample_kde_mean(self):
        size = 100_000
        rng = np.random.default_rng(1234)
        params = Parameters(
            a=Param(low=0, high=10, truth=5),
            b=Param(low=-1e6, high=1e6, truth=1000),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )

        # Sample a small ball around the truth values.
        truths = np.array([p.truth for p in params.values()])
        xs = params.sample_ball(truths, cov_factor=0.001**2, size=size, rng=rng)
        assert xs.shape[-1] == len(params)
        assert np.all(params.bounds_contain(xs))

        vs = params.sample_kde(xs, size=size, rng=rng, probs=np.ones(size))
        assert xs.shape == vs.shape
        assert np.all(params.bounds_contain(vs))
        np.testing.assert_allclose(xs.mean(axis=0), vs.mean(axis=0), rtol=0.01)

    def test_sample_kde_weighted(self):
        size = 100_000
        rng = np.random.default_rng(1234)
        params = Parameters(
            # Truth values need to be at the midpoint for this test.
            a=Param(low=0, high=10, truth=5),
            b=Param(low=0, high=1e6, truth=0.5e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )

        xs = params.sample_prior(size=size, rng=rng)
        assert xs.shape[-1] == len(params)
        assert np.all(params.bounds_contain(xs))

        # Weight samples by their proximity to the truth values.
        truths = np.array([p.truth for p in params.values()])
        widths = np.array([p.high - p.low for p in params.values()])
        weights = np.linalg.norm((xs - truths) / widths, axis=1)

        vs = params.sample_kde(xs, size=size, rng=rng, probs=weights)
        assert xs.shape == vs.shape
        assert np.all(params.bounds_contain(vs))
        np.testing.assert_allclose(truths, vs.mean(axis=0), rtol=0.01)

    def test_geometric_median(self):
        size = 100_000
        rng = np.random.default_rng(1234)
        params = Parameters(
            a=Param(low=0, high=10, truth=5),
            b=Param(low=-1e6, high=1e6, truth=1000),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )

        # Sample a small ball around the truth values.
        truths = np.array([p.truth for p in params.values()])
        xs = params.sample_ball(truths, cov_factor=0.001**2, size=size, rng=rng)
        assert xs.shape[-1] == len(params)
        assert np.all(params.bounds_contain(xs))

        m = params.geometric_median(xs)
        assert len(m.shape) == 1
        assert m.shape[0] == len(params)
        assert np.all(params.bounds_contain(m))
        np.testing.assert_allclose(xs.mean(axis=0), m, rtol=0.01)

    def test_geometric_median_weighted(self):
        size = 100_000
        rng = np.random.default_rng(1234)
        params = Parameters(
            # Truth values need to be at the midpoint for this test.
            a=Param(low=0, high=10, truth=5),
            b=Param(low=0, high=1e6, truth=0.5e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )

        xs = params.sample_prior(size=size, rng=rng)
        assert xs.shape[-1] == len(params)
        assert np.all(params.bounds_contain(xs))

        # Weight samples by their proximity to the truth values.
        truths = np.array([p.truth for p in params.values()])
        widths = np.array([p.high - p.low for p in params.values()])
        weights = np.linalg.norm((xs - truths) / widths, axis=1)

        m = params.geometric_median(xs, probs=weights)
        assert len(m.shape) == 1
        assert m.shape[0] == len(params)
        assert np.all(params.bounds_contain(m))
        np.testing.assert_allclose(truths, m, rtol=0.01)

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
        thetas = params.sample_prior(size=size, rng=rng)
        thetas_unbounded = params.transform(thetas)
        for j in range(thetas.shape[-1]):
            idx1 = np.argsort(thetas[:, j])
            idx2 = np.argsort(thetas_unbounded[:, j])
            np.testing.assert_array_equal(idx1, idx2)

    @pytest.mark.parametrize("size", (1, 10_000, (2, 3)))
    def test_transform_itransform_inverses(self, size):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        rng = np.random.default_rng(1234)
        thetas = params.sample_prior(size=size, rng=rng)
        thetas_unbounded = params.transform(thetas)
        thetas2 = params.itransform(thetas_unbounded)
        np.testing.assert_allclose(thetas, thetas2)

    @pytest.mark.parametrize("size", (1, 10_000, (2, 3)))
    def test_itransform_bounded(self, size):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        if isinstance(size, tuple):
            shape = size + (len(params),)
        else:
            shape = (size, len(params))
        rng = np.random.default_rng(1234)
        for lo, hi in ((0, 1), (-1000, -50), (50, 1000), (-1e50, 1e50)):
            U = rng.uniform(low=lo, high=hi, size=shape)
            thetas = params.itransform(U)
            assert U.shape == thetas.shape
            assert np.all(params.bounds_contain(thetas))

    @pytest.mark.parametrize("size", (1, 10_000, (2, 3)))
    def test_truncate_bounded(self, size):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        if isinstance(size, tuple):
            shape = size + (len(params),)
        else:
            shape = (size, len(params))
        rng = np.random.default_rng(1234)
        for lo, hi in ((0, 1), (-1000, -50), (50, 1000), (-1e50, 1e50)):
            U = rng.uniform(low=lo, high=hi, size=shape)
            thetas = params.truncate(U)
            assert U.shape == thetas.shape
            assert np.all(params.bounds_contain(thetas))
            # Truncation is idempotent.
            thetas2 = params.truncate(thetas)
            np.testing.assert_array_equal(thetas, thetas2)

    @pytest.mark.parametrize("size", (1, 10_000, (2, 3)))
    def test_truncate_doesnt_change_bounded_values(self, size):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        rng = np.random.default_rng(1234)
        thetas = params.sample_prior(size=size, rng=rng)
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

    @pytest.mark.parametrize("size", (1, 10_000, (2, 3)))
    def test_reflect_bounded(self, size):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        if isinstance(size, tuple):
            shape = size + (len(params),)
        else:
            shape = (size, len(params))
        rng = np.random.default_rng(1234)
        for lo, hi in ((0, 1), (-1000, -50), (50, 1000), (-1e50, 1e50)):
            U = rng.uniform(low=lo, high=hi, size=shape)
            thetas = params.reflect(U)
            assert U.shape == thetas.shape
            assert np.all(params.bounds_contain(thetas))
            # Reflection is idempotent.
            thetas2 = params.reflect(thetas)
            np.testing.assert_array_equal(thetas, thetas2)

    @pytest.mark.parametrize("size", (1, 10_000, (2, 3)))
    def test_reflect_doesnt_change_bounded_values(self, size):
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        rng = np.random.default_rng(1234)
        thetas = params.sample_prior(size=size, rng=rng)
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

    @pytest.mark.parametrize("n", (1, 50))
    def test_top_n(self, n):
        size = 10_000
        params = Parameters(
            a=Param(low=0, high=10),
            b=Param(low=-1e6, high=1e6),
            c=Param(low=10_000, high=20_000, truth=15_000),
        )
        rng = np.random.default_rng(1234)
        thetas = params.sample_prior(size=size, rng=rng)
        p = np.linspace(1, 0, size)
        top_thetas, top_probs = params.top_n(thetas, probs=p, n=n)
        assert top_thetas.shape == (n, len(params))
        assert len(top_probs) == n
        for th, pj in zip(thetas[:n], p[:n]):
            assert th in top_thetas
            assert pj in top_probs
            (i,) = np.where(top_probs == pj)[0]
            assert all(th == top_thetas[i])
