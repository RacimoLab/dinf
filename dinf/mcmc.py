import functools
from typing import Callable
import warnings

import jax
import jax.numpy as jnp
import numpy as np


def rw_proposal(key, x, step=1):
    proposal = x + step * jax.random.normal(key, x.shape, x.dtype)
    return proposal


@functools.partial(jax.jit, static_argnums=(0, 1))
def mh_run(
    num_steps: int, log_prob: Callable, initial_position: np.ndarray, *, seed: int
):
    def step(i, state):
        key, samples, samples_lp, acceptance_count = state
        current_position = samples[i - 1]
        current_lp = samples_lp[i - 1]
        key1, key2, key3 = jax.random.split(key, 3)
        proposed_position = rw_proposal(key1, current_position)
        proposed_lp = log_prob(proposed_position)
        u = jax.random.uniform(key2)
        accept = jnp.log(u) < proposed_lp - current_lp
        position = jnp.where(accept, proposed_position, current_position)
        position_lp = jnp.where(accept, proposed_lp, current_lp)
        return (
            key3,
            samples.at[i].set(position),
            samples_lp.at[i].set(position_lp),
            acceptance_count + accept.astype(int),
        )
        return

    key = jax.random.PRNGKey(seed)
    samples = jnp.empty((num_steps + 1, *initial_position.shape))
    samples = samples.at[0].set(initial_position)
    samples_lp = jnp.empty(num_steps + 1)
    samples_lp = samples_lp.at[0].set(log_prob(initial_position))
    init_state = (
        key,
        samples,
        samples_lp,
        0,
    )
    out_state = jax.lax.fori_loop(1, num_steps + 1, step, init_state)
    _, samples, samples_lp, acceptance_count = out_state
    acceptance_rate = acceptance_count / num_steps
    return samples[1:], samples_lp[1:], acceptance_rate

def _surrogate_log_prob(theta, surrogate, parameters):
    num_params, = theta.shape
    assert num_params == len(parameters)
    in_bounds = parameters.bounds_contain(theta)
    return np.where(in_bounds, surrogate.predict(theta), -np.inf)

def rw_mcmc(
    start,
    surrogate,
    genobuilder,
    steps: int,
    rng,
):
    samples, lp, acceptance_rate = mh_run(
        steps,
        log_prob=functools.partial(
            _surrogate_log_prob, surrogate=surrogate, parameters=parameters
        ),
        initial_position=start,
        seed=rng.integers(low=0, high=2**31),
    )
    # XXX: use logger
    print("MCMC acceptance rate", acceptance_rate)
    datadict = {
        "posterior": {
            p: np.expand_dims(np.array(samples[..., j]), 0)
            for j, p in enumerate(genobuilder.parameters)
        },
        "sample_stats": {
            "lp": np.expand_dims(np.array(lp), 0),
            "acceptance_rate": acceptance_rate,
        },
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="More chains.*than draws",
            module="arviz",
        )
        dataset = az.from_dict(**datadict)
    return dataset


if __name__ == "__main__":
    mu = [0.1, -2.1]
    std = [1.0, 1.5]
    num_steps = 50_000

    def log_prob(x, loc=np.array(mu), scale=np.array(std)):
        return jnp.sum(jax.scipy.stats.norm.logpdf(x, loc, scale))

    init = np.array([4.2, -1.0])
    samples, lp, acceptance = mh_run(
        num_steps,
        log_prob,
        init,
        seed=np.random.default_rng().integers(low=0, high=2 ** 31),
    )
    samples = np.array(samples)
    lp = np.array(lp)
    assert samples.shape == (num_steps, len(init))
    assert len(lp) == num_steps

    print("acceptance ratio", acceptance)
    est_mu = np.mean(samples[len(samples) // 2 :], axis=0)
    est_std = np.std(samples[len(samples) // 2 :], axis=0)
    print("mu", *zip(mu, est_mu))
    print("std", *zip(std, est_std))
