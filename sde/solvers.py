from functools import partial
import random as py_random

import jax
import jax.numpy as np
import jax.random as random
from jax.config import config
from jax import lax
from numpyro.distributions.continuous import MultivariateNormal
import numpyro

from jax_aux import aux_math

config.update("jax_enable_x64", True)


class BaseSolver:
    def __init__(self, delta_t, beta_dims):
        self.delta_t = delta_t
        self.beta_dims = beta_dims
        self.key = random.PRNGKey(py_random.randrange(9999))

        """
        Diffusion is always a function.
        Should have shape [n_batch, x_0.shape[-1], x_0.shape[-1]]
        """
        self.drift_function = None
        self.diffusion_function = None

    def step(self, x_0, time):
        pass


class EulerMaruyamaSolver(BaseSolver):
    def __init__(self, delta_t, beta_dims):
        super(EulerMaruyamaSolver, self).__init__(delta_t, beta_dims)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, x_0, time):
        self.key, subkey = random.split(self.key)

        shape = np.array([np.shape(x_0)[0], self.beta_dims], dtype=np.int8)
        delta_beta = numpyro.sample("delta_gamma_beta",
                                    MultivariateNormal(
                                        loc=np.zeros(shape),
                                        scale_tril=np.sqrt(self.delta_t) * aux_math.diag(np.ones(shape))),
                                    rng_key=subkey)
        x_1 = x_0 + self.drift_function(x_0, time) * self.delta_t + \
            np.einsum("abc,ac->ab", self.diffusion_function(x_0, time), delta_beta)

        return x_1


class StrongOrder3HalfsSolver(BaseSolver):
    """
    See page 151 of the Applied Stochastic Differential Equations book
    """
    def __init__(self, delta_t, beta_dims):
        super(StrongOrder3HalfsSolver, self).__init__(delta_t, beta_dims)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, x_0, time):
        shape = [np.shape(x_0)[0], self.beta_dims]

        # Vector of zeros
        beta_mean_vector = np.concatenate((np.zeros(shape), np.zeros(shape)), axis=1)

        # Covariance matrix for the betas and gammas
        beta_covariance_top_left = self.delta_t ** 3 / 3 * aux_math.diag(np.ones(shape))
        beta_covariance_top_right = self.delta_t ** 2 / 2 * aux_math.diag(np.ones(shape))
        beta_covariance_bottom_right = self.delta_t * aux_math.diag(np.ones(shape))
        beta_covariance_top = np.concatenate((beta_covariance_top_left, beta_covariance_top_right), axis=2)
        beta_covariance_bottom = np.concatenate((beta_covariance_top_right, beta_covariance_bottom_right), axis=2)
        beta_covariance = np.concatenate((beta_covariance_top, beta_covariance_bottom), axis=1)

        self.key, subkey = random.split(self.key)
        delta_gamma_beta = numpyro.sample("delta_gamma_beta",
                                          MultivariateNormal(loc=beta_mean_vector,
                                                             covariance_matrix=beta_covariance),
                                          rng_key=subkey)

        delta_gamma = delta_gamma_beta[:, 0:self.beta_dims]
        delta_beta = delta_gamma_beta[:, self.beta_dims:]

        # Supporting values
        drift_0 = self.drift_function(x_0, time) * self.delta_t

        init_x_1 = x_0 + drift_0 + np.einsum("abc,ac->ab", self.diffusion_function(x_0, time), delta_beta)

        def scan_fn(carry, s):
            x_1 = carry
            x_0_plus = \
                x_0 + drift_0 / self.beta_dims + \
                self.diffusion_function(x_0, time)[..., s] * np.sqrt(self.delta_t)
            x_0_minus = \
                x_0 + drift_0 / self.beta_dims - \
                self.diffusion_function(x_0, time)[..., s] * np.sqrt(self.delta_t)

            drift_0_plus = self.drift_function(x_0_plus, time)
            drift_0_minus = self.drift_function(x_0_minus, time)
            x_1 += 0.25 * self.delta_t * (drift_0_plus + drift_0_minus)
            x_1 -= 0.5 * drift_0
            x_1 += \
                1. / (2 * np.sqrt(self.delta_t)) * (drift_0_plus-drift_0_minus) * \
                np.expand_dims(delta_gamma[:, s], axis=-1)

            return x_1, None

        final_x_1, _ = lax.scan(scan_fn, init_x_1, np.arange(self.beta_dims))

        return final_x_1
