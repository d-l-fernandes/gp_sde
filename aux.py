from collections import OrderedDict
from functools import partial
import random as py_random

import jax
import jax.nn as nn
import jax.numpy as np
import jax.random as random
import numpyro
from jax.config import config
from jax.experimental import stax
from numpyro.distributions.continuous import MultivariateNormal

from gp import kernels
from jax_aux import aux_math

config.update("jax_enable_x64", True)


class GammaParams:
    def __init__(self, dims, scope_var: OrderedDict):

        self.dims = dims
        key = random.PRNGKey(py_random.randrange(9999))

        key, subkey = random.split(key)
        scope_var["concentration"] = random.normal(subkey, shape=[self.dims])

        key, subkey = random.split(key)

        scope_var["rate"] = random.normal(subkey, shape=[self.dims])

    @staticmethod
    def build(scope_var: OrderedDict):
        return scope_var["concentration"], scope_var["rate"]

    def __call__(self, scope_var: OrderedDict):
        return 0.5 * np.sum(np.square(scope_var["concentration"]) + np.square(scope_var["rate"]))


class ExponentiatedQuadraticKernel:
    def __init__(self, dims_in, dims_out, scope_var: OrderedDict, time=None):
        self.kernel_var = None
        self.time = time
        self.key = random.PRNGKey(py_random.randrange(9999))
        self.key, subkey = random.split(self.key)
        scope_var["log_amplitude_latent"] = random.normal(subkey, shape=[dims_out])

        self.key, subkey = random.split(self.key)
        scope_var["log_kernel_weights_latent"] = random.normal(subkey, [dims_out, dims_in])

    def build(self, sc: OrderedDict):
        if self.time is None:
            return kernels.ExponentiatedQuadratic(sc["log_amplitude_latent"], sc["log_kernel_weights_latent"])
        else:
            return kernels.ExponentiatedQuadratic(sc["log_amplitude_latent"], sc["log_kernel_weights_latent"],
                                                  time=self.time)

    @staticmethod
    def __call__(sc: OrderedDict):
        return 0.5 * (np.sum(np.square(sc["log_amplitude_latent"])) +
                      np.sum(np.square(sc["log_kernel_weights_latent"])))


class MultipliedKernel:
    def __init__(self, kernel, multiplication_factor):
        self.kernel_var = kernel
        # multiplication_factor is a vector
        self.multiplication_factor = np.expand_dims(np.expand_dims(multiplication_factor, -1), -1)

    @partial(jax.jit, static_argnums=(0,))
    def matrix(self, x0, x1):
        return self.multiplication_factor * self.kernel_var.matrix(x0, x1)


class LikelihoodVariance:
    def __init__(self, dims, scope_var: OrderedDict):
        key = random.PRNGKey(py_random.randrange(9999))
        scope_var["log_signal_variance"] = random.normal(key, shape=[dims])

    @staticmethod
    def build(sc: OrderedDict):
        return sc["log_signal_variance"]

    def __call__(self, sc: OrderedDict):
        return 0.5 * np.sum(np.square(sc["log_signal_variance"]))


class Encoder:
    def __init__(self, input_dims, output_dims, scope_var: OrderedDict):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.key = random.PRNGKey(py_random.randrange(9999))

        init_random_params, self.predict = stax.serial(
            stax.Flatten,
            stax.Dense(input_dims * 4,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Softplus,
            stax.Dense(input_dims * 4,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Softplus,
            stax.Dense(output_dims * 2,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
        )

        _, init_params = init_random_params(self.key, (-1, input_dims))

        scope_var["encoder_params"] = init_params

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, y, sc, multiplicative_factor=None):
        net = self.predict(sc["encoder_params"], y)
        if multiplicative_factor is None:
            scale_tril = aux_math.diag(nn.softplus(net[..., self.output_dims:]))
        else:
            scale_tril = np.einsum("ab,cbd->cad",
                                   multiplicative_factor,
                                   aux_math.diag(nn.softplus(net[..., self.output_dims:])))
        return net[..., :self.output_dims], scale_tril

    def sample(self, mean, scale_tril):
        self.key, subkey = random.split(self.key)
        z = numpyro.sample('z',
                           MultivariateNormal(loc=mean,
                                              scale_tril=scale_tril),
                           rng_key=subkey)
        return z

    def kl_divergence(self, mean, scale_tril):
        return np.mean(aux_math.kl_divergence_multivariate_normal(
            mean,
            scale_tril,
            np.zeros_like(mean),
            np.tile(np.expand_dims(np.eye(self.output_dims), axis=0), [np.shape(mean)[0], 1, 1])
        ))
