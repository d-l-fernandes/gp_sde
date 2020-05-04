from functools import partial

import jax
import jax.numpy as np
from jax import lax
from jax import ops
from jax.config import config

config.update("jax_enable_x64", True)


class ExponentiatedQuadratic:
    """The ExponentiatedQuadratic kernel.
    Sometimes called the "squared exponential", "Gaussian" or "radial basis
    function", this kernel function has the form
      ```none
      k(x, y) = amplitude**2 * exp(-sum(length_scale_i^-2||x_i - y_i||**2)_{i=1}^D / 2))
      ```
    where the double-bars represent vector length (ie, Euclidean, or L2 norm).
    """

    def __init__(self,
                 amplitude=None,
                 length_scale=None,
                 use_log_squared=True,
                 time=0.,
                 ):
        self._amplitude = amplitude
        self._length_scale = length_scale
        self._use_log_squared = use_log_squared
        self._time = time

    @property
    def amplitude(self):
        """Amplitude parameter."""
        return lax.cond(self._use_log_squared,
                        None,
                        lambda _: np.exp(self._amplitude / 2.),
                        None,
                        lambda _: self._amplitude
                        )

    @property
    def length_scale(self):
        """Length scale parameter."""
        length_scale = lax.cond(self._use_log_squared,
                                None,
                                lambda _: np.exp(self._length_scale / 2.),
                                None,
                                lambda _: self._length_scale
                                )
        return ops.index_add(length_scale, ops.index[:, -1], self._time)


    @partial(jax.jit, static_argnums=(0,))
    def matrix(self, x, z):
        xx = np.sum(self.length_scale[:, None, :] ** 2 * x * x, axis=-1, keepdims=True)
        zz = np.sum(self.length_scale[:, None, :] ** 2 * z * z, axis=-1, keepdims=True)
        xz = np.einsum('abc, acd->abd',
                       self.length_scale[:, None, :] * x,
                       np.transpose(self.length_scale[:, None, :] * z, (0, 2, 1)))
        exponent = np.exp(-0.5 * (xx + np.transpose(zz, (0, 2, 1)) - 2 * xz))
        amplitude = self.amplitude[:, None, None]
        exponent *= amplitude**2

        return exponent
