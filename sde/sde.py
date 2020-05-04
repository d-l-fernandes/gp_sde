import jax
import jax.numpy as np
from jax.config import config
from jax import ops
from jax import lax
from functools import partial

config.update("jax_enable_x64", True)


class SDE:
    def __init__(self,
                 delta_t,
                 solver,
                 num_dim):

        self.delta_t = delta_t
        self.solver = solver
        self.num_dim = num_dim

        self.drift_function = None
        self.diffusion_function = None

        self.solver = self.solver(self.delta_t, self.num_dim)

    # LAX SCAN VERSION - Works and it's not slow!!!!!!!!
    @partial(jax.jit, static_argnums=(0, 2))
    def __call__(self, x_init, num_steps, t_init=-0.5):

        self.solver.drift_function = self.drift_function
        self.solver.diffusion_function = self.diffusion_function

        paths_x = np.tile(np.expand_dims(x_init, axis=0), [num_steps + 1, 1, 1])

        def scan_fn(carry, _):
            step_input_x, time = carry
            output_x = self.solver.step(step_input_x, time)

            time += self.delta_t
            return (output_x, time), output_x

        (final_output_x, final_t), paths_x_minus_initial = \
            lax.scan(scan_fn, (x_init, t_init), None, length=num_steps)

        paths_x = ops.index_update(paths_x, ops.index[1:], paths_x_minus_initial)
        return final_output_x, paths_x
