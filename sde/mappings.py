from collections import OrderedDict
from functools import partial

import jax
import jax.nn as nn
import jax.numpy as np
import jax.random as random
from jax.config import config
from jax.experimental import stax

import random as py_random

config.update("jax_enable_x64", True)


class BaseMap:
    def __init__(self, input_dims, output_dims, scope_var: OrderedDict):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.scope = scope_var

    @partial(jax.jit, static_argnums=(0,))
    def _map(self, input_array, time, sc: OrderedDict):
        """
        returns array of shape [output_dims] - non-batched version of map
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def map(self, input_array, time, sc: OrderedDict):
        """
        returns array of shape [n_batch, output_dims]
        """
        return jax.vmap(self._map, (0, None, None))(input_array, time, sc)

    @partial(jax.jit, static_argnums=(0,))
    def time_derivative_map(self, input_array, time, sc: OrderedDict):
        """
        returns array of shape [n_batch, output_dims]
        """
        return jax.vmap(jax.jacfwd(self._map, 1), (0, None, None))(input_array, time, sc)

    @partial(jax.jit, static_argnums=(0,))
    def first_derivative_map(self, input_array, time, sc: OrderedDict):
        """
        returns array of shape [n_batch, output_dims, input_dims]
        """
        return jax.vmap(jax.jacfwd(self._map, 0), (0, None, None))(input_array, time, sc)

    @partial(jax.jit, static_argnums=(0, 3))
    def hessian_map(self, input_array, time, sc: OrderedDict):
        """
        returns array of shape [n_batch, output_dims, input_dims, input_dims]
        """
        return jax.vmap(jax.hessian(self._map, 0), (0, None, None))(input_array, time, sc)


class LinearCombination(BaseMap):
    def __init__(self, input_dims, output_dims, scope_var: OrderedDict):
        super(LinearCombination, self).__init__(input_dims, output_dims, scope_var)

        self.key = random.PRNGKey(py_random.randrange(9999))
        self.key, subkey = random.split(self.key)
        scope_var["matrix_a"] = random.normal(subkey, shape=[self.output_dims, self.input_dims])
        self.key, subkey = random.split(self.key)
        scope_var["b"] = random.normal(subkey, shape=[self.output_dims])

    @partial(jax.jit, static_argnums=(0,))
    def _map(self, input_array, time, sc: OrderedDict):
        return np.einsum('ab,b->a', sc["matrix_a"], input_array) + sc["b"]


class LinearCombinationWithTime(BaseMap):
    def __init__(self, input_dims, output_dims, scope_var: OrderedDict):
        super(LinearCombinationWithTime, self).__init__(input_dims, output_dims, scope_var)

        self.key = random.PRNGKey(py_random.randrange(9999))
        self.key, subkey = random.split(self.key)
        scope_var["matrix_a"] = random.normal(subkey, shape=[self.output_dims, self.input_dims+1])
        self.key, subkey = random.split(self.key)
        scope_var["b"] = random.normal(subkey, shape=[self.output_dims])

    @partial(jax.jit, static_argnums=(0,))
    def _map(self, input_array, time, sc: OrderedDict):
        return np.einsum('ab,b->a', sc["matrix_a"], np.append(input_array, time)) + sc["b"]


class LinearCombinationWithSoftplus(BaseMap):
    def __init__(self, input_dims, output_dims, scope_var: OrderedDict):
        super(LinearCombinationWithSoftplus, self).__init__(input_dims, output_dims, scope_var)

        intermediate_dims = 2 * self.output_dims

        self.key = random.PRNGKey(py_random.randrange(9999))

        self.key, subkey = random.split(self.key)
        scope_var["matrix_a"] = random.normal(subkey, shape=[intermediate_dims, self.input_dims])

        self.key, subkey = random.split(self.key)
        scope_var["matrix_b"] = random.normal(subkey, shape=[self.output_dims, intermediate_dims])

        self.key, subkey = random.split(self.key)
        scope_var["b"] = random.normal(subkey, shape=[intermediate_dims])

        self.key, subkey = random.split(self.key)
        scope_var["c"] = random.normal(subkey, shape=[self.output_dims])

        self.layer1 = lambda input_array, sc: np.einsum('ab,b->a', sc["matrix_a"], input_array) + sc["b"]

    @partial(jax.jit, static_argnums=(0,))
    def _map(self, input_array, time, sc: OrderedDict):
        return np.einsum('ab,b->a', sc["matrix_b"], nn.softplus(self.layer1(input_array, sc))) + sc["c"]


class NN(BaseMap):
    def __init__(self, input_dims, output_dims, scope_var: OrderedDict):
        super(NN, self).__init__(input_dims, output_dims, scope_var)

        intermediate_dims = 4 * self.output_dims
        init_random_params, self.predict = stax.serial(
            stax.Flatten,
            stax.Dense(intermediate_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Sigmoid,
            stax.Dense(intermediate_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Sigmoid,
            stax.Dense(output_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
        )

        self.key = random.PRNGKey(py_random.randrange(9999))
        _, init_params = init_random_params(self.key, (1, input_dims))
        scope_var["params"] = init_params

    @partial(jax.jit, static_argnums=(0,))
    def _map(self, input_array, time, sc: OrderedDict):
        values = self.predict(sc["params"], np.expand_dims(input_array, 0))
        return np.squeeze(values)


class NNWithTime(BaseMap):
    def __init__(self, input_dims, output_dims, scope_var: OrderedDict):
        super(NNWithTime, self).__init__(input_dims, output_dims, scope_var)

        intermediate_dims = 4 * self.output_dims
        init_random_params, self.predict = stax.serial(
            stax.Flatten,
            stax.Dense(intermediate_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Sigmoid,
            stax.Dense(intermediate_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Sigmoid,
            stax.Dense(output_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
        )

        self.key = random.PRNGKey(py_random.randrange(9999))
        _, init_params = init_random_params(self.key, (1, input_dims+1))
        scope_var["params"] = init_params

    @partial(jax.jit, static_argnums=(0,))
    def _map(self, input_array, time, sc: OrderedDict):
        values = self.predict(sc["params"], np.expand_dims(np.append(input_array, time), 0))
        return np.squeeze(values)


class NeuralODE(BaseMap):
    def __init__(self, input_dims, output_dims, scope_var: OrderedDict):
        super(NeuralODE, self).__init__(input_dims, output_dims, scope_var)

        intermediate_dims = 2 * self.output_dims * self.input_dims
        init_random_params, self.predict = stax.serial(
            stax.Flatten,
            stax.Dense(intermediate_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Sigmoid,
            stax.Dense(intermediate_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Sigmoid,
            stax.Dense(output_dims * input_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
        )

        self.key = random.PRNGKey(py_random.randrange(9999))
        _, init_params = init_random_params(self.key, (1, input_dims))
        scope_var["params"] = init_params

    @partial(jax.jit, static_argnums=(0,))
    def _map(self, input_array, time, sc: OrderedDict):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def map(self, input_array, time, sc: OrderedDict):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def time_derivative_map(self, input_array, time, sc: OrderedDict):
        return 0.

    @partial(jax.jit, static_argnums=(0,))
    def _first_derivative_map(self, input_array, time, sc: OrderedDict):
        values = self.predict(sc["params"], np.expand_dims(input_array, 0))
        return np.reshape(values, [self.output_dims, self.input_dims])

    @partial(jax.jit, static_argnums=(0,))
    def first_derivative_map(self, input_array, time, sc: OrderedDict):
        return jax.vmap(self._first_derivative_map, (0, None, None))(input_array, time, sc)

    @partial(jax.jit, static_argnums=(0, 2))
    def hessian_map(self, input_array, time, sc: OrderedDict):
        return jax.vmap(jax.jacfwd(self._first_derivative_map, 0), (0, None, None))(input_array, time, sc)


class NeuralODEWithTime(BaseMap):
    def __init__(self, input_dims, output_dims, scope_var: OrderedDict):
        super(NeuralODEWithTime, self).__init__(input_dims, output_dims, scope_var)

        # Space NN
        intermediate_dims = 3 * self.output_dims * self.input_dims
        init_random_params, self.predict = stax.serial(
            stax.Flatten,
            stax.Dense(intermediate_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Sigmoid,
            stax.Dense(intermediate_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Sigmoid,
            stax.Dense(output_dims * input_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
        )

        self.key = random.PRNGKey(py_random.randrange(9999))
        self.key, subkey = random.split(self.key)
        _, init_params = init_random_params(subkey, (1, input_dims+1))
        scope_var["params_space"] = init_params

        # Time NN
        intermediate_dims = 3 * self.output_dims * 1
        init_random_params_time, self.predict_time = stax.serial(
            stax.Flatten,
            stax.Dense(intermediate_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Sigmoid,
            stax.Dense(intermediate_dims,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
            stax.Sigmoid,
            stax.Dense(output_dims * 1,
                       partial(nn.initializers.glorot_normal(), dtype=np.float64),
                       partial(nn.initializers.normal(), dtype=np.float64),
                       ),
        )

        self.key, subkey = random.split(self.key)
        _, init_params_time = init_random_params_time(subkey, (1, input_dims+1))
        scope_var["params_time"] = init_params_time

    @partial(jax.jit, static_argnums=(0,))
    def _map(self, input_array, time, sc: OrderedDict):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def map(self, input_array, time, sc: OrderedDict):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def _time_derivative_map(self, input_array, time, sc: OrderedDict):
        values = self.predict_time(sc["params_time"], np.expand_dims(np.append(input_array, time), 0))
        return np.squeeze(values)

    @partial(jax.jit, static_argnums=(0,))
    def time_derivative_map(self, input_array, time, sc: OrderedDict):
        return jax.vmap(self._time_derivative_map, (0, None, None))(input_array, time, sc)

    @partial(jax.jit, static_argnums=(0,))
    def _first_derivative_map(self, input_array, time, sc: OrderedDict):
        values = self.predict(sc["params_space"], np.expand_dims(np.append(input_array, time), 0))
        return np.reshape(values, [self.output_dims, self.input_dims])

    @partial(jax.jit, static_argnums=(0,))
    def first_derivative_map(self, input_array, time, sc: OrderedDict):
        return jax.vmap(self._first_derivative_map, (0, None, None))(input_array, time, sc)

    @partial(jax.jit, static_argnums=(0, 2))
    def hessian_map(self, input_array, time, sc: OrderedDict):
        return jax.vmap(jax.jacfwd(self._first_derivative_map, 0), (0, None, None))(input_array, time, sc)
