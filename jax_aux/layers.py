from functools import partial

import jax.numpy as np
from jax import lax
from jax import random
from jax.nn import (log_softmax, sigmoid)
from jax.nn.initializers import glorot_normal, normal

# aliases for backwards compatibility
glorot = glorot_normal
randn = normal
logsoftmax = log_softmax


def GRU(out_dim, W_init=glorot(), b_init=normal()):
    """ Layer construction function for Gated Recurrent Unit (GRU) layer. """
    def init_fun(rng, input_shape):
        """ Initialize the GRU layer for stax. """
        hidden = b_init(rng, (input_shape[0], out_dim))

        k1, k2, k3 = random.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        k1, k2, k3 = random.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        k1, k2, k3 = random.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )
        # Input dim 0 represents the batch dimension
        # Input dim 1 represents the time dimension (before scan moveaxis)
        output_shape = (input_shape[0], input_shape[1], out_dim)
        return (output_shape,
                (hidden,
                 (update_W, update_U, update_b),
                 (reset_W, reset_U, reset_b),
                 (out_W, out_U, out_b),),)

    def apply_fun_scan(params, hidden, inp):
        """ Perform single timestep update of the network. """
        _, (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b) = params

        update_gate = sigmoid(np.dot(inp, update_W) + np.dot(hidden, update_U)
                              + update_b)
        reset_gate = sigmoid(np.dot(inp, reset_W) + np.dot(hidden, reset_U)
                             + reset_b)
        output_gate = np.tanh(np.dot(inp, out_W)
                              + np.dot(np.multiply(reset_gate, hidden), out_U)
                              + out_b)
        output = np.multiply(update_gate, hidden) + np.multiply(1-update_gate,
                                                                output_gate)
        hidden = output
        return hidden, hidden

    def apply_fun(params, inputs, **kwargs):
        """ Loop over the time steps of the input sequence. """
        h = params[0]
        # Move the time dimension to position 0
        inputs = np.moveaxis(inputs, 1, 0)
        f = partial(apply_fun_scan, params)
        # Use lax.scan for fast compilation
        _, h_new = lax.scan(f, h, inputs)
        return h_new

    return init_fun, apply_fun


def LSTM(out_dim, W_init=glorot(), b_init=normal()):
    """ Layer construction function for LSTM layer. """
    def init_fun(rng, input_shape):
        k1, k2 = random.split(rng)
        cell, hidden = (b_init(k1, (input_shape[0], out_dim)),
                        b_init(k2, (input_shape[0], out_dim)))

        k1, k2, k3 = random.split(k1, num=3)
        forget_W, forget_U, forget_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        k1, k2, k3 = random.split(k1, num=3)
        in_W, in_U, in_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )


        k1, k2, k3 = random.split(k1, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        k1, k2, k3 = random.split(k1, num=3)
        change_W, change_U, change_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),
        )

        output_shape = (input_shape[0], input_shape[0], out_dim)
        return (output_shape,
                ((cell, hidden),
                 (forget_W, forget_U, forget_b),
                 (in_W, in_U, in_b),
                 (out_W, out_U, out_b),
                 (change_W, change_U, change_b),),)

    def apply_fun_scan(params, hidden_cell, inp):
        """ Perform single timestep update of the network. """
        _, (forget_W, forget_U, forget_b), (in_W, in_U, in_b), (
            out_W, out_U, out_b), (change_W, change_U, change_b) = params

        hidden, cell = hidden_cell
        input_gate = sigmoid(np.dot(inp, in_W) + np.dot(hidden, in_U) + in_b)
        change_gate = np.tanh(np.dot(inp, change_W) + np.dot(hidden, change_U)
                              + change_b)
        forget_gate = sigmoid(np.dot(inp, forget_W) + np.dot(hidden, forget_U)
                              + forget_b)

        cell = np.multiply(change_gate, input_gate) + np.multiply(cell,
                                                                  forget_gate)

        output_gate = sigmoid(np.dot(inp, out_W)
                              + np.dot(hidden, out_U) + out_b)
        output = np.multiply(output_gate, np.tanh(cell))
        hidden_cell = (hidden, cell)
        return hidden_cell, hidden_cell

    def apply_fun(params, inputs, **kwargs):
        """ Loop over the time steps of the input sequence. """
        h = params[0]
        # Move the time dimension to position 0
        inputs = np.moveaxis(inputs, 1, 0)
        f = partial(apply_fun_scan, params)
        # Use lax.scan for fast compilation
        _, h_new = lax.scan(f, h, inputs)
        return h_new[0]

    return init_fun, apply_fun