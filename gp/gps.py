import sys
sys.path.append("..")

from functools import partial
import jax
import jax.numpy as np
import jax.nn as nn
import jax.scipy as scipy
import jax.random as random
from jax_aux import aux_math
import numpyro
from numpyro.distributions.continuous import MultivariateNormal
from jax.config import config

import random as py_random
from collections import OrderedDict

config.update("jax_enable_x64", True)


class SolveGP:
    def __init__(self,
                 scope_var: OrderedDict,
                 ndims_in,
                 ndims_out,
                 n_inducing_beta=30,
                 n_inducing_gamma=80,
                 ):

        # Dimensions
        self.ndims_in = ndims_in
        self.ndims_out = ndims_out
        self.n_inducing_beta = n_inducing_beta
        self.n_inducing_gamma = n_inducing_gamma

        self.jitter = 1.e-4
        self.scope = scope_var
        self.key = random.PRNGKey(py_random.randrange(9999))

        # Global variational parameters
        # Beta
        self.key, subkey = random.split(self.key)
        self.scope["mu_u_beta"] = np.zeros(shape=(self.ndims_out, self.n_inducing_beta))
        # self.scope["mu_u_beta"] = random.normal(subkey, shape=(self.ndims_out, self.n_inducing_beta))

        random_shape = [self.ndims_out, self.n_inducing_beta, self.n_inducing_beta]
        self.key, subkey = random.split(self.key)
        self.scope["S_u_beta"] = np.zeros(shape=random_shape)
        # self.scope["S_u_beta"] = random.normal(subkey, shape=random_shape)

        # Gamma
        self.key, subkey = random.split(self.key)
        self.scope["mu_u_gamma"] = np.zeros(shape=[self.ndims_out, self.n_inducing_gamma])
        # self.scope["mu_u_gamma"] = random.normal(subkey, shape=(self.ndims_out, self.n_inducing_gamma))

        random_shape = [self.ndims_out, self.n_inducing_gamma, self.n_inducing_gamma]
        self.key, subkey = random.split(self.key)
        self.scope["S_u_gamma"] = np.zeros(shape=random_shape)
        # self.scope["S_u_gamma"] = random.normal(subkey, shape=random_shape)

        # Inducing points
        self.key, subkey = random.split(self.key)
        # self.scope["X_u_beta"] = random.normal(subkey, shape=[self.n_inducing_beta, self.ndims_in])
        self.scope["X_u_beta"] = np.zeros(shape=[self.n_inducing_beta, self.ndims_in])

        self.key, subkey = random.split(self.key)
        # self.scope["X_u_gamma"] = random.normal(subkey, shape=[self.n_inducing_gamma, self.ndims_in])
        self.scope["X_u_gamma"] = np.zeros(shape=[self.n_inducing_gamma, self.ndims_in])

        self.kernel = None

    @partial(jax.jit, static_argnums=(0, ))
    def build(self, sc: OrderedDict):
        gp_matrices = OrderedDict()
        gp_matrices["l_u_beta"] = aux_math.matrix_diag_transform(np.tril(sc["S_u_beta"]), nn.softplus)
        gp_matrices["l_u_gamma"] = aux_math.matrix_diag_transform(np.tril(sc["S_u_gamma"]), nn.softplus)

        # Kernel
        gp_matrices["k_beta_beta"] = self.kernel.matrix(sc["X_u_beta"], sc["X_u_beta"])
        gp_matrices["l_beta_beta"] = scipy.linalg.cholesky(gp_matrices["k_beta_beta"] +
                                                           self.jitter * np.eye(self.n_inducing_beta),
                                                           lower=True)

        k_gamma_gamma = \
            self.kernel.matrix(sc["X_u_gamma"], sc["X_u_gamma"]) + self.jitter * np.eye(self.n_inducing_gamma)

        k_beta_gamma = self.kernel.matrix(sc["X_u_beta"], sc["X_u_gamma"])

        l_beta_inv_k_beta_gamma = scipy.linalg.solve_triangular(gp_matrices["l_beta_beta"], k_beta_gamma,
                                                                lower=True)
        gp_matrices["l_beta_inv_k_beta_gamma"] = l_beta_inv_k_beta_gamma

        c_gamma_gamma = k_gamma_gamma - np.matmul(
            np.transpose(gp_matrices["l_beta_inv_k_beta_gamma"], (0, 2, 1)), gp_matrices["l_beta_inv_k_beta_gamma"])

        gp_matrices["l_gamma_gamma"] = scipy.linalg.cholesky(c_gamma_gamma +
                                                             self.jitter * np.eye(self.n_inducing_gamma), lower=True)

        # U_beta_dists
        gp_matrices["q_u_beta_mean"] = sc["mu_u_beta"]
        gp_matrices["q_u_beta_tril"] = gp_matrices["l_u_beta"]
        gp_matrices["p_u_beta_mean"] = np.zeros([self.ndims_out, self.n_inducing_beta])
        gp_matrices["p_u_beta_tril"] = gp_matrices["l_beta_beta"]

        # U_gamma_dists
        gp_matrices["q_u_gamma_mean"] = sc["mu_u_gamma"]
        gp_matrices["q_u_gamma_tril"] = gp_matrices["l_u_gamma"]
        gp_matrices["p_u_gamma_mean"] = np.zeros([self.ndims_out, self.n_inducing_gamma])
        gp_matrices["p_u_gamma_tril"] = gp_matrices["l_gamma_gamma"]

        return gp_matrices

    @staticmethod
    def regularization(matrices_dict):
        reg = \
            np.mean(aux_math.kl_divergence_multivariate_normal(
                matrices_dict["q_u_beta_mean"],
                matrices_dict["q_u_beta_tril"],
                matrices_dict["p_u_beta_mean"],
                matrices_dict["p_u_beta_tril"],
                lower=True
            )) \
            + np.mean(aux_math.kl_divergence_multivariate_normal(
                matrices_dict["q_u_gamma_mean"],
                matrices_dict["q_u_gamma_tril"],
                matrices_dict["p_u_gamma_mean"],
                matrices_dict["p_u_gamma_tril"],
                lower=True
            ))
        return reg

    @property
    def hyperprior(self):
        return 0.

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x_input, sc: OrderedDict, matrices_dist: OrderedDict):
        # Predict Kernel
        k_beta_x = self.kernel.matrix(sc["X_u_beta"], x_input)
        k_gamma_x = self.kernel.matrix(sc["X_u_gamma"], x_input)
        k_x_x = self.kernel.matrix(x_input, x_input)

        l_beta_inv_k_beta_x = scipy.linalg.solve_triangular(matrices_dist["l_beta_beta"], k_beta_x, lower=True)
        a_beta = scipy.linalg.solve_triangular(matrices_dist["l_beta_beta"], l_beta_inv_k_beta_x, lower=True)

        c_gamma_x = k_gamma_x - np.matmul(np.transpose(matrices_dist["l_beta_inv_k_beta_gamma"], (0, 2, 1)),
                                          l_beta_inv_k_beta_x)

        c_x_x = k_x_x - np.matmul(np.transpose(l_beta_inv_k_beta_x, (0, 2, 1)), l_beta_inv_k_beta_x)

        l_gamma_inv_c_gamma_x = scipy.linalg.solve_triangular(matrices_dist["l_gamma_gamma"],
                                                              c_gamma_x, lower=True)

        a_gamma = scipy.linalg.solve_triangular(matrices_dist["l_gamma_gamma"], l_gamma_inv_c_gamma_x, lower=True)

        mean_in = np.einsum("abc,ab->ac", a_beta, matrices_dist["q_u_beta_mean"])
        mean_perp = np.einsum("abc,ab->ac", a_gamma, matrices_dist["q_u_gamma_mean"])

        l_q_beta_a_beta = np.einsum("abc,acd->abd", np.transpose(matrices_dist["q_u_beta_tril"], (0, 2, 1)),
                                    a_beta)
        var_in = np.einsum("abc,abd->acd", l_q_beta_a_beta, l_q_beta_a_beta)
        l_q_gamma_a_gamma = np.einsum("abc,acd->abd", np.transpose(matrices_dist["q_u_gamma_tril"], (0, 2, 1)),
                                      a_gamma)
        var_perp = \
            np.einsum("abc,abd->acd", l_q_gamma_a_gamma, l_q_gamma_a_gamma) + \
            c_x_x - \
            np.einsum("abc,abd->acd", l_gamma_inv_c_gamma_x, l_gamma_inv_c_gamma_x)

        self.key, subkey = random.split(self.key)
        likelihood = numpyro.sample('likelihood_gp', MultivariateNormal(
            mean_in + mean_perp,
            scale_tril=scipy.linalg.cholesky(var_in + var_perp + self.jitter * np.eye(var_perp.shape[-1]))
        ), rng_key=subkey)

        return likelihood
