from collections import OrderedDict
from functools import partial
import random as py_random

import jax
import jax.nn as nn
import jax.numpy as np
import jax.ops as ops
import jax.random as random
from jax import lax
from jax.config import config as cfig

import numpyro
from numpyro.distributions.continuous import InverseGamma

import aux
import base_model as bm
from gp import gps
from jax_aux import aux_math
from sde import sde, solvers

cfig.update("jax_enable_x64", True)


class GPSDE(bm.BaseModel):

    def __init__(self, main_scope, config):
        super(GPSDE, self).__init__(main_scope, config)

        # Dimensions
        self.t_n_batch = self.config["batch_size"]
        t_original_q = np.prod(self.config["state_size"])
        m_beta = self.config["num_ind_points_beta"]
        m_gamma = self.config["num_ind_points_gamma"]

        self.weight = self.t_n_batch / self.config["num_data_points"]

        # Random key
        self.key = random.PRNGKey(py_random.randrange(9999))

        # Solver
        if self.config["solver"] == "StrongOrder3HalfsSolver":
            self.solver = solvers.StrongOrder3HalfsSolver
        else:
            self.solver = solvers.EulerMaruyamaSolver

        # Gamma Params
        if self.config["constant_diffusion"]:
            gamma_scope = OrderedDict()
            self.scope["gamma_params"] = gamma_scope
            self.gamma_params = aux.GammaParams(t_original_q, gamma_scope)
        else:
            # Kernel - doesn't need to be built, GP does it
            kernel_diffusion_scope = OrderedDict()
            self.scope["kernel_diffusion"] = kernel_diffusion_scope
            if self.config["time_dependent_gp"]:
                self.sde_kernel_diffusion = aux.ExponentiatedQuadraticKernel(t_original_q+1, t_original_q,
                                                                             kernel_diffusion_scope,
                                                                             time=self.config["delta_t"])
            else:
                self.sde_kernel_diffusion = aux.ExponentiatedQuadraticKernel(t_original_q, t_original_q,
                                                                             kernel_diffusion_scope)
            gp_diffusion_scope = OrderedDict()
            self.scope["sde_gp_diffusion"] = gp_diffusion_scope
            if self.config["time_dependent_gp"]:
                self.sde_gp_diffusion = gps.SolveGP(gp_diffusion_scope, t_original_q+1, t_original_q,
                                                    m_beta, m_gamma)
            else:
                self.sde_gp_diffusion = gps.SolveGP(gp_diffusion_scope, t_original_q, t_original_q,
                                                    m_beta, m_gamma)

        # Kernel - doesn't need to be built, GP does it
        kernel_scope = OrderedDict()
        self.scope["kernel"] = kernel_scope
        if self.config["time_dependent_gp"]:
            self.sde_kernel = aux.ExponentiatedQuadraticKernel(t_original_q+1, t_original_q, kernel_scope,
                                                               time=self.config["delta_t"])
        else:
            self.sde_kernel = aux.ExponentiatedQuadraticKernel(t_original_q, t_original_q, kernel_scope)

        # SDE GP - Needs to be built
        gp_scope = OrderedDict()
        self.scope["sde_gp"] = gp_scope
        if self.config["time_dependent_gp"]:
            self.sde_gp = gps.SolveGP(gp_scope, t_original_q+1, t_original_q,
                                      m_beta, m_gamma)
        else:
            self.sde_gp = gps.SolveGP(gp_scope, t_original_q, t_original_q,
                                      m_beta, m_gamma)

        # SDE
        self.sde_var = sde.SDE(self.config["delta_t"], self.solver, t_original_q)

        # Likelihood vars - doesn't need to be built
        likelihood_scope = OrderedDict()
        self.scope["likelihood"] = likelihood_scope
        self.signal_variance = aux.LikelihoodVariance(t_original_q, likelihood_scope)

        self.y_t = None
        self.paths_y = None

    def latent_drift_function(self, x, t, param_dict, gp_matrices):
        if self.config["time_dependent_gp"]:
            time = np.ones(shape=(x.shape[0], 1)) * t
            y = np.concatenate((x, time), axis=1)
            return np.transpose(self.sde_gp(y, param_dict["sde_gp"], gp_matrices))
        return np.transpose(self.sde_gp(x, param_dict["sde_gp"], gp_matrices))

    def latent_diffusion_function(self, x, t, param_dict, gp_matrices):
        if self.config["constant_diffusion"]:
            concentration, rate = self.gamma_params.build(param_dict["gamma_params"])
            self.key, subkey = random.split(self.key)
            inverse_lambdas = numpyro.sample("inverse_lambdas",
                                             InverseGamma(nn.softplus(concentration),
                                                          nn.softplus(rate)),
                                             rng_key=subkey)
            return np.tile(np.expand_dims(aux_math.diag(inverse_lambdas), axis=0),
                           [x.shape[0], 1, 1])
        else:
            if self.config["time_dependent_gp"]:
                time = np.ones(shape=(x.shape[0], 1)) * t
                y = np.concatenate((x, time), axis=1)
                return aux_math.diag(np.transpose(self.sde_gp_diffusion(y, param_dict["sde_gp"], gp_matrices)))
            return aux_math.diag(np.transpose(self.sde_gp_diffusion(x, param_dict["sde_gp"], gp_matrices)))

    def build(self, sc: OrderedDict):
        self.sde_gp.kernel = self.sde_kernel.build(sc["kernel"])
        gp_matrices_drift = self.sde_gp.build(sc["sde_gp"])

        if not self.config["constant_diffusion"]:
            self.sde_gp_diffusion.kernel = self.sde_kernel_diffusion.build(sc["kernel_diffusion"])
            gp_matrices_diffusion = self.sde_gp_diffusion.build(sc["sde_gp_diffusion"])
        else:
            gp_matrices_diffusion = dict()

        gp_matrices = dict()
        gp_matrices["drift"] = gp_matrices_drift
        gp_matrices["diffusion"] = gp_matrices_diffusion

        return gp_matrices, \
               partial(self.latent_drift_function, param_dict=sc, gp_matrices=gp_matrices["drift"]), \
               partial(self.latent_diffusion_function, param_dict=sc, gp_matrices=gp_matrices["diffusion"])

    def get_metrics(self, metrics_dict, gp_matrices, param_dict):
        metrics_dict["kl_global"] = self.weight * self.sde_gp.regularization(gp_matrices["drift"])
        if not self.config["constant_diffusion"]:
            metrics_dict["kl_global"] += self.weight * self.sde_gp_diffusion.regularization(gp_matrices["diffusion"])
        # metrics_dict["hyperprior"] = self.weight * (self.sde_kernel(param_dict["kernel"]) + self.sde_gp.hyperprior +
        #                                             self.signal_variance(param_dict["likelihood"]) +
        #                                             self.gamma_params(param_dict["gamma_params"])
        #                                             )
        metrics_dict["hyperprior"] = 0.
        metrics_dict["elbo"] = metrics_dict["reco"] - metrics_dict["kl_global"] - metrics_dict["hyperprior"]
        metrics_dict["paths_y"] = self.paths_y

        if self.config["time_dependent_gp"]:
            metrics_dict["ard_weights"] = np.mean(
                np.exp(param_dict["kernel"]["log_kernel_weights_latent"]), axis=0)[:-1]
        else:
            metrics_dict["ard_weights"] = np.mean(np.exp(param_dict["kernel"]["log_kernel_weights_latent"]), axis=0)

    @partial(jax.jit, static_argnums=(0, 2, 4))
    def loss(self, y_input, t_indices: list, param_dict, num_steps):

        if type(t_indices) is not list:
            raise TypeError("Time indices object must be a list")
        # if self.config["mapping"] == "neural_ode_with_softplus" and 0 not in t_indices:
        #     print(f"For mapping {self.config['mapping']}, the initial point (index 0) is required.")

        y_0 = y_input[0].reshape(y_input[0].shape[0], -1)

        gp_matrices, latent_drift_function, latent_diffusion_function = self.build(param_dict)

        self.sde_var.drift_function = latent_drift_function
        self.sde_var.diffusion_function = latent_diffusion_function

        self.y_t, self.paths_y = self.sde_var(y_0, num_steps)

        y_t_to_compare = self.paths_y[ops.index[t_indices]]
        metrics = dict()
        metrics["reco"] = \
            np.mean(
                aux_math.log_prob_multivariate_normal(
                    y_t_to_compare,
                    aux_math.diag(np.sqrt(nn.softplus(self.signal_variance.build(param_dict["likelihood"])))),
                    y_input[t_indices]))

        self.get_metrics(metrics, gp_matrices, param_dict)
        return -metrics["elbo"], metrics

    @partial(jax.jit, static_argnums=(0, 2, 4))
    def grad(self, y_input, t_indices: list, param_dict, num_steps):
        return jax.grad(self.loss, argnums=2, has_aux=True)(y_input, t_indices, param_dict, num_steps)

    @partial(jax.jit, static_argnums=(0, 2, 4))
    def value_and_grad(self, y_input, t_indices: list, param_dict, num_steps):
        return jax.value_and_grad(self.loss, argnums=2, has_aux=True)(y_input, t_indices, param_dict, num_steps)
