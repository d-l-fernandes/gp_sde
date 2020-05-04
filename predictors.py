from functools import partial

import jax
import jax.numpy as np
from jax import lax
from jax import ops
import numpy as onp
from tqdm import tqdm

import base_model as bm
import aux_plot


class GPSDEPredictor(bm.BasePredict):
    def __init__(self, model, data, config, logger):
        super(GPSDEPredictor, self).__init__(model, data, config, logger)

    def predict(self):
        print("Getting predictions...")
        self._predict_from_y()

    def _predict_from_y(self):

        self.get_label = self.config["label_latent_manifold"]
        # name of metric: [variable, if mean is to be used, if it goes in summary]
        self.metrics = {
            "elbo": [[], True, True],
            "reco": [[], True, True],
            "kl_global": [[], True, True],
            "hyperprior": [[], True, True],
            "paths_y": [[], False, False],
            "y_grid": [[], False, False],
            "ard_weights": [[], False, False],
            "drift_grid": [[], False, False],
            "diffusion_grid": [[], False, False],
        }

        if self.get_label:
            self.metrics["labels"] = [[], False, False]

        print("    Getting metrics...")

        # Predictions Loop
        loop = tqdm(range(self.config["prediction_samples"]), desc=f"Y Testing Epoch", ascii=True)
        for i in loop:
            self.dataset = self.data.select_phase("testing_y")
            for _ in range(self.config["num_iter_per_epoch_test"]):
                metrics_step = self.train_step(next(self.dataset))
                self.metrics = self.update_metrics_dict(self.metrics, metrics_step, i)
            if np.prod(self.config["state_size"]) == 2:
                y_grid, drift_grid, diffusion_grid = self._get_2d_latent_grid(self.metrics["paths_y"][0][i])
                metrics_step = dict()
                metrics_step["y_grid"] = y_grid
                metrics_step["drift_grid"] = drift_grid
                metrics_step["diffusion_grid"] = diffusion_grid
                self.metrics = self.update_metrics_dict(self.metrics, metrics_step, i)
            if np.prod(self.config["state_size"]) == 1:
                y_grid, drift_grid, diffusion_grid = self._get_1d_latent_grid(self.metrics["paths_y"][0][i])
                metrics_step = dict()
                metrics_step["y_grid"] = y_grid
                metrics_step["drift_grid"] = drift_grid
                metrics_step["diffusion_grid"] = diffusion_grid
                self.metrics = self.update_metrics_dict(self.metrics, metrics_step, i)

        average_metrics = self.average_metrics(self.metrics)

        # Label stuff
        if self.get_label:
            labels = average_metrics["labels"][0]
        else:
            labels = None

        # ARD Weights
        aux_plot.plot_ard_weights(self.config, average_metrics["ard_weights"][0][:onp.prod(self.config["state_size"])])

        # Time-series data
        if self.config["multiple_1d_data"]:
            aux_plot.plot_1d_timeseries_data(self.config,
                                             onp.transpose(onp.array(average_metrics["paths_y"][0]), (1, 0, 2)),
                                             self.data.input_test_np,
                                             onp.prod(self.config["state_size"]),
                                             self.config["num_steps"], self.config["num_steps_test"])

        if onp.prod(self.config["state_size"]) == 1:
            xt_grid, drift, diffusion = self._get_1d_latent_grid(onp.array(average_metrics["paths_y"][0]))
            aux_plot.plot_1d_paths(
                self.config,
                onp.array(average_metrics["paths_y"][0]),
                xt_grid,
                drift,
                diffusion,
                labels=labels
            )

        if onp.prod(self.config["state_size"]) == 2:
            aux_plot.plot_2d_paths(self.config, self.data.input_test_np,
                                   onp.array(average_metrics["paths_y"][0]), labels)

        summaries_dict = self.create_summaries_dict(average_metrics)

        self.logger.summarize(1, summaries_dict=summaries_dict, summarizer="test")

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, batch):
        batch_label = None
        if self.get_label:
            batch_y, batch_t, batch_x, batch_label = batch
        else:
            batch_y, batch_t, batch_x = batch

        elbo, metrics = self.model.loss(batch_y, batch_t, self.model.model_vars(), self.config["num_steps_test"])

        metrics["paths_y"] = np.transpose(metrics["paths_y"], (1, 0, 2))

        if self.get_label:
            metrics["labels"] = batch_label

        return metrics

    @partial(jax.jit, static_argnums=(0,))
    def _get_2d_latent_grid(self, paths_x):

        num_points_grid = 30

        def scan_fn(carry, paths):
            x, drift, diffusion, index = carry
            time = index * self.config["delta_t"] - 0.5
            max_x = np.amax(paths_x[:, :, 0])
            min_x = np.amin(paths_x[:, :, 0])
            max_y = np.amax(paths_x[:, :, 1])
            min_y = np.amin(paths_x[:, :, 1])
            xx, yy = np.meshgrid(np.linspace(min_x, max_x, num_points_grid), np.linspace(min_y, max_y, num_points_grid))
            temp = np.transpose(np.vstack([xx.reshape(-1), yy.reshape(-1)]))

            gp_matrices, temp_drift_function, temp_diffusion_function = self.model.build(self.model.model_vars())
            temp_drift = temp_drift_function(temp, time)
            temp_diffusion = np.linalg.det(temp_diffusion_function(temp, time))

            x = ops.index_add(x, ops.index[index], temp)
            drift = ops.index_add(drift, ops.index[index], temp_drift)
            diffusion = ops.index_add(diffusion, ops.index[index], temp_diffusion)
            index += 1

            return (x, drift, diffusion, index), np.array([0.])

        x_grid = np.zeros((paths_x.shape[1], num_points_grid**2, 2))
        drift_grid = np.zeros((paths_x.shape[1], num_points_grid**2, 2))
        diffusion_grid = np.zeros((paths_x.shape[1], num_points_grid**2))
        (x_grid, drift_grid, diffusion_grid, index), _ = lax.scan(scan_fn,
                                                                  (x_grid, drift_grid, diffusion_grid, 0),
                                                                  np.transpose(paths_x, (1, 0, 2)))

        return x_grid, drift_grid, diffusion_grid

    @partial(jax.jit, static_argnums=(0,))
    def _get_1d_latent_grid(self, paths_x):

        num_points_grid = 20

        max_x = np.amax(paths_x)
        min_x = np.amin(paths_x)
        x_array = np.linspace(min_x, max_x, 20)[:, None]
        xx, tt = np.meshgrid(np.linspace(min_x, max_x, 20), np.linspace(0, 1, paths_x.shape[1]))
        txpairs = np.transpose(np.vstack([tt.reshape(-1), xx.reshape(-1)]))

        def scan_fn(carry, paths):
            drift, diffusion, index = carry
            time = index * self.config["delta_t"] - 0.5

            gp_matrices, temp_drift_function, temp_diffusion_function = self.model.build(self.model.model_vars())
            temp_drift = temp_drift_function(x_array, time)
            temp_diffusion = np.linalg.det(temp_diffusion_function(x_array, time))

            drift = ops.index_add(drift, ops.index[index], temp_drift)
            diffusion = ops.index_add(diffusion, ops.index[index], temp_diffusion)
            index += 1

            return (drift, diffusion, index), np.array([0.])

        drift_grid = np.zeros((paths_x.shape[1], num_points_grid, 1))
        diffusion_grid = np.zeros((paths_x.shape[1], num_points_grid))
        (drift_grid, diffusion_grid, index), _ = lax.scan(scan_fn,
                                                          (drift_grid, diffusion_grid, 0),
                                                          np.transpose(paths_x, (1, 0, 2)))
        return txpairs, drift_grid.reshape(-1), diffusion_grid.reshape(-1)
