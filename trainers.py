from functools import partial

import jax
from jax.config import config as cfig
from jax.experimental import optimizers
from tqdm import tqdm

import base_model as bm

cfig.update("jax_enable_x64", True)


class GPSDETrainer(bm.BaseTrain):
    def __init__(self, model, data, config, logger):
        super(GPSDETrainer, self).__init__(model, data, config, logger)
        self.dataset = self.data.select_phase("training")
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(self.config["learning_rate_global"])
        self.opt_state = self.opt_init(self.model.model_vars())

    def train_epoch(self, cur_epoch):
        # name of metric: [variable, if mean is to be used]
        metrics = {
            "elbo": [[], True],
            "reco": [[], True],
            "kl_global": [[], True],
            "hyperprior": [[], True],
        }

        loop = tqdm(range(self.config["num_iter_per_epoch"]), desc=f"Epoch {cur_epoch+1}/{self.config['num_epochs']}",
                    ascii=True)

        for i in loop:
            metrics_step, self.opt_state = self.train_step(i, self.opt_state, next(self.dataset))
            metrics = self.update_metrics_dict(metrics, metrics_step)

        summaries_dict = self.create_summaries_dict(metrics)

        self.logger.summarize(cur_epoch+1, summaries_dict=summaries_dict)
        self.model.save(summaries_dict["Metrics/elbo"], self.get_params(self.opt_state))

        return summaries_dict["Metrics/elbo"]

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, i, opt_state, batch):
        batch_y = batch[0]
        batch_t = batch[1]
        # batch_x = batch[2]

        params = self.get_params(opt_state)
        (elbo, metrics), g = self.model.value_and_grad(batch_y, batch_t, params, self.config["num_steps"])
        opt_state = self.opt_update(i, g, opt_state)

        return metrics, opt_state

    def reset_trainer(self):
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(self.config["learning_rate_global"])
        self.opt_state = self.opt_init(self.model.model_vars())
