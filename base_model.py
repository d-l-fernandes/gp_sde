import os
import re
from collections import OrderedDict

import numpy as onp
import jax.numpy as np
import jsonpickle
import tensorboardX
from jax.config import config as cfig
from sklearn.decomposition import PCA

cfig.update("jax_enable_x64", True)


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code:  0:sucess -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print(f"Creating directories error: {err}")
        exit(-1)


class Logger:
    def __init__(self, config, predict):
        self.config = config

        self.predict = predict

        if self.predict == 0:
            self.train_summary_writer = tensorboardX.SummaryWriter(os.path.join(self.config["summary_dir"], "train"))
        elif self.predict == 1:
            self.train_summary_writer = tensorboardX.SummaryWriter(os.path.join(self.config["summary_dir"], "train"))
            self.test_summary_writer = tensorboardX.SummaryWriter(os.path.join(self.config["summary_dir"], "test"))
        else:
            self.test_summary_writer = tensorboardX.SummaryWriter(os.path.join(self.config["summary_dir"], "test"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """

        if summarizer == "train":
            summary_writer = self.train_summary_writer
        else:
            summary_writer = self.test_summary_writer

        if summaries_dict is not None:
            for tag, value in summaries_dict.items():
                if len(value.shape) <= 1:
                    if "hist" in tag:
                        summary_writer.add_histogram(tag, value, step)
                    else:
                        summary_writer.add_scalar(tag, value, step)
                else:
                    summary_writer.add_image(tag, value, step)

            summary_writer.flush()

    def close(self):
        """Closes the SummaryWriter"""
        if self.predict == 0:
            self.train_summary_writer.close()
        elif self.predict == 1:
            self.train_summary_writer.close()
            self.test_summary_writer.close()
        else:
            self.test_summary_writer.close()


class BaseModel:
    def __init__(self, main_scope: OrderedDict, config):
        self.config = config
        self.scope = main_scope

        self.scope["cur_epoch"] = 0
        self.metric = -np.inf

    # save function that saves the checkpoint in the path defined in the config file
    # only saves if the current model is better than the best
    def save(self, cur_metric, params):
        if cur_metric > self.metric:
            self.metric = cur_metric
            cur_epoch = self.scope["cur_epoch"]
            self.scope = params.copy()
            self.scope["cur_epoch"] = cur_epoch
            print("Saving model...")
            checkpoints = sorted(os.listdir(self.config["checkpoint_dir"]), key=lambda x: int(re.sub('\D', '', x)))
            if len(checkpoints) == self.config["max_to_keep"]:
                os.remove(os.path.join(self.config["checkpoint_dir"], checkpoints[0]))
            with open(os.path.join(self.config["checkpoint_dir"],
                                   f"checkpoint{self.scope['cur_epoch']+1}.json"), 'w') as f:
                f.write(jsonpickle.encode(self.scope))
            print("Model saved")
        else:
            print("Not saved, as the metric is worse")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self):
        checkpoints = sorted(os.listdir(self.config["checkpoint_dir"]), key=lambda x: int(re.sub('\D', '', x)))
        if checkpoints:
            print(f"Loading model checkpoint {checkpoints[-1]} ...\n")
            with open(os.path.join(self.config["checkpoint_dir"],
                                   checkpoints[-1]), 'r') as f:
                s = jsonpickle.decode(f.read())

            self.scope = s.copy()
            print("Model loaded")

    def increment_cur_epoch_var(self):
        self.scope["cur_epoch"] += 1

    def model_vars(self):
        sc = self.scope.copy()
        del sc["cur_epoch"]
        return sc

    def build(self, sc: OrderedDict):
        raise NotImplementedError


class BaseTrain:
    def __init__(self, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.data = data
        self.max_epoch_diff = 200  # The training will stop if the objective will not improve after this many epochs
        self.reset_epochs = 20  # Trainer will reset after these many epochs, unless it's still improving just before
        # Number of epochs before trainer reset that the model can improve to stop the reset
        self.leeway_epochs = self.config["num_epochs"]
        self.epochs_without_leeway = 20 * self.reset_epochs
        self.cur_epoch_diff = 0
        self.metric = -np.inf  # Current best value of objective

    def train(self):
        for cur_epoch in range(self.model.scope["cur_epoch"], self.config["num_epochs"], 1):

            if cur_epoch % self.reset_epochs == 0:
                if cur_epoch < self.epochs_without_leeway+1:
                    self.reset_trainer()
                else:
                    if self.cur_epoch_diff > self.leeway_epochs:
                        self.reset_trainer()

            objective = self.train_epoch(cur_epoch)

            if objective > self.metric:
                self.metric = objective
                self.cur_epoch_diff = 0
            else:
                self.cur_epoch_diff += 1

            self.model.increment_cur_epoch_var()

            if self.cur_epoch_diff == self.max_epoch_diff:
                print("Training stopped as it was not improving")
                break

    def train_epoch(self, cur_epoch):
        """
        implement the logic of epoch:
        - loop over the number of iterations in the config and call the train step
        - add any summaries you want using the summary

        returns objective
        """
        raise NotImplementedError

    def train_step(self, batch):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metric you need to summarize
        """
        raise NotImplementedError

    def reset_trainer(self):
        """
        resets the trainer so that training goes faster
        """
        raise NotImplementedError

    @staticmethod
    def update_metrics_dict(metrics_epoch, metrics_step):
        for metric in metrics_epoch.keys():
            if metrics_epoch[metric][1]:
                metrics_epoch[metric][0].append(onp.array(metrics_step[metric]))
            else:
                metrics_epoch[metric][0] += onp.array(metrics_step[metric]).tolist()

        return metrics_epoch

    @staticmethod
    def create_summaries_dict(metrics_epoch):
        summaries_dict = {}
        for metric in metrics_epoch.keys():
            if metrics_epoch[metric][1]:
                summaries_dict[f"Metrics/{metric}"] = onp.mean(onp.array(metrics_epoch[metric][0]), axis=0)
            else:
                summaries_dict[f"Metrics/{metric}"] = onp.array(metrics_epoch[metric][0])

        return summaries_dict


class BasePredict:
    def __init__(self, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.data = data

    def predict(self):
        """
        Base class to do predictions
        :return: None, saves predictions to file
        """
        raise NotImplementedError

    @staticmethod
    def average_metrics(metrics):
        averaged_metrics = dict()
        for metric in metrics.keys():
            averaged_metrics[metric] = []
            if metric == "labels":
                averaged_metrics[metric].append(onp.mean(onp.array(metrics[metric][0]), axis=0, dtype=onp.int))
            else:
                averaged_metrics[metric].append(onp.mean(onp.array(metrics[metric][0]), axis=0))
            averaged_metrics[metric].append(metrics[metric][1])
            averaged_metrics[metric].append(metrics[metric][2])

        return averaged_metrics

    @staticmethod
    def update_metrics_dict(metrics_epoch, metrics_step, sample_num):
        for metric in metrics_step.keys():
            if len(metrics_epoch[metric][0]) == sample_num:
                metrics_epoch[metric][0].append(metrics_step[metric])
            else:
                if metrics_epoch[metric][1]:
                    metrics_epoch[metric][0][sample_num] = np.append(metrics_epoch[metric][0][sample_num],
                                                                     metrics_step[metric])
                else:
                    metrics_epoch[metric][0][sample_num] = np.append(metrics_epoch[metric][0][sample_num],
                                                                     metrics_step[metric], axis=0)

        return metrics_epoch

    @staticmethod
    def create_summaries_dict(metrics_epoch):
        summaries_dict = {}
        for metric in metrics_epoch.keys():
            if not metrics_epoch[metric][2]:
                continue
            if metrics_epoch[metric][1]:
                summaries_dict[f"Metrics/{metric}"] = onp.mean(onp.array(metrics_epoch[metric][0]))
            else:
                summaries_dict[f"Metrics/{metric}"] = onp.array(metrics_epoch[metric][0])

        return summaries_dict


class BaseDataGenerator:
    def __init__(self, config):
        self.config = config
        self.total_n_points = 0
        self.num_batches = self.config["num_iter_per_epoch"]
        self.num_batches_test = self.config["num_iter_per_epoch_test"]
        self.b_size = self.config["batch_size"]
        self.n_points = self.config["num_data_points"]
        self.n_points_test = self.config["num_data_points_test"]

        self.mean = None
        self.stddev = None

        self.min = None
        self.max = None

    def standardize(self, data, time_series=False):
        if time_series:
            self.mean = np.mean(data, axis=1)
            self.stddev = np.std(data, axis=1)
            norm_data = (data - np.expand_dims(self.mean, axis=1)) / np.expand_dims(self.stddev, axis=1)
        else:
            self.mean = np.mean(data, axis=0)
            self.stddev = np.std(data, axis=0)
            norm_data = (data - self.mean) / self.stddev

        return norm_data

    def reverse_standardize(self, data):
        return data * self.stddev + self.mean

    def minmax(self, data, max_val=None, min_val=None):

        if min_val is None:
            self.min = np.min(data, axis=0)
        else:
            self.min = min_val

        if max_val is None:
            self.max = np.max(data, axis=0)
        else:
            self.max = max_val

        minmax_data = (data - self.min) / (self.max - self.min)

        return minmax_data

    def reverse_minmax(self, data):
        return data * (self.max - self.min) + self.min

    @staticmethod
    def pca_transform(data, pca_dims, pca_fit=None):
        data_flat = np.reshape(data, [data.shape[0], np.prod(data.shape[1:])])
        if pca_fit is None:
            pca_fit = PCA(n_components=pca_dims)
            pca_fit.fit(data_flat)

        return pca_fit.transform(data_flat)

    @staticmethod
    def binarize(x: np.ndarray, boundary: float = 0.5, max_value: float = 255.0) -> np.ndarray:
        """Binarizes x' = x / max_value
        1 if x' > boundary
        0 if x' <= boundary
        :rtype: np.ndarray"""

        x = x / max_value

        return np.where(x > boundary, 1, 0)

    def select_batch_generator(self, phase):
        pass

    def plot_data_point(self, data, axis):
        pass
