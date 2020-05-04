from jax.nn import softplus
import numpy as np
from scipy.io import arff
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import base_model as bm


class DataGeneratorToyDataset1DChirp(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorToyDataset1DChirp, self).__init__(config)

        t_steps = self.config["num_steps"]

        t = np.linspace(-1, 1, t_steps + 1)
        # frequency = -18 * t**2 + 20
        frequency = 6 * t + 8
        noise_train = np.random.normal(scale=0.05, size=(t_steps + 1, self.config["num_data_points"]))
        noise_amplitude_train = np.random.normal(scale=0.5, size=(self.config["num_data_points"]))
        noise_test = np.zeros(shape=(t_steps + 1, self.config["num_data_points_test"]))
        y_train = np.cos(frequency * t)[:, None] + noise_train + noise_amplitude_train[None]

        y_test = np.sin(frequency * t)[:, None] + noise_test
        y_test[:, :self.config["num_data_points_test"] // 8] += 2
        y_test[:, self.config["num_data_points_test"] // 8: 2 * self.config["num_data_points_test"] // 8] -= 2
        y_test[:, 2 * self.config["num_data_points_test"] // 8: 3 * self.config["num_data_points_test"] // 8] += 1
        y_test[:, 3 * self.config["num_data_points_test"] // 8: 4 * self.config["num_data_points_test"] // 8] -= 1
        y_test[:, 4 * self.config["num_data_points_test"] // 8: 5 * self.config["num_data_points_test"] // 8] -= 0
        y_test[:, 5 * self.config["num_data_points_test"] // 8: 6 * self.config["num_data_points_test"] // 8] += 4
        y_test[:, 6 * self.config["num_data_points_test"] // 8: 7 * self.config["num_data_points_test"] // 8] -= 4
        y_test[:, 7 * self.config["num_data_points_test"] // 8: 8 * self.config["num_data_points_test"] // 8] += 6

        labels = np.concatenate(
            (
                0 * np.ones(self.config["num_data_points_test"] // 8),
                1 * np.ones(self.config["num_data_points_test"] // 8),
                2 * np.ones(self.config["num_data_points_test"] // 8),
                3 * np.ones(self.config["num_data_points_test"] // 8),
                4 * np.ones(self.config["num_data_points_test"] // 8),
                5 * np.ones(self.config["num_data_points_test"] // 8),
                6 * np.ones(self.config["num_data_points_test"] // 8),
                7 * np.ones(self.config["num_data_points_test"] // 8),
            )
        )

        self.input_train_np = np.concatenate((np.tile(t[:, None, None], [1, self.config["num_data_points"], 1]),
                                              y_train[:, :, None]), axis=2)
        self.input_test_np = np.concatenate((np.tile(t[:, None, None], [1, self.config["num_data_points_test"], 1]),
                                             y_test[:, :, None]), axis=2)
        self.input_train_pca_np = self.input_train_np
        self.input_test_pca_np = self.input_test_np
        self.input_test_np_labels = labels
        self.input_t = list(range(self.config["num_steps"]+1))

    def select_phase(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train_np[:, idx], self.input_t, self.input_train_pca_np[:, idx]
        elif phase == "testing_y":
            for i in range(self.num_batches_test):
                yield self.input_test_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_t, \
                      self.input_test_pca_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_test_np_labels[i * self.b_size:(i+1) * self.b_size]
        else:
            raise ValueError("Invalid phase")

    def plot_data_point(self, data, axis):
        return None


class DataGeneratorToyDatasetSineDraw(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorToyDatasetSineDraw, self).__init__(config)

        t_steps = self.config["num_steps"]
        frequency = 5

        x = np.linspace(-1, 1, t_steps + 1)
        # noise_x_train = np.random.normal(scale=0.1, size=(self.config["num_data_points"]))
        noise_x_train = np.zeros(shape=(self.config["num_data_points"]))
        noise_train = np.random.normal(scale=0.05, size=(t_steps + 1, self.config["num_data_points"]))
        noise_amplitude_train = np.random.normal(scale=0.5, size=(self.config["num_data_points"]))
        noise_test = np.zeros(shape=(t_steps + 1, self.config["num_data_points_test"]))
        y_train = np.sin(frequency * (x[:, None] + noise_x_train)) + noise_train + noise_amplitude_train[None]

        y_test = np.sin(frequency * x)[:, None] + noise_test
        y_test[:, :self.config["num_data_points_test"] // 8] += 2
        y_test[:, self.config["num_data_points_test"] // 8: 2 * self.config["num_data_points_test"] // 8] -= 2
        y_test[:, 2 * self.config["num_data_points_test"] // 8: 3 * self.config["num_data_points_test"] // 8] += 1
        y_test[:, 3 * self.config["num_data_points_test"] // 8: 4 * self.config["num_data_points_test"] // 8] -= 1
        y_test[:, 4 * self.config["num_data_points_test"] // 8: 5 * self.config["num_data_points_test"] // 8] -= 0
        y_test[:, 5 * self.config["num_data_points_test"] // 8: 6 * self.config["num_data_points_test"] // 8] += 4
        y_test[:, 6 * self.config["num_data_points_test"] // 8: 7 * self.config["num_data_points_test"] // 8] -= 4
        y_test[:, 7 * self.config["num_data_points_test"] // 8: 8 * self.config["num_data_points_test"] // 8] += 6

        labels = np.concatenate(
            (
                0 * np.ones(self.config["num_data_points_test"] // 8),
                1 * np.ones(self.config["num_data_points_test"] // 8),
                2 * np.ones(self.config["num_data_points_test"] // 8),
                3 * np.ones(self.config["num_data_points_test"] // 8),
                4 * np.ones(self.config["num_data_points_test"] // 8),
                5 * np.ones(self.config["num_data_points_test"] // 8),
                6 * np.ones(self.config["num_data_points_test"] // 8),
                7 * np.ones(self.config["num_data_points_test"] // 8),
            )
        )

        self.input_train_np = np.concatenate((np.tile(x[:, None, None], [1, self.config["num_data_points"], 1]),
                                              y_train[:, :, None]), axis=2)
        self.input_test_np = np.concatenate((np.tile(x[:, None, None], [1, self.config["num_data_points_test"], 1]),
                                             y_test[:, :, None]), axis=2)
        self.input_train_pca_np = self.input_train_np
        self.input_test_pca_np = self.input_test_np
        self.input_test_np_labels = labels
        self.input_t = list(range(self.config["num_steps"]+1))

    def select_phase(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train_np[:, idx], self.input_t, self.input_train_pca_np[:, idx]
        elif phase == "testing_y":
            for i in range(self.num_batches_test):
                yield self.input_test_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_t, \
                      self.input_test_pca_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_test_np_labels[i * self.b_size:(i+1) * self.b_size]
        else:
            raise ValueError("Invalid phase")

    def plot_data_point(self, data, axis):
        return None


class DataGeneratorToyDatasetExpandingCircle(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorToyDatasetExpandingCircle, self).__init__(config)

        np.random.seed(0)
        self.matrix_a = np.random.normal(size=(np.prod(self.config["state_size"]), 2))

        t_steps = self.config["num_steps"]

        r_i = 0.1
        r_f = 1.
        delta_r = (r_f - r_i) / t_steps

        angle_random_train = np.random.uniform(size=(self.config["num_data_points"], 1))
        angle_random_test = np.random.uniform(size=(self.config["num_data_points_test"], 1))
        radius_random_train = np.random.normal(scale=r_i / 25, size=(self.config["num_data_points"], 1))
        radius_random_test = np.random.normal(scale=r_i / 25, size=(self.config["num_data_points_test"], 1))

        x_steps_train = np.zeros((t_steps+1, self.n_points, 2))
        x_steps_test = np.zeros((t_steps+1, self.n_points_test, 2))

        x_0_train = np.concatenate(((r_i + radius_random_train) * np.cos(2 * np.pi * angle_random_train),
                                    (r_i + radius_random_train) * np.sin(2 * np.pi * angle_random_train)),
                                   axis=1)
        x_0_test = np.concatenate(((r_i + radius_random_test) * np.cos(2 * np.pi * angle_random_test),
                                   (r_i + radius_random_test) * np.sin(2 * np.pi * angle_random_test)),
                                  axis=1)

        x_steps_train[0] = x_0_train
        x_steps_test[0] = x_0_test

        for i in range(t_steps):
            norm_v_train = x_steps_train[i]/np.sqrt(x_steps_train[i, :, 0] ** 2 + x_steps_train[i, :, 1] ** 2)[:, None]
            norm_v_test = x_steps_test[i]/np.sqrt(x_steps_test[i, :, 0] ** 2 + x_steps_test[i, :, 1] ** 2)[:, None]
            x_steps_train[i+1] = x_steps_train[i] + norm_v_train * delta_r
            x_steps_test[i+1] = x_steps_test[i] + norm_v_test * delta_r

        self.input_train_pca_np = x_steps_train
        self.input_test_pca_np = x_steps_test

        self.input_train_np = np.einsum("ab,cdb->cda", self.matrix_a, self.input_train_pca_np)
        self.input_test_np = np.einsum("ab,cdb->cda", self.matrix_a, self.input_test_pca_np)
        self.input_t = list(range(self.config["num_steps"]+1))

    def select_phase(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train_np[:, idx], self.input_t, self.input_train_pca_np[:, idx]
        elif phase == "testing_y":
            for i in range(self.num_batches_test):
                yield self.input_test_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_t, \
                      self.input_test_pca_np[:, i * self.b_size:(i+1) * self.b_size]
        else:
            raise ValueError("Invalid phase")

    def plot_data_point(self, data, axis):
        return None


class DataGeneratorToyDatasetRotatingRectangle(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorToyDatasetRotatingRectangle, self).__init__(config)

        np.random.seed(0)
        self.matrix_a = np.random.normal(size=(np.prod(self.config["state_size"]), 2))
        t_steps = self.config["num_steps"]
        delta_t = self.config["delta_t"]

        x_max = 0.4
        y_max = 0.3

        num_points_train = self.config["num_data_points"]
        num_points_test = self.config["num_data_points_test"]

        frequency = 2 / (delta_t * 30)  # The 30 is to have a decent rotation at each time step
        trig_arg = frequency * delta_t

        rotation_matrix = np.array([[np.cos(trig_arg), -np.sin(trig_arg)], [np.sin(trig_arg), np.cos(trig_arg)]])

        x_steps_train = np.zeros((t_steps+1, self.n_points, 2))
        x_steps_test = np.zeros((t_steps+1, self.n_points_test, 2))

        down_train = np.concatenate((
            np.random.uniform(low=-x_max, high=x_max, size=(num_points_train // 4, 1)),
            -0.3 + np.random.normal(scale=0.01, size=(num_points_train // 4, 1))),
            axis=1)
        down_test = np.concatenate((
            np.random.uniform(low=-x_max, high=x_max, size=(num_points_test // 4, 1)),
            -0.3 + np.random.normal(scale=0.01, size=(num_points_test // 4, 1))),
            axis=1)
        up_train = np.concatenate((
            np.random.uniform(low=-x_max, high=x_max, size=(num_points_train // 4, 1)),
            0.3 + np.random.normal(scale=0.01, size=(num_points_train // 4, 1))),
            axis=1)
        up_test = np.concatenate((
            np.random.uniform(low=-x_max, high=x_max, size=(num_points_test // 4, 1)),
            0.3 + np.random.normal(scale=0.01, size=(num_points_test // 4, 1))),
            axis=1)
        left_train = np.concatenate((
            -0.4 + np.random.normal(scale=0.01, size=(num_points_train // 4, 1)),
            np.random.uniform(low=-y_max, high=y_max, size=(num_points_train // 4, 1))),
            axis=1)
        left_test = np.concatenate((
            -0.4 + np.random.normal(scale=0.01, size=(num_points_test // 4, 1)),
            np.random.uniform(low=-y_max, high=y_max, size=(num_points_test // 4, 1))),
            axis=1)
        right_train = np.concatenate((
            0.4 + np.random.normal(scale=0.01, size=(num_points_train // 4, 1)),
            np.random.uniform(low=-y_max, high=y_max, size=(num_points_train // 4, 1))),
            axis=1)
        right_test = np.concatenate((
            0.4 + np.random.normal(scale=0.01, size=(num_points_test // 4, 1)),
            np.random.uniform(low=-y_max, high=y_max, size=(num_points_test // 4, 1))),
            axis=1)

        x_0_train = np.concatenate((up_train, down_train, left_train, right_train), axis=0)
        x_0_test = np.concatenate((up_test, down_test, left_test, right_test), axis=0)

        x_steps_train[0] = x_0_train
        x_steps_test[0] = x_0_test

        for i in range(t_steps):
            x_steps_train[i+1] = np.einsum("ab,cb->ca", rotation_matrix, x_steps_train[i])
            x_steps_test[i+1] = np.einsum("ab,cb->ca", rotation_matrix, x_steps_test[i])

        self.input_train_pca_np = x_steps_train
        self.input_test_pca_np = x_steps_test

        self.input_train_np = np.einsum("ab,cdb->cda", self.matrix_a, self.input_train_pca_np)
        self.input_test_np = np.einsum("ab,cdb->cda", self.matrix_a, self.input_test_pca_np)
        self.input_t = list(range(self.config["num_steps"]+1))

    def select_phase(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train_np[:, idx], self.input_t, self.input_train_pca_np[:, idx]
        elif phase == "testing_y":
            for i in range(self.num_batches_test):
                yield self.input_test_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_t, \
                      self.input_test_pca_np[:, i * self.b_size:(i+1) * self.b_size]
        else:
            raise ValueError("Invalid phase")

    def plot_data_point(self, data, axis):
        return None


class DataGeneratorToyDatasetLinearCombination(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorToyDatasetLinearCombination, self).__init__(config)

        np.random.seed(0)
        self.matrix_a = np.random.normal(size=(np.prod(self.config["state_size"]), 2))
        t_steps = self.config["num_steps"]
        delta_t = self.config["delta_t"]
        frequency = 2 / (delta_t * 30)  # The 30 is to have a decent rotation at each time step
        trig_arg = frequency * delta_t

        rotation_matrix = np.array([[np.cos(trig_arg), -np.sin(trig_arg)], [np.sin(trig_arg), np.cos(trig_arg)]])

        x_steps_train = np.zeros((t_steps+1, self.n_points, 2))
        x_steps_test = np.zeros((t_steps+1, self.n_points_test, 2))

        x_0_train, _ = make_moons(self.n_points, noise=.05)
        x_0_test, _ = make_moons(self.n_points_test, noise=.05)

        x_steps_train[0] = x_0_train
        x_steps_test[0] = x_0_test

        for i in range(t_steps):
            noise_train = np.random.normal(scale=0.03, size=[self.n_points, 2])
            noise_test = np.random.normal(scale=0.03, size=[self.n_points_test, 2])
            x_steps_train[i+1] = np.einsum("ab,cb->ca", rotation_matrix, x_steps_train[i]) + noise_train
            x_steps_test[i+1] = np.einsum("ab,cb->ca", rotation_matrix, x_steps_test[i]) + noise_test

        self.input_train_pca_np = x_steps_train
        self.input_test_pca_np = x_steps_test

        self.input_train_np = np.einsum("ab,cdb->cda", self.matrix_a, self.input_train_pca_np)
        self.input_test_np = np.einsum("ab,cdb->cda", self.matrix_a, self.input_test_pca_np)
        self.input_t = list(range(self.config["num_steps"]+1))

    def select_phase(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train_np[:, idx], self.input_t, self.input_train_pca_np[:, idx]
        elif phase == "testing_y":
            for i in range(self.num_batches_test):
                yield self.input_test_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_t, \
                      self.input_test_pca_np[:, i * self.b_size:(i+1) * self.b_size]
        else:
            raise ValueError("Invalid phase")

    def plot_data_point(self, data, axis):
        return None


class DataGeneratorToyDatasetLinearCombinationDifferentRotations(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorToyDatasetLinearCombinationDifferentRotations, self).__init__(config)

        np.random.seed(0)
        self.matrix_a = np.random.normal(size=(np.prod(self.config["state_size"]), 2))
        t_steps = self.config["num_steps"]
        delta_t = self.config["delta_t"]
        frequency = 2 / (delta_t * 30)  # The 30 is to have a decent rotation at each time step
        trig_arg = frequency * delta_t

        rotation_matrix = np.array([[np.cos(trig_arg), -np.sin(trig_arg)], [np.sin(trig_arg), np.cos(trig_arg)]])
        inverse_rotation_matrix = np.array([[np.cos(trig_arg), np.sin(trig_arg)],
                                            [-np.sin(trig_arg), np.cos(trig_arg)]])

        x_steps_train = np.zeros((t_steps+1, self.n_points, 2))
        x_steps_test = np.zeros((t_steps+1, self.n_points_test, 2))

        x_0_train, y_train = make_moons(self.n_points, noise=.05)
        x_0_test, y_test = make_moons(self.n_points_test, noise=.05)

        x_steps_train[0] = x_0_train
        x_steps_test[0] = x_0_test

        for i in range(t_steps):
            noise_train = np.random.normal(scale=0.03, size=[self.n_points, 2])
            noise_test = np.random.normal(scale=0.03, size=[self.n_points_test, 2])
            x_steps_train[i+1] = x_steps_train[i]
            x_steps_train[i+1][y_train == 0] = \
                np.einsum("ab,cb->ca", rotation_matrix, x_steps_train[i][y_train == 0])
            x_steps_train[i+1][y_train == 1] = \
                np.einsum("ab,cb->ca", inverse_rotation_matrix, x_steps_train[i][y_train == 1])
            x_steps_train[i+1] += noise_train
            x_steps_test[i+1] = x_steps_test[i]
            x_steps_test[i+1][y_test == 0] = \
                np.einsum("ab,cb->ca", rotation_matrix, x_steps_test[i][y_test == 0])
            x_steps_test[i+1][y_test == 1] = \
                np.einsum("ab,cb->ca", inverse_rotation_matrix, x_steps_test[i][y_test == 1])
            x_steps_test[i+1] += noise_test

        self.input_train_pca_np = x_steps_train
        self.input_test_pca_np = x_steps_test

        self.input_train_np = np.einsum("ab,cdb->cda", self.matrix_a, self.input_train_pca_np)
        self.input_test_np = np.einsum("ab,cdb->cda", self.matrix_a, self.input_test_pca_np)
        self.input_t = list(range(self.config["num_steps"]+1))

    def select_phase(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train_np[:, idx], self.input_t, self.input_train_pca_np[:, idx]
        elif phase == "testing_y":
            for i in range(self.num_batches_test):
                yield self.input_test_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_t, \
                      self.input_test_pca_np[:, i * self.b_size:(i+1) * self.b_size]
        else:
            raise ValueError("Invalid phase")

    def plot_data_point(self, data, axis):
        return None


class DataGeneratorToyDatasetLinearCombinationWithSoftplus(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorToyDatasetLinearCombinationWithSoftplus, self).__init__(config)

        np.random.seed(0)
        intermediate_state = 2 * np.prod(self.config["state_size"])
        self.matrix_a = np.random.normal(size=(intermediate_state, 2))
        self.matrix_b = np.random.normal(size=(np.prod(self.config["state_size"]), intermediate_state))

        t_steps = self.config["num_steps"]
        delta_t = self.config["delta_t"]
        frequency = 2 / (delta_t * 30)  # The 30 is to have a decent rotation at each time step
        trig_arg = frequency * delta_t

        rotation_matrix = np.array([[np.cos(trig_arg), -np.sin(trig_arg)], [np.sin(trig_arg), np.cos(trig_arg)]])

        x_steps_train = np.zeros((t_steps+1, self.n_points, 2))
        x_steps_test = np.zeros((t_steps+1, self.n_points_test, 2))

        x_0_train, _ = make_moons(self.n_points, noise=.05)
        x_0_test, _ = make_moons(self.n_points_test, noise=.05)

        x_steps_train[0] = x_0_train
        x_steps_test[0] = x_0_test

        for i in range(t_steps):
            noise_train = np.random.normal(scale=0.03, size=[self.n_points, 2])
            noise_test = np.random.normal(scale=0.03, size=[self.n_points_test, 2])
            x_steps_train[i+1] = np.einsum("ab,cb->ca", rotation_matrix, x_steps_train[i]) + noise_train
            x_steps_test[i+1] = np.einsum("ab,cb->ca", rotation_matrix, x_steps_test[i]) + noise_test

        self.input_train_pca_np = x_steps_train
        self.input_test_pca_np = x_steps_test

        self.input_train_np = \
            np.einsum('ab,cdb->cda',
                      self.matrix_b,
                      softplus(np.einsum("ab,cdb->cda", self.matrix_a, self.input_train_pca_np)))
        self.input_test_np = \
            np.einsum('ab,cdb->cda',
                      self.matrix_b,
                      softplus(np.einsum("ab,cdb->cda", self.matrix_a, self.input_test_pca_np)))
        self.input_t = list(range(self.config["num_steps"]+1))

    def select_phase(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train_np[:, idx], self.input_t, self.input_train_pca_np[:, idx]
        elif phase == "testing_y":
            for i in range(self.num_batches_test):
                yield self.input_test_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_t, \
                      self.input_test_pca_np[:, i * self.b_size:(i+1) * self.b_size]
        else:
            raise ValueError("Invalid phase")

    def plot_data_point(self, data, axis):
        return None


class DataGeneratorEEGEyeState(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorEEGEyeState, self).__init__(config)

        # http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State

        data, meta = arff.loadarff("Data/time_series_data/eeg_eye_state.arff")

        num_samples = 14980 // (self.config["num_steps"] + 1)
        data = np.array(data.tolist())
        data_values = np.asarray(data[:num_samples*(self.config["num_steps"] + 1), :-1], dtype=np.float32)
        data_values = data_values.reshape((num_samples, self.config["num_steps"]+1, 14))
        # data_labels = np.asarray(data[:, -1], dtype=np.int32).reshape((num_samples, self.config["num_steps"], 1))
        # x_train, x_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=100, random_state=0)
        x_train, x_test = train_test_split(data_values, test_size=100, random_state=0)

        x_train = np.transpose(x_train, (1, 0, 2))
        x_test = np.transpose(x_test, (1, 0, 2))

        self.input_train_np = self.standardize(x_train)
        self.input_train_pca_np = x_train

        self.input_test_np = self.standardize(x_test)
        self.input_test_pca_np = x_test
        # self.input_test_np_labels = y_test

    def select_phase(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train_np[:, idx], self.input_train_pca_np[:, idx]
        elif phase == "testing_y":
            for i in range(self.num_batches_test):
                yield self.input_test_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_test_pca_np[:, i * self.b_size:(i+1) * self.b_size]
        else:
            raise ValueError("Invalid phase")

    def plot_data_point(self, data, axis):
        return None


class DataGeneratorExchangeRate(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorExchangeRate, self).__init__(config)

        # https://github.com/laiguokun/multivariate-time-series-data

        data = np.loadtxt("Data/time_series_data/exchange_rate.txt.gz", delimiter=",")
        num_samples = self.config["num_data_points"] + self.config["num_data_points_test"]
        num_samples_test = self.config["num_data_points_test"]

        data = data[:num_samples*(self.config["num_steps"] + 1)].reshape(
            (self.config["num_steps"] + 1, -1, 8))

        self.input_t = list(range(self.config["num_steps"] + 1))
        self.input_train_np = data[:, :num_samples-num_samples_test]
        self.input_train_pca_np = self.input_train_np

        self.input_test_np = data[:, num_samples-num_samples_test:]
        self.input_test_pca_np = self.input_test_np
        # self.input_test_np_labels = y_test

    def select_phase(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train_np[:, idx], self.input_t, self.input_train_pca_np[:, idx]
        elif phase == "testing_y":
            for i in range(self.num_batches_test):
                yield self.input_test_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_t, \
                      self.input_test_pca_np[:, i * self.b_size:(i+1) * self.b_size]
        else:
            raise ValueError("Invalid phase")

    def plot_data_point(self, data, axis):
        return None


class DataGeneratorMissile2Air(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorMissile2Air, self).__init__(config)

        # (2011) Lazaro-Gredilla, M - Overlapping Mixtures of Gaussian Processes for the Data Association Problem
        # (2001) R. Karlsson, F. Gustafsson, Monte Carlo data association for multiple target tracking

        delta_t = 60 / self.config["num_steps"]
        n_samples = self.config["num_data_points"]
        n_samples_test = self.config["num_data_points_test"]

        # Normalizing factor of initial states, using s_1, since it's the one with the biggest norm
        normalizing_factor = np.sqrt(6500**2 + 1000**2 + 2000**2 + 50**2 + 100**2) * 1
        norm = np.pi / 2  # arctan normalization

        q_matrix = np.eye(3) * 10 / normalizing_factor**2
        r_matrix = np.eye(3) * 0.01 / norm**2
        r_matrix[0, 0] = 50 / normalizing_factor**2

        state_evolution_matrix = np.eye(6)
        state_evolution_matrix[0, 3] = delta_t
        state_evolution_matrix[1, 4] = delta_t
        state_evolution_matrix[2, 5] = delta_t

        state_random_matrix = np.vstack([delta_t**2 / 2 * np.eye(3), delta_t * np.eye(3)])

        v_random_vectors_train = np.random.multivariate_normal(
            mean=np.zeros(3),
            cov=q_matrix,
            size=[self.config["num_steps"], n_samples])
        v_random_vectors_test = np.random.multivariate_normal(
            mean=np.zeros(3),
            cov=q_matrix,
            size=[self.config["num_steps"], n_samples_test])
        e_random_vectors_train = np.random.multivariate_normal(
            mean=np.zeros(3),
            cov=r_matrix,
            size=[self.config["num_steps"]+1, n_samples]
        )
        e_random_vectors_test = np.random.multivariate_normal(
            mean=np.zeros(3),
            cov=r_matrix,
            size=[self.config["num_steps"]+1, n_samples_test]
        )

        initial_state_1 = np.array([6500., -1000., 2000., -50., 100., 0.]) / normalizing_factor
        initial_state_2 = np.array([5050., -450., 2000., 100., 50., 0.]) / normalizing_factor
        initial_state_3 = np.array([8000., 500., 2000., -100., 0., 0.]) / normalizing_factor

        latent_states_train = np.zeros([self.config["num_steps"]+1, n_samples, 6])
        labels_train = np.zeros(n_samples, dtype=np.int8)
        latent_states_test = np.zeros([self.config["num_steps"]+1, n_samples_test, 6])
        labels_test = np.zeros(n_samples_test, dtype=np.int8)

        if self.config["num_trajectories"] == 1:
            latent_states_train[0, :] = initial_state_1
            labels_train[:] = 0
            latent_states_test[0, :] = initial_state_1
            labels_test[:] = 0
        elif self.config["num_trajectories"] == 2:
            # From initial state 1
            latent_states_train[0, :n_samples // 2] = initial_state_1
            labels_train[:n_samples // 2] = 0
            latent_states_test[0, :n_samples_test // 2] = initial_state_1
            labels_test[:n_samples_test // 2] = 0
            # From initial state 2
            latent_states_train[0, n_samples // 2:] = initial_state_2
            labels_train[n_samples // 2:] = 1
            latent_states_test[0, n_samples_test // 2:] = initial_state_2
            labels_test[n_samples_test // 2:] = 1
        else:
            # From initial state 1
            latent_states_train[0, :n_samples // 3] = initial_state_1
            labels_train[:n_samples // 3] = 0
            latent_states_test[0, :n_samples_test // 3] = initial_state_1
            labels_test[:n_samples_test // 3] = 0
            # From initial state 2
            latent_states_train[0, n_samples // 3:2 * n_samples // 3] = initial_state_2
            labels_train[n_samples // 3: 2 * n_samples] = 1
            latent_states_test[0, n_samples_test // 3:2 * n_samples_test // 3] = initial_state_2
            labels_test[n_samples_test // 3: 2 * n_samples_test] = 1
            # From initial state 3
            latent_states_train[0, 2 * n_samples // 3:] = initial_state_3
            labels_train[2 * n_samples // 3:] = 2
            latent_states_test[0, 2 * n_samples_test // 3:] = initial_state_3
            labels_test[2 * n_samples_test // 3:] = 2

        for i in range(1, self.config["num_steps"]+1):
            latent_states_train[i] = np.einsum("abc,ac->ab",
                                               np.tile([state_evolution_matrix], [n_samples, 1, 1]),
                                               latent_states_train[i-1]
                                               )
            latent_states_train[i] += np.einsum("abc,ac->ab",
                                                np.tile([state_random_matrix], [n_samples, 1, 1]),
                                                v_random_vectors_train[i-1]
                                                )
            latent_states_test[i] = np.einsum("abc,ac->ab",
                                              np.tile([state_evolution_matrix], [n_samples_test, 1, 1]),
                                              latent_states_test[i-1]
                                              )
            latent_states_test[i] += np.einsum("abc,ac->ab",
                                               np.tile([state_random_matrix], [n_samples_test, 1, 1]),
                                               v_random_vectors_test[i-1]
                                               )

        # Standardize latent space
        # latent_states_train = self.standardize(latent_states_train, True)
        # latent_states_test = self.standardize(latent_states_test, True)

        self.input_train_np = np.zeros([self.config["num_steps"]+1, n_samples, 3])
        self.input_test_np = np.zeros([self.config["num_steps"]+1, n_samples_test, 3])

        # First y component
        self.input_train_np[:, :, 0] = np.sqrt(
            latent_states_train[:, :, 0]**2 + latent_states_train[:, :, 1]**2 + latent_states_train[:, :, 3]**2
        )
        self.input_test_np[:, :, 0] = np.sqrt(
            latent_states_test[:, :, 0]**2 + latent_states_test[:, :, 1]**2 + latent_states_test[:, :, 3]**2
        )
        # Second y component
        self.input_train_np[:, :, 1] = np.arctan(latent_states_train[:, :, 1] / latent_states_train[:, :, 0]) / norm
        self.input_test_np[:, :, 1] = np.arctan(latent_states_test[:, :, 1] / latent_states_test[:, :, 0]) / norm
        # Third y component
        self.input_train_np[:, :, 0] = np.arctan(
             -latent_states_train[:, :, 3] / np.sqrt(latent_states_train[:, :, 1]**2 + latent_states_train[:, :, 0]**2)
        ) / norm
        self.input_test_np[:, :, 0] = np.arctan(
            -latent_states_test[:, :, 3] / np.sqrt(latent_states_test[:, :, 1]**2 + latent_states_test[:, :, 0]**2)
        ) / norm
        # Y Noise
        self.input_train_np += e_random_vectors_train
        self.input_test_np += e_random_vectors_test

        self.input_t = list(range(self.config["num_steps"]+1))
        self.input_train_pca_np = latent_states_train[:, :, :2]
        self.input_test_pca_np = latent_states_test[:, :, :2]
        self.input_test_np_labels = labels_test

    def select_phase(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train_np[:, idx], self.input_t, self.input_train_pca_np[:, idx]
        elif phase == "testing_y":
            for i in range(self.num_batches_test):
                yield self.input_test_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_t, \
                      self.input_test_pca_np[:, i * self.b_size:(i+1) * self.b_size], \
                      self.input_test_np_labels[i * self.b_size:(i+1) * self.b_size]
        else:
            raise ValueError("Invalid phase")

    def plot_data_point(self, data, axis):
        return None
