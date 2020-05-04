import argparse
import json
import os
import shutil
from collections import OrderedDict

from jax.config import config as cfig

import base_model as bm
import config_dicts as cd
import data_generators as dg
import models
import predictors
import trainers


def run(config, model, trainer, predictor, data, args):
    main_scope = OrderedDict()
    model_instance = model(main_scope, config)

    if args.restore:
        model_instance.load()
        model_instance.increment_cur_epoch_var()

    logger = bm.Logger(config, args.predict)

    if args.predict == 0:
        trainer_instance = trainer(model_instance, data, config, logger)
        trainer_instance.train()
    elif args.predict == 1:
        trainer_instance = trainer(model_instance, data, config, logger)
        trainer_instance.train()

        # Restore best model
        model_instance.load()
        predictor_instance = predictor(model_instance, data, config, logger)
        predictor_instance.predict()
    else:
        model_instance.load()
        predictor_instance = predictor(model_instance, data, config, logger)
        predictor_instance.predict()

    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Trains VAE-GPLVM model")
    parser.add_argument("-p", "--predict",
                        help="run predictions (0 - only train, 1 - train and predict, 2 - only predict)",
                        action="count", default=0)
    parser.add_argument("-d", "--debug", help="start TF debug", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--restore", help="restore checkpoint", action="store_true")
    group.add_argument("-e", "--erase", help="erase summary dirs before training", action="store_true")
    args = parser.parse_args()

    if args.debug:
        cfig.update("jax_debug_nans", True)

    # Time-series datasets
    # datasets = ["toy_dataset_linear_combination", "toy_dataset_linear_combination_different_rotations",
    #             "toy_dataset_linear_combination_with_softplus", "toy_dataset_expanding_circle",
    #             "toy_dataset_rotating_rectangle", "toy_dataset_sine_draw",
    #             "toy_dataset_1d_chirp",
    #             "eeg_eye_state", "exchange_rate",
    #             "missile_to_air"]
    datasets = ["toy_dataset_1d_chirp"]

    model_name = "gp_sde"

    constant_diffusion = False
    constant_diffusion_legend = {False: "varying_diffusion", True: "constant_diffusion"}

    # time_dependent_drifts = [False, True]
    time_dependent_drifts = [False]
    time_dependent_drift_legend = {False: "time_independent_drift", True: "time_dependent_drift"}

    for time_dependent_drift in time_dependent_drifts:
        for dataset in datasets:
            parent_folder = f"results/{model_name}/"
            parent_folder += f"{time_dependent_drift_legend[time_dependent_drift]}/"
            parent_folder += f"{constant_diffusion_legend[constant_diffusion]}/{dataset}"

            model = None
            trainer = None
            predictor = None

            if model_name == "gp_sde":
                model = models.GPSDE
                trainer = trainers.GPSDETrainer
                predictor = predictors.GPSDEPredictor

            config = cd.config_dict(parent_folder, dataset)
            config["time_dependent_gp"] = time_dependent_drift

            config["constant_diffusion"] = constant_diffusion

            if not constant_diffusion:
                config["solver"] = "EulerMaruyamaSolver"

            # Create the experiment dirs
            if args.erase:
                if os.path.exists(config["summary_dir"]):
                    shutil.rmtree(config["summary_dir"], ignore_errors=True)
                    shutil.rmtree(config["checkpoint_dir"], ignore_errors=True)

            config["results_dir"] += f"{config['delta_t']}deltaT_{config['num_steps']}tsteps"
            bm.create_dirs([config["summary_dir"], config["checkpoint_dir"], config["results_dir"]])

            with open(os.path.join(config["summary_dir"], "config_dict.txt"), 'w') as f:
                f.write(json.dumps(config))

            data = None
            if dataset == "toy_dataset_sine_draw":
                data = dg.DataGeneratorToyDatasetSineDraw(config)
            elif dataset == "toy_dataset_1d_chirp":
                data = dg.DataGeneratorToyDataset1DChirp(config)
            elif dataset == "toy_dataset_expanding_circle":
                data = dg.DataGeneratorToyDatasetExpandingCircle(config)
            elif dataset == "toy_dataset_rotating_rectangle":
                data = dg.DataGeneratorToyDatasetRotatingRectangle(config)
            elif dataset == "toy_dataset_linear_combination":
                data = dg.DataGeneratorToyDatasetLinearCombination(config)
            elif dataset == "toy_dataset_linear_combination_different_rotations":
                data = dg.DataGeneratorToyDatasetLinearCombinationDifferentRotations(config)
            elif dataset == "toy_dataset_linear_combination_with_softplus":
                data = dg.DataGeneratorToyDatasetLinearCombinationWithSoftplus(config)
            elif dataset == "eeg_eye_state":
                data = dg.DataGeneratorEEGEyeState(config)
            elif dataset == "exchange_rate":
                data = dg.DataGeneratorExchangeRate(config)
            elif dataset == "missile_to_air":
                data = dg.DataGeneratorMissile2Air(config)

            run(config, model, trainer, predictor, data, args)


if __name__ == "__main__":
    main()
