def config_dict(parent_folder, dataset):

    num_steps = 50

    if dataset == "toy_dataset_sine_draw":
        config = {
            # Data properties
            "batch_size": 20,
            "num_data_points": 500,
            "num_data_points_test": 40,
            "state_size": [2],
            # Number of inducing points for GP
            "num_ind_points_beta": 10,
            "num_ind_points_gamma": 10,
            # Plot options
            "label_latent_manifold": True,
            "multiple_1d_data": False,
            "y_max": 1,
            "y_min": -1,
            "x_max": 1,
            "x_min": -1,
            # True latent dims == 2
            "2d_latent_space": False,
        }
    elif dataset == "toy_dataset_1d_chirp":
        config = {
            # Data properties
            "batch_size": 20,
            "num_data_points": 500,
            "num_data_points_test": 40,
            "state_size": [2],
            # Number of inducing points for GP
            "num_ind_points_beta": 10,
            "num_ind_points_gamma": 10,
            # Plot options
            "label_latent_manifold": True,
            "multiple_1d_data": False,
            "y_max": 1,
            "y_min": -1,
            "x_max": 1,
            "x_min": -1,
            # True latent dims == 2
            "2d_latent_space": False,
        }
    elif dataset == "toy_dataset_expanding_circle":
        config = {
            # Data properties
            "batch_size": 20,
            "num_data_points": 500,
            "num_data_points_test": 100,
            "state_size": [8],
            # Number of inducing points for GP
            "num_ind_points_beta": 30,
            "num_ind_points_gamma": 30,
            # Plot options
            "label_latent_manifold": False,
            "multiple_1d_data": False,
            "y_max": 1,
            "y_min": -1,
            "x_max": 1,
            "x_min": -1,
            # True latent dims == 2
            "2d_latent_space": True,
        }
    elif dataset == "toy_dataset_rotating_rectangle":
        config = {
            # Data properties
            "batch_size": 20,
            "num_data_points": 500,
            "num_data_points_test": 100,
            "state_size": [8],
            # Number of inducing points for GP
            "num_ind_points_beta": 20,
            "num_ind_points_gamma": 20,
            # Plot options
            "label_latent_manifold": False,
            "multiple_1d_data": False,
            "y_max": 0.6,
            "y_min": -0.6,
            "x_max": 0.6,
            "x_min": -0.6,
            # True latent dims == 2
            "2d_latent_space": True,
        }
    elif dataset == "toy_dataset_linear_combination":
        config = {
            # Data properties
            "batch_size": 20,
            "num_data_points": 500,
            "num_data_points_test": 100,
            "state_size": [8],
            # Number of inducing points for GP
            "num_ind_points_beta": 20,
            "num_ind_points_gamma": 20,
            # Plot options
            "label_latent_manifold": False,
            "multiple_1d_data": False,
            # True latent dims == 2
            "2d_latent_space": True,
        }
    elif dataset == "toy_dataset_linear_combination_different_rotations":
        config = {
            # Data properties
            "batch_size": 20,
            "num_data_points": 500,
            "num_data_points_test": 100,
            "state_size": [8],
            # Number of inducing points for GP
            "num_ind_points_beta": 20,
            "num_ind_points_gamma": 20,
            # Plot options
            "label_latent_manifold": False,
            "multiple_1d_data": False,
            # True latent dims == 2
            "2d_latent_space": True,
        }
    elif dataset == "toy_dataset_linear_combination_with_softplus":
        config = {
            # Data properties
            "batch_size": 20,
            "num_data_points": 500,
            "num_data_points_test": 100,
            "state_size": [8],
            # Number of inducing points for GP
            "num_ind_points_beta": 20,
            "num_ind_points_gamma": 20,
            # Plot options
            "label_latent_manifold": False,
            "multiple_1d_data": False,
            # True latent dims == 2
            "2d_latent_space": True,
        }
    elif dataset == "eeg_eye_state":
        config = {
            # Data properties
            "batch_size": 20,
            "num_data_points": 383,
            "num_data_points_test": 100,
            "state_size": [14],
            # Number of inducing points for GP
            "num_ind_points_beta": 20,
            "num_ind_points_gamma": 20,
            # Plot options
            "label_latent_manifold": False,
            "multiple_1d_data": False,
            # True latent dims == 2
            "2d_latent_space": False,
        }
    elif dataset == "exchange_rate":
        config = {
            # Data properties
            "batch_size": 20,
            "num_data_points": 7588 // (num_steps + 1) - 60,
            "num_data_points_test": 60,
            "state_size": [8],
            # Number of inducing points for GP
            "num_ind_points_beta": 30,
            "num_ind_points_gamma": 30,
            # Plot options
            "label_latent_manifold": False,
            "multiple_1d_data": True,
            # True latent dims == 2
            "2d_latent_space": False,
        }
    elif dataset == "missile_to_air":
        # Number of trajectories (up to 3)
        num_trajectories = 3
        parent_folder += f"/{num_trajectories}trajectories"
        config = {
            # Data properties
            "batch_size": 20,
            "num_data_points": 200 * num_trajectories,
            "num_data_points_test": 60 * num_trajectories,
            "state_size": [3],
            # Number of trajectories (up to 3)
            "num_trajectories": num_trajectories,
            # Number of inducing points for GP
            "num_ind_points_beta": 30,
            "num_ind_points_gamma": 30,
            # Plot options
            "label_latent_manifold": True,
            "multiple_1d_data": False,
            # True latent dims == 2
            "2d_latent_space": True,
        }
    else:
        raise RuntimeError("Invalid dataset")

    config["dataset"] = dataset
    # Folders where everything is saved
    config["summary_dir"] = f"{parent_folder}/summary/"
    config["results_dir"] = f"{parent_folder}/results/"
    config["checkpoint_dir"] = f"{parent_folder}/checkpoint/"
    # Max number of checkpoints to keep
    config["max_to_keep"] = 5

    # Training parameters
    config["learning_rate_global"] = 0.01

    # Flow parameters
    config["num_steps"] = num_steps
    config["delta_t"] = 1. / config["num_steps"]
    config["num_steps_test"] = num_steps + 10
    # config["solver"] = "StrongOrder3HalfsSolver"
    config["solver"] = "EulerMaruyamaSolver"

    # Prediction samples
    if dataset == "toy_dataset_sine_draw" or dataset == "toy_dataset_1d_chirp":
        config["prediction_samples"] = 100
    else:
        config["prediction_samples"] = 20

    # Max training epochs
    config["num_epochs"] = 5000

    # Number of iterations per epoch
    config["num_iter_per_epoch"] = config["num_data_points"] // config["batch_size"]
    config["num_iter_per_epoch_test"] = config["num_data_points_test"] // config["batch_size"]

    return config
