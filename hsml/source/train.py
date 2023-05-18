#!/usr/bin/env python3

import argparse
import json
import os
import warnings
import yaml

from matplotlib import pyplot as plt
import optuna
from sklearn.model_selection import KFold

import engine
from utils import plot_meta_train_losses, set_random_seeds, set_cuda_reproducibility, \
                  set_model_args


# Suppress warning that occurs due to accessing a non-leaf tensor
warnings.filterwarnings("ignore")


def objective(trial, ht_config, data_config, data_filenames, embeddings):
    # TODO: Update docstring
    """Training process of the hyperparameter tuning process.

    Firstly, the cross-validation schema is defined. Based on that, the model is trained for
    a specific set of hyperparameter values and the mean query set loss of the meta-validation
    tasks, which is the objective value to be minimized, is calcualted.

    Args:
        trial: An optuna trial object, handled solely by optuna.
        ht_config: A dictionary that defines the hyperparameter search space.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        data_filenames: A list of strings that contains the paths to the training tasks.
    Returns:
        A float that represents the mean query set loss of the meta-validation
        tasks.
    """

    # Hyperparameter search space
    config = {
        'kfolds': ht_config['k_folds'],
        'task_batch_size': ht_config['task_batch_size'],
        'sample_batch_size': ht_config['sample_batch_size'],
        'num_inner_steps': ht_config['num_inner_steps'],
        'eta_min': float(ht_config['eta_min']),
        'second_order': ht_config['second_order'],
        'second_to_first_order_epoch': ht_config['second_to_first_order_epoch'],
        'num_levels': ht_config['num_levels'],
        'num_centers': ht_config['num_centers'],
        'sigma': ht_config['sigma'],
        'embedding_size': ht_config['embedding_size'],
        'loss': ht_config['loss'],
        'kappa': trial.suggest_float('kappa',
                                     ht_config['kappa']['lower_bound'],
                                     ht_config['kappa']['upper_bound']),
        'train_epochs': trial.suggest_int(
            'train_epochs',
            int(ht_config['train_epochs']['lower_bound']),
            int(ht_config['train_epochs']['upper_bound'])
        ),
        'lstm_hidden_units': trial.suggest_int(
            'lstm_hidden_units',
            int(ht_config['lstm_hidden_units']['upper_bound']),
            int(ht_config['lstm_hidden_units']['upper_bound'])
        ),
        'init_learning_rate': trial.suggest_float(
            'init_learning_rate',
            float(ht_config['init_learning_rate']['lower_bound']),
            float(ht_config['init_learning_rate']['upper_bound']),
            log=ht_config['init_learning_rate']['log']
        ),
        'meta_learning_rate': trial.suggest_float(
            'meta_learning_rate',
            float(ht_config['meta_learning_rate']['lower_bound']),
            float(ht_config['meta_learning_rate']['upper_bound']),
            log=ht_config['meta_learning_rate']['log']
        )
    }

    mean_fold_val_losses = {}  # Contains the mean validation loss of each fold

    k_folds = config['kfolds']
    args = set_model_args(config)

    # Cross-validation splits - they should be random since we are splitting the tasks
    # (non-sequential) and not a single timeseries (sequential)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_idx,val_idx) in enumerate(kfold.split(data_filenames)):
        meta_learner = engine.build_meta_learner(args=args,
                                                 data_config=data_config)

        # Tasks used for training within this fold
        train_filenames = [data_filenames[idx] for idx in train_idx]

        # Tasks used for validation within this fold
        val_filenames = [data_filenames[idx] for idx in val_idx]

        # Meta-Train
        meta_learner.meta_train(train_filenames, embeddings, optimal_mode=False)

        # Meta-Evaluation
        mean_fold_val_loss = meta_learner.meta_test(data_filenames=val_filenames,
                                                    embeddings=embeddings,
                                                    optimal_mode=False)

        mean_fold_val_losses[fold] = mean_fold_val_loss

    val_loss_sum = 0.0
    for _, value in mean_fold_val_losses.items():
        val_loss_sum += value
    mean_val_loss = val_loss_sum / len(mean_fold_val_losses.items())

    return mean_val_loss


def hyperparameter_tuning(n_trials,
                          results_dir_name,
                          ht_config,
                          data_config,
                          data_filenames,
                          embeddings):
    # TODO: Update docstring
    """Perform hyperparameter tuning of the model on the given search space.

    This is done using the optuna library. Additionally to defining the best hyperparameters,
    a number of plots is created and saved that gives additional insights on the process.

    Args:
        n_trials: An integer that defines the total number of hyperparameter values' combinations
            to be tried out.
        results_dir_name: A string with the name of the directory the results will be saved.
        ht_config: A dictionary that defines the hyperparameter search space.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        data_filenames: A list of strings that contains the paths to the training tasks.
    Returns:
        A dictionary that contains the optimal hyperparameter values.
    """

    # Hyperparameter tuning process
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                direction='minimize')
    study.optimize(
        lambda trial: objective(trial, ht_config, data_config, data_filenames, embeddings),
        n_trials=n_trials
    )

    # Create visializations regarding the hyperparameter tuning process
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    target_file = results_dir_name + 'parallel_coordinates.png'
    plt.savefig(target_file)

    optuna.visualization.matplotlib.plot_slice(study)
    target_file = results_dir_name + 'slice_plot.png'
    plt.savefig(target_file)

    optuna.visualization.matplotlib.plot_param_importances(study)
    target_file = results_dir_name + 'hyperparameter_importances.png'
    plt.savefig(target_file)

    optuna.visualization.matplotlib.plot_optimization_history(study)
    target_file = results_dir_name + 'optimization_history.png'
    plt.savefig(target_file)

    # Save info about each combination of hyperparameters tried out
    trials_df = study.trials_dataframe().drop(['state','datetime_start', 'datetime_complete'],
                                              axis=1)
    target_file = results_dir_name + 'trials.csv'
    trials_df.to_csv(target_file, index=False)

    # The best trial is the one that minimizes the objective value (mean validation loss)
    best_trial = trials_df[trials_df.value == trials_df.value.min()]

    opt_config = {
        'task_batch_size': ht_config['task_batch_size'],
        'sample_batch_size': ht_config['sample_batch_size'],
        'num_inner_steps': ht_config['num_inner_steps'],
        'eta_min': ht_config['eta_min'],
        'num_levels': ht_config['num_levels'],
        'num_centers': ht_config['num_centers'],
        'sigma': ht_config['sigma'],
        'embedding_size': ht_config['embedding_size'],
        'loss': ht_config['loss'],
        'kappa': best_trial['params_kappa'].values[0],
        'train_epochs': best_trial['params_train_epochs'].values[0],
        'lstm_hidden_units': best_trial['params_lstm_hidden_units'].values[0],
        'init_learning_rate': best_trial['params_init_learning_rate'].values[0],
        'meta_learning_rate': best_trial['params_meta_learning_rate'].values[0],
        'second_order': ht_config['second_order'],
        'second_to_first_order_epoch': ht_config['second_to_first_order_epoch']
    }

    return opt_config


def meta_train_optimal(opt_config, data_config, data_filenames, train_embeddings, results_dir_name):
    # TODO: Update docstring
    """Meta-Training process of the optimal model.

    The model is trained based on the optimal hyperparameters determined from the previously
    done hyperparameter tuning. The learning curve of the model during meta-training is plotted
    and the optimal weights and parameters of the meta-trained model are saved.

    Args:
        opt_config: A dictionary that contains the optimal hyperparameter values.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        data_filenames: A list of strings that contains the paths to the training tasks.
        results_dir_name: A string with the name of the directory the results will be saved.
    """

    args = set_model_args(opt_config)

    meta_learner = engine.build_meta_learner(args=args,
                                             data_config=data_config)

    # Meta-train model using the optimal hyperparameters
    _, epoch_mean_support_losses, epoch_mean_query_losses = meta_learner.meta_train(
        data_filenames,
        train_embeddings,
        optimal_mode=True
    )

    # Plot learning curve during meta-training
    plot_meta_train_losses(epoch_mean_support_losses,
                           epoch_mean_query_losses,
                           results_dir_name,
                           meta_learner.loss)

    # Save the weights that occur from meta-training the optimal model
    meta_learner.save_parameters(results_dir_name)


def meta_evaluate_optimal(opt_config, data_config, data_filenames, test_embeddings, results_dir_name):
    # TODO: Update docstring
    """Meta-evaluation process of the meta-learner within each task of the meta-test set.

    The meta-learner is initialized using the optimal initial base model weights and learned
    learning rates. Then it is evaluated on the query set of each test task after being trained
    on the support set of the examined task and a set of different metrics and plots related to
    the evaluation is created. This is done separately for each task of the test tasks set.

    Args:
        opt_config: A dictionary that contains the optimal hyperparameters for the model.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        data_filenames: A list of strings that contains the paths to the test tasks.
        results_dir_name: A string with the name of the directory the results will be saved.
    """

    args = set_model_args(opt_config)

    meta_learner = engine.build_meta_learner(args=args,
                                             data_config=data_config)

    # Create initial weights (weights_names attribute) learned during meta-training
    meta_learner.load_optimal_inner_loop_params(results_dir_name)

    # Meta-evaluation
    meta_learner.meta_test(data_filenames=data_filenames,
                           embeddings=test_embeddings,
                           optimal_mode=True,
                           results_dir_name=results_dir_name)


def main():
    # TODO: Add documentation and comments where needed

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', dest='train_dir')
    parser.add_argument('--test_dir', dest='test_dir')
    parser.add_argument('--train_embeddings_path', dest='train_embeddings_path')
    parser.add_argument('--test_embeddings_path', dest='test_embeddings_path')
    parser.add_argument('--config', '-C', dest='config_filepath')
    args = parser.parse_args()

    train_dir = args.train_dir
    test_dir = args.test_dir

    # Embeddings of train tasks
    train_embeddings_path = args.train_embeddings_path
    with open(train_embeddings_path, 'r', encoding='utf8') as stream:
        train_embeddings = json.load(stream)

    # Embeddings of test tasks
    test_embeddings_path = args.test_embeddings_path
    with open(test_embeddings_path, 'r', encoding='utf8') as stream:
        test_embeddings = json.load(stream)

    # Configurations
    config_filepath = args.config_filepath
    with open(config_filepath, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    # Paths to time series used for training
    train_filenames = []
    for root, _, files in os.walk(train_dir, topdown=False):
        for file in files:
            filename = root + file
            train_filenames.append(filename)

    # Paths to time series used for testing
    test_filenames = []
    for root, _, files in os.walk(test_dir, topdown=False):
        for file in files:
            filename = root + file
            test_filenames.append(filename)

    # Create results directory
    results_dir_name = './hsml/results/'
    if not os.path.exists(results_dir_name):
        os.makedirs(results_dir_name)

    # Set CUDA environment variables for reproducibility purposes
    # So that the LSTMs show deterministic behavior
    set_cuda_reproducibility()

    # Set random seeds for reproducibility purposes
    set_random_seeds(config['seed'])

    # Hyperparameter tuning
    n_trials = config['n_trials']
    ht_config = config['ht_config']
    data_config = config['data_config']

    opt_config = hyperparameter_tuning(n_trials,
                                       results_dir_name,
                                       ht_config,
                                       data_config,
                                       train_filenames,
                                       train_embeddings)

    # Meta-train optimal meta-learner
    meta_train_optimal(opt_config,
                       data_config,
                       train_filenames,
                       train_embeddings,
                       results_dir_name)

    # Load optimal meta-trained model and evaluate it on test tasks
    meta_evaluate_optimal(opt_config,
                          data_config,
                          test_filenames,
                          test_embeddings,
                          results_dir_name)

    # num_levels = 4
    # num_centers = [1, 4 ,2, 1]
    # embedding_size = 134
    # device = torch.device("cpu")
    # sigma = 2.0

    # network = model_builder.build_cluster_network(num_levels,
    #                                               num_centers,
    #                                               sigma,
    #                                               embedding_size,
    #                                               device)

    # h_i_final = network(task_embedding)

    # for key, value in network.named_parameters():
    #     print(key, value.shape)

    # args = {
    #     'train_epochs': 5,
    #     'task_batch_size': 1,
    #     'sample_batch_size': 1,
    #     'lstm_hidden_units': 16,
    #     'init_learning_rate': 1e-3,
    #     'meta_learning_rate': 1e-4,
    #     'eta_min': 1e-6,
    #     'num_inner_steps': 3,
    #     'second_order': False,
    #     'second_to_first_order_epoch': 5,
    #     'num_levels': 4,
    #     'num_centers': [1, 4 ,2, 1],
    #     'sigma': 2.0,
    #     'embedding_size': 134,
    #     'loss': 'MSE',
    #     'kappa': 2.0
    # }

    # data_config = {
    #     'day_measurements': 96,
    #     'week_num': 1,
    #     'pred_days': 1,
    #     'test_days': 7
    # }

    # meta_learner = engine.build_meta_learner(args=args,
    #                                              data_config=data_config)
    
    # names_weights_copy = meta_learner.get_inner_loop_params(state_dict=meta_learner.names_weights,
    #                                                         is_copy=True)
    
    # weights_mask = torch.rand(1, meta_learner.get_inner_loop_params_number())
    
    # names_weights_masked = meta_learner.apply_weights_mask(names_weights_copy, weights_mask)

    # print()
    # for name, key in names_weights_masked.items():
    #         print(name, key.shape, np.prod(key.shape))


if __name__ == "__main__":
    main()
