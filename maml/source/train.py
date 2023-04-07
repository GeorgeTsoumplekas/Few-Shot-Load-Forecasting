#!/usr/bin/env python3

import argparse
import os
import warnings
import yaml

from matplotlib import pyplot as plt
import optuna
from sklearn.model_selection import KFold
import torch

import engine
from utils import plot_meta_train_losses


# Suppress warning that occurs due to accessing a non-leaf tensor
warnings.filterwarnings("ignore")


def objective(trial, ht_config, data_config, data_filenames):

    # Hyperparameter search space
    config = {
        'kfolds': ht_config['k_folds'],
        'seed': ht_config['seed'],
        'task_batch_size': ht_config['task_batch_size'],
        'sample_batch_size': ht_config['sample_batch_size'],
        'num_inner_steps': ht_config['num_inner_steps'],
        'eta_min': float(ht_config['eta_min']),
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

    # Set random seed for reproducibility purposes
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    mean_fold_val_losses = {}  # Contains the mean validation loss of each fold

    k_folds = config['kfolds']
    args = {
        'train_epochs': config['train_epochs'],
        'task_batch_size': config['task_batch_size'],
        'sample_batch_size': config['sample_batch_size'],
        'lstm_hidden_units': config['lstm_hidden_units'],
        'init_learning_rate': config['init_learning_rate'],
        'meta_learning_rate': config['meta_learning_rate'],
        'eta_min': config['eta_min'],
        'num_inner_steps': config['num_inner_steps'],
        'second_order': config['second_order'],
        'second_to_first_order_epoch': config['second_to_first_order_epoch']
    }

    # Cross-validation splits - they should be random since we are splitting the tasks
    # (non-sequential) and not a single timeseries (sequential)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_idx,val_idx) in enumerate(kfold.split(data_filenames)):
        print(f"Fold {fold+1}/{k_folds}\n")
        meta_learner = engine.build_meta_learner(args=args,
                                                 data_config=data_config)
        
        # Tasks used for training within this fold
        train_filenames = [data_filenames[idx] for idx in train_idx]

        # Tasks used for validation within this fold
        val_filenames = [data_filenames[idx] for idx in val_idx]

        # Meta-Train
        print("Meta-Training started..")
        meta_learner.hp_tuning_meta_train(train_filenames)
        print("Meta-Training finshed.\n")

        # Meta-Evaluation
        print("Meta-Validation started..")
        mean_fold_val_loss = meta_learner.hp_tuning_meta_evaluate(val_filenames)
        print("Meta-Validation finshed.\n")

        mean_fold_val_losses[fold] = mean_fold_val_loss

    val_loss_sum = 0.0
    for _, value in mean_fold_val_losses.items():
        val_loss_sum += value
    mean_val_loss = val_loss_sum / len(mean_fold_val_losses.items())

    return mean_val_loss


def hyperparameter_tuning(n_trials, results_dir_name, ht_config, data_config, data_filenames):

    # Hyperparameter tuning process
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                direction='minimize')
    study.optimize(lambda trial: objective(trial, ht_config, data_config, data_filenames),
                   n_trials=n_trials)
    
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

    print(f"Best trial: {best_trial}")

    opt_config = {
        'seed': ht_config['seed'],
        'task_batch_size': ht_config['task_batch_size'],
        'sample_batch_size': ht_config['sample_batch_size'],
        'num_inner_steps': ht_config['num_inner_steps'],
        'eta_min': ht_config['eta_min'],
        'train_epochs': best_trial['params_train_epochs'].values[0],
        'lstm_hidden_units': best_trial['params_lstm_hidden_units'].values[0],
        'init_learning_rate': best_trial['params_init_learning_rate'].values[0],
        'meta_learning_rate': best_trial['params_meta_learning_rate'].values[0],
        'second_order': ht_config['second_order'],
        'second_to_first_order_epoch': ht_config['second_to_first_order_epoch']
    }

    return opt_config


def meta_train_optimal(opt_config, data_config, data_filenames, results_dir_name):

    # Set random seed for reproducibility purposes
    torch.manual_seed(opt_config['seed'])
    torch.cuda.manual_seed(opt_config['seed'])

    args = {
        'train_epochs': opt_config['train_epochs'],
        'task_batch_size': opt_config['task_batch_size'],
        'sample_batch_size': opt_config['sample_batch_size'],
        'lstm_hidden_units': opt_config['lstm_hidden_units'],
        'init_learning_rate': opt_config['init_learning_rate'],
        'meta_learning_rate': opt_config['meta_learning_rate'],
        'eta_min': float(opt_config['eta_min']),
        'num_inner_steps': opt_config['num_inner_steps'],
        'second_order': opt_config['second_order'],
        'second_to_first_order_epoch': opt_config['second_to_first_order_epoch']
    }

    meta_learner = engine.build_meta_learner(args=args,
                                             data_config=data_config)

    train_losses, epoch_mean_support_losses, epoch_mean_query_losses = meta_learner.meta_train(
        data_filenames)

    plot_meta_train_losses(epoch_mean_support_losses,
                           epoch_mean_query_losses,
                           results_dir_name)

    # Save the weights that occur from meta-training the optimal model
    meta_learner.save_parameters(results_dir_name)

    return train_losses


def meta_evaluate_optimal(opt_config, data_config, data_filenames, results_dir_name):

    # Set random seed for reproducibility purposes
    torch.manual_seed(opt_config['seed'])
    torch.cuda.manual_seed(opt_config['seed'])

    args = {
        'train_epochs': opt_config['train_epochs'],
        'task_batch_size': opt_config['task_batch_size'],
        'sample_batch_size': opt_config['sample_batch_size'],
        'lstm_hidden_units': opt_config['lstm_hidden_units'],
        'init_learning_rate': opt_config['init_learning_rate'],
        'meta_learning_rate': opt_config['meta_learning_rate'],
        'eta_min': float(opt_config['eta_min']),
        'num_inner_steps': opt_config['num_inner_steps'],
        'second_order': opt_config['second_order'],
        'second_to_first_order_epoch': opt_config['second_to_first_order_epoch']
    }

    meta_learner = engine.build_meta_learner(args=args,
                                             data_config=data_config)

    # Create initial weights (weights_names attribute)
    meta_learner.load_optimal_inner_loop_params(results_dir_name)

    # Meta-evaluation
    meta_learner.meta_test_optimal(data_filenames, results_dir_name)


# TODO: change appropriately
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', dest='train_dir')
    parser.add_argument('--test_dir', dest='test_dir')
    parser.add_argument('--config', '-C', dest='config_filepath')
    args = parser.parse_args()

    train_dir = args.train_dir
    test_dir = args.test_dir
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

    opt_config = {
        'seed': 42,
        'task_batch_size': 1,
        'sample_batch_size': 1,
        'num_inner_steps': 2,
        'eta_min': float(1e-6),
        'train_epochs': 2,
        'lstm_hidden_units': 16,
        'init_learning_rate': 0.0016075883117690475,
        'meta_learning_rate': 0.0005439505246783044,
        'second_order': True,
        'second_to_first_order_epoch': 1
    }

    # Create results directory
    results_dir_name = './maml/results/'
    if not os.path.exists(results_dir_name):
        os.makedirs(results_dir_name)

    # Hyperparameter tuning
    n_trials = config['n_trials']
    ht_config = config['ht_config']
    data_config = config['data_config']

    # opt_config = hyperparameter_tuning(n_trials,
    #                                    results_dir_name,
    #                                    ht_config,
    #                                    data_config,
    #                                    train_filenames)

    print(f"Opt config: {opt_config}")

    # Train optimal model and plot train loss
    _ = meta_train_optimal(opt_config,
                                      data_config,
                                      train_filenames,
                                      results_dir_name)

    # Load optimal meta-trained model and evaluate it on test tasks
    meta_evaluate_optimal(opt_config,
                          data_config,
                          test_filenames,
                          results_dir_name)


if __name__ == "__main__":
    main()


# TODO: Na 3anatsekarw gia reproducibility, random seeds etc
# TODO: Na 3anadokimasw me second order kai gpu monitoring