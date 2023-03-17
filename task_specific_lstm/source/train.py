"""Pipeline for training and evaluating a task-specific LSTM.

At first, the data of a single time series is loaded and transformed to a format appropriate
to be used by the LSTM. Then a hyperparameter optimization is performed using cross-validation
on the training set. After the optimal hyperparameters have been determined, a model is trained
based on these and is then evaluated on the test set. The optimal model is saved in a results
directory as well as its learning curve and other plots relative to its hyperparameter tuning.

Typical usage example:
python3 train.py --filepath "path/to/raw_time_series --config "path/to/config.yaml"
"""

#!/usr/bin/env python3

import argparse
import os
import yaml

from matplotlib import pyplot as plt
import optuna
from sklearn.model_selection import TimeSeriesSplit
import torch
from torch import nn

import data_setup
import engine
import model_builder
import utils


def objective(trial, x_train, y_train, ht_config):
    """Training loop of the hyperparameter tuning process.

    Firstly, the cross-validation schema is defined. Based on that, the model is trained for
    a specific set of hyperparameter values and the mean validation loss, which is the objective
    value to be minimized, is calcualted.

    Args:
        trial: An optuna trial object, handled solely by optuna.
        x_train: A torch.Tensor that contains the input features of the training set.
        y_train: A torch.Tensor that contains the output values of the training set.
        ht_config: A dictionary that defines the hyperparameter search space.
    Returns:
        A float that represents the mean validation error, which we want to minimize.
    """

    # Hyperparameter search space
    config = {
        'kfolds': ht_config['k_folds'],
        'batch_size': ht_config['batch_size'],
        'seed': ht_config['seed'],
        'scheduler_factor': ht_config['scheduler_factor'],
        'scheduler_patience': int(ht_config['scheduler_patience']),
        'scheduler_threshold': float(ht_config['scheduler_threshold']),
        'learning_rate': trial.suggest_float('learning_rate',
                                             float(ht_config['learning_rate']['lower_bound']),
                                             float(ht_config['learning_rate']['upper_bound']),
                                             log=ht_config['learning_rate']['log']),
        'epochs': trial.suggest_int('epochs',
                                    int(ht_config['epochs']['lower_bound']),
                                    int(ht_config['epochs']['upper_bound'])),
        'lstm_hidden_units': trial.suggest_int('lstm_hidden_units',
                                               int(ht_config['lstm_hidden_units']['lower_bound']),
                                               int(ht_config['lstm_hidden_units']['upper_bound']),
                                               log=ht_config['lstm_hidden_units']['log'])
    }

    # Set random seed for reproducibility purposes
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    device = utils.set_device()
    k_folds = config['kfolds']
    num_epochs = config['epochs']
    batch_size = config['batch_size']
    lstm_hidden_units = config['lstm_hidden_units']
    learning_rate = config['learning_rate']
    scheduler_factor = config['scheduler_factor']
    scheduler_patience = config['scheduler_patience']
    scheduler_threshold = config['scheduler_threshold']

    results = {}  # Contains the validation loss of each fold

    # Cross-validation splits
    tscv = TimeSeriesSplit(n_splits=k_folds, test_size=1)

    # Cross-validation loop
    for i, (train_idx, val_idx) in enumerate(tscv.split(x_train)):
        x_train_fold = x_train[train_idx]
        y_train_fold = y_train[train_idx]
        x_val_fold = x_train[val_idx]
        y_val_fold = y_train[val_idx]

        # Get the dataloaders, model, optimizer, scheduler and loss function
        train_dataloader, val_dataloader = data_setup.build_dataset(x_train_fold,
                                                                    y_train_fold,
                                                                    x_val_fold,
                                                                    y_val_fold,
                                                                    batch_size)
        network = model_builder.build_network(1, y_train.shape[1], lstm_hidden_units, device)
        optimizer = engine.build_optimizer(network, learning_rate)
        scheduler = engine.build_scheduler(optimizer,
                                           scheduler_factor,
                                           scheduler_patience,
                                           scheduler_threshold)
        loss_fn = nn.MSELoss()

        # Training loop
        for _ in range(num_epochs):
            train_loss = engine.train_epoch(network, train_dataloader, optimizer, loss_fn, device)
            scheduler.step(train_loss)

        # Model evaluation for the specific fold
        val_loss, _ = engine.evaluate(network, val_dataloader, loss_fn, device)
        results[i] = val_loss
        trial.report(val_loss, i)

    val_loss_sum = 0.0
    for _, value in results.items():
        val_loss_sum += value
    mean_val_loss = val_loss_sum / len(results.items())

    return mean_val_loss


def train_optimal(opt_config, x_train, y_train, x_test, y_test, results_dir_name):
    """Training loop of the optimal model.

    The model is trained based on the optimal hyperparameters determined from the previously
    done hyperparameter tuning. The train and test losses are calculated for each epoch and
    after training is done, the optimal model is saved.

    Args:
        opt_config: A dictionary that contains the optimal hyperparameter values.
        x_train: A torch.Tensor that contains the input features of the training set.
        y_train: A torch.Tensor that contains the output values of the training set.
        x_test: A torch.Tensor that contains the input features of the test set.
        y_test: A torch.Tensor that contains the output values of the test set.
        results_dir_name: A string with the name of the directory the results will be saved.
    Returns:
        Two dictionaries, one that contains the training loss and one that contains the test loss
        of each training epoch.
    """

    # Set random seed for reproducibility purposes
    torch.manual_seed(opt_config['seed'])
    torch.cuda.manual_seed(opt_config['seed'])

    device = utils.set_device()
    num_epochs = opt_config['epochs']
    batch_size = opt_config['batch_size']
    lstm_hidden_units = opt_config['lstm_hidden_units']
    learning_rate = opt_config['learning_rate']
    scheduler_factor = opt_config['scheduler_factor']
    scheduler_patience = opt_config['scheduler_patience']
    scheduler_threshold = opt_config['scheduler_threshold']

    # Get the dataloaders, model, optimizer, scheduler and loss function
    train_dataloader, test_dataloader = data_setup.build_dataset(x_train,
                                                      y_train,
                                                      x_test,
                                                      y_test,
                                                      batch_size)
    network = model_builder.build_network(1, y_train.shape[1], lstm_hidden_units, device)
    optimizer = engine.build_optimizer(network, learning_rate)
    scheduler = engine.build_scheduler(optimizer,
                                       scheduler_factor,
                                       scheduler_patience,
                                       scheduler_threshold)
    loss_fn = nn.MSELoss()

    # Train and test losses on each epoch
    train_losses = {}
    test_losses = {}

    # TODO: Maybe the states should be reset after each epoch?

    # Training loop
    for epoch in range(num_epochs):
        train_loss = engine.train_epoch(network, train_dataloader, optimizer, loss_fn, device)
        train_losses[epoch] = train_loss

        test_loss, _ = engine.evaluate(network, test_dataloader, loss_fn, device)
        test_losses[epoch] = test_loss

        scheduler.step(train_loss)

    # Prediction plots
    _, y_preds = engine.evaluate(network, test_dataloader, loss_fn, device)
    utils.plot_predictions(y_test, y_preds, results_dir_name)

    # Save optimal model
    utils.save_optimal_model(network, results_dir_name)

    return train_losses, test_losses


def hyperparameter_tuning(n_trials, results_dir_name, x_train, y_train, ht_config):
    """Perform hyperparameter tuning of the model on the given search space.

    This is done using the optuna library. Additionally to defining the best hyperparameters,
    a number of plots is created and saved that gives additional insights on the process.

    Args:
        n_trials: An integer that defines the total number of hyperparameter values' combinations
            to be tried out.
        results_dir_name: A string with the name of the directory the results will be saved.
        x_train: A torch.Tensor that contains the input features of the training set.
        y_train: A torch.Tensor that contains the output values of the training set.
        ht_config: A dictionary that defines the hyperparameter search space.
    Returns:
        A dictionary that contains the optimal hyperparameter values.
    """

    # Hyperparameter tuning process
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                direction='minimize')

    study.optimize(lambda trial: objective(trial, x_train, y_train, ht_config),
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

    opt_config = {
        'batch_size': 1,
        'seed': 42,
        'scheduler_factor': ht_config['scheduler_factor'],
        'scheduler_patience': int(ht_config['scheduler_patience']),
        'scheduler_threshold': float(ht_config['scheduler_threshold']),
        'learning_rate': best_trial['params_learning_rate'].values[0],
        'epochs': best_trial['params_epochs'].values[0],
        'lstm_hidden_units': best_trial['params_lstm_hidden_units'].values[0]
    }

    return opt_config


def main():
    """End-to-end logic to run the whole experiment.

    Initially, the given command line arguments are parsed and the time series to be examined is
    loaded and processed. Hyperparameter tuning is then performed to determine the optimal
    hyperparameters. Based on these, the final model is trained and then evaluated on the specific
    time series.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', '-F', dest='raw_series_filepath')
    parser.add_argument('--config', '-C', dest='config_filepath')
    args = parser.parse_args()

    raw_series_filepath =  args.raw_series_filepath
    timeseries_code = raw_series_filepath[-7:-4]

    config_filepath = args.config_filepath
    with open(config_filepath, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    # Create results directory
    results_dir_name = './task_specific_lstm/results/' + timeseries_code + '/'
    if not os.path.exists(results_dir_name):
        os.makedirs(results_dir_name)

    # Data loading and processing
    data = data_setup.load_timeseries(raw_series_filepath)
    x_train, y_train, x_test, y_test = data_setup.split_train_test(data,
        config['data_split_constants'])

    # Hyperparameter tuning
    n_trials = config['n_trials']
    ht_config = config['ht_config']

    opt_config = hyperparameter_tuning(n_trials, results_dir_name, x_train, y_train, ht_config)

    # Optimal model training and evaluation
    train_losses, test_losses = train_optimal(opt_config,
                                              x_train,
                                              y_train,
                                              x_test,
                                              y_test,
                                              results_dir_name)

    # Additional visualizations
    utils.plot_learning_curve(train_losses, test_losses, results_dir_name)


if __name__ == "__main__":
    main()
