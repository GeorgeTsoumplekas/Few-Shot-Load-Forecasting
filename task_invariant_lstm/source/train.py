"""Pipeline for training and evaluating a task-invariant LSTM.

At first, the time series used for training and testing are defined based on the given
directories. Then a hyperparameter optimization is performed using cross-validation
on the tasks of the training set. After the optimal hyperparameters have been determined,
a model is trained based on these. Then, for each time series in the tasks test set, the
model is fine-tuned on some of its data and evaluated on the rest of it. The optimal models are
saved as well as the corresponding learning curves and other plots relative to its
hyperparameter tuning.

Typical usage example:
python3 train.py --train_dir "path/to/train_dir" \
                 --test_dir "path/to/train_dir" \
                 --config "path/to/config.yaml"
"""

#!/usr/bin/env python3

import argparse
import os
import yaml

from matplotlib import pyplot as plt
import optuna
import pandas as pd
from sklearn.model_selection import KFold
import torch

import data_setup
import engine
import losses
import model_builder
import utils


def objective(trial, ht_config, data_config, data_filenames):
    """Training process of the hyperparameter tuning process.

    First, the cross-validation schema is defined. Based on that, the model is trained for
    a specific set of hyperparameter values and the mean validation loss, which is the objective
    value to be minimized, is calcualted.

    Args:
        trial: An optuna trial object, handled solely by optuna.
        ht_config: A dictionary that defines the hyperparameter search space.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        data_filenames: A list of strings that contains the paths to the training tasks.
    Returns:
        A float that represents the mean validation error, which we want to minimize.
    """

    # Hyperparameter search space
    config = {
        'kfolds': ht_config['k_folds'],
        'task_batch_size': ht_config['task_batch_size'],
        'sample_batch_size': ht_config['sample_batch_size'],
        'seed': ht_config['seed'],
        'loss': ht_config['loss'],
        'kappa': trial.suggest_float('kappa',
                                     ht_config['kappa']['lower_bound'],
                                     ht_config['kappa']['upper_bound']),
        'learning_rate': trial.suggest_float('learning_rate',
                                             float(ht_config['learning_rate']['lower_bound']),
                                             float(ht_config['learning_rate']['upper_bound']),
                                             log=ht_config['learning_rate']['log']),
        'train_epochs': trial.suggest_int('train_epochs',
                                          int(ht_config['train_epochs']['lower_bound']),
                                          int(ht_config['train_epochs']['upper_bound'])),
        'finetune_epochs': trial.suggest_int('finetune_epochs',
                                          int(ht_config['finetune_epochs']['lower_bound']),
                                          int(ht_config['finetune_epochs']['upper_bound'])),
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
    num_train_epochs = config['train_epochs']
    num_finetune_epochs = config['finetune_epochs']
    task_batch_size = config['task_batch_size']
    sample_batch_size = config['sample_batch_size']
    lstm_hidden_units = config['lstm_hidden_units']
    learning_rate = config['learning_rate']
    output_shape = data_config['pred_days']*data_config['day_measurements']
    loss = config['loss']
    kappa = config['kappa']

    results = {}  # Contains the validation loss of each fold

    # Cross-validation splits - they should be random since we are splitting the tasks
    # (non-sequential) and not a single timeseries (sequential)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Cross-validation loop
    for fold, (train_idx,val_idx) in enumerate(kfold.split(data_filenames)):

        # Tasks used for training within this fold
        train_filenames = [data_filenames[idx] for idx in train_idx]

        # Tasks used for validation within this fold
        val_filenames = [data_filenames[idx] for idx in val_idx]

        # Get the tasks dataloaders, model, optimizer and loss function
        train_tasks_dataloader = data_setup.build_tasks_set(train_filenames,
                                                            data_config,
                                                            task_batch_size,
                                                            True)
        val_tasks_dataloader = data_setup.build_tasks_set(val_filenames,
                                                          data_config,
                                                          task_batch_size,
                                                          False)

        network = model_builder.build_network(1, output_shape, lstm_hidden_units, device)
        optimizer = engine.build_optimizer(network, learning_rate)
        loss_fn = losses.get_loss(loss, kappa, device)

        # Train model using the train tasks
        for _ in range(num_train_epochs):
            _ = engine.train_epoch(network,
                                   train_tasks_dataloader,
                                   loss_fn,
                                   optimizer,
                                   device,
                                   sample_batch_size,
                                   data_config)

        # Fine-tuning and evaluation in each validation task
        val_loss = 0.0
        for val_task_data, _ in val_tasks_dataloader:
            # Each fine-tuned network is derived from the same trained model
            # but is different for each validation task, since it's fine-tuned in different data.
            finetuned_network = model_builder.build_network(input_shape=1,
                                                            output_shape=output_shape,
                                                            hidden_units=lstm_hidden_units,
                                                            device=device)
            finetuned_network.load_state_dict(network.state_dict())
            finetuned_network.to(device)

            # Define an optimizer for the new model
            optimizer = engine.build_optimizer(finetuned_network, learning_rate)

            # Create the dataloaders
            val_train_dataloader, val_test_dataloader, _ = data_setup.build_test_task(
                val_task_data,
                sample_batch_size,
                data_config
            )

            # Fine-tune model with task's training data
            for _ in range(num_finetune_epochs):
                _ = engine.train_step(finetuned_network,
                                      val_train_dataloader,
                                      loss_fn,
                                      optimizer,
                                      device)

            #  Evaluate on task's test data
            val_task_loss, _ = engine.evaluate(finetuned_network,
                                               val_test_dataloader,
                                               loss_fn,
                                               device)
            val_loss += val_task_loss

        # Total validation loss for the specific fold
        val_loss /= len(val_tasks_dataloader)
        results[fold] = val_loss

    val_loss_sum = 0.0
    for _, value in results.items():
        val_loss_sum += value
    mean_val_loss = val_loss_sum / len(results.items())

    return mean_val_loss


def hyperparameter_tuning(n_trials, results_dir_name, ht_config, data_config, data_filenames):
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

    opt_config = {
        'task_batch_size': ht_config['task_batch_size'],
        'sample_batch_size': ht_config['sample_batch_size'],
        'seed': ht_config['seed'],
        'loss': ht_config['loss'],
        'kappa': best_trial['params_kappa'].values[0],
        'learning_rate': best_trial['params_learning_rate'].values[0],
        'train_epochs': best_trial['params_train_epochs'].values[0],
        'finetune_epochs': best_trial['params_finetune_epochs'].values[0],
        'lstm_hidden_units': best_trial['params_lstm_hidden_units'].values[0]
    }

    return opt_config


def train_optimal(opt_config, data_config, data_filenames, results_dir_name):
    """Training process of the optimal model.

    The model is trained based on the optimal hyperparameters determined from the previously
    done hyperparameter tuning. The train losses are calculated for each epoch and
    after training is done, the optimal model is saved.

    Args:
        opt_config: A dictionary that contains the optimal hyperparameter values.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        data_filenames: A list of strings that contains the paths to the training tasks.
        results_dir_name: A string with the name of the directory the results will be saved.
    """

    # Set random seed for reproducibility purposes
    torch.manual_seed(opt_config['seed'])
    torch.cuda.manual_seed(opt_config['seed'])

    device = utils.set_device()
    num_epochs = opt_config['train_epochs']
    task_batch_size = opt_config['task_batch_size']
    sample_batch_size = opt_config['sample_batch_size']
    lstm_hidden_units = opt_config['lstm_hidden_units']
    learning_rate = opt_config['learning_rate']
    output_shape = data_config['pred_days']*data_config['day_measurements']
    loss = opt_config['loss']
    kappa = opt_config['kappa']

    # Create dataloader for the training tasks
    train_tasks_dataloader = data_setup.build_tasks_set(data_filenames,
                                                        data_config,
                                                        task_batch_size,
                                                        True)

    # Get the model, optimizer and loss function
    network = model_builder.build_network(input_shape=1,
                                          output_shape=output_shape,
                                          hidden_units=lstm_hidden_units,
                                          device=device)
    optimizer = engine.build_optimizer(network, learning_rate)
    loss_fn = losses.get_loss(loss, kappa, device)

    train_losses = []

    # Model training using training tasks
    for _ in range(num_epochs):
        train_loss = engine.train_epoch(network,
                                        train_tasks_dataloader,
                                        loss_fn,
                                        optimizer,
                                        device,
                                        sample_batch_size,
                                        data_config)
        train_losses.append(train_loss)

    utils.plot_train_loss(train_losses, results_dir_name, loss)

    # Save trained model
    utils.save_model(network, results_dir_name)


def finetune_optimal(opt_config, data_config, test_filenames, results_dir_name):
    """Fine-tuning process of optimal model within each test task.

    The optimal model that occured after training using the optimal hyperparameters is loaded
    and then fine-tuned using the task's train data. The train and test losses are calculated
    for each fine-tuning epoch and after fine-tuning is done, the optimal model as well as a
    number of different logs and metrics are saved. This is done separately for each task of
    the test tasks set.

    Args:
        opt_config: A dictionary that contains the optimal hyperparameters for the model.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        test_filenames: A list of strings that contains the paths to the test tasks.
        results_dir_name: A string with the name of the directory the results will be saved.
    """

    # Set random seed for reproducibility purposes
    torch.manual_seed(opt_config['seed'])
    torch.cuda.manual_seed(opt_config['seed'])

    device = utils.set_device()
    num_epochs = opt_config['finetune_epochs']
    task_batch_size = opt_config['task_batch_size']
    sample_batch_size = opt_config['sample_batch_size']
    lstm_hidden_units = opt_config['lstm_hidden_units']
    learning_rate = opt_config['learning_rate']
    output_shape = data_config['pred_days']*data_config['day_measurements']
    model_save_path = results_dir_name + 'optimal_trained_model.pth'
    loss = opt_config['loss']
    kappa = opt_config['kappa']

    # Get the dataloader for the set of test tasks
    test_tasks_dataloader = data_setup.build_tasks_set(test_filenames,
                                                       data_config,
                                                       task_batch_size,
                                                       False)

    # Fine-tuning process for each test task
    for test_task_data, test_timeseries_code in test_tasks_dataloader:
        # Each task has an assigned directory to save relative results
        target_dir_name = results_dir_name + test_timeseries_code[0] + '/'
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        # Each fine-tuned network is derived from the same trained model
        # but is different for each validation task, since it's fine-tuned in different data.
        network = model_builder.build_network(1, output_shape, lstm_hidden_units, device)
        network.load_state_dict(torch.load(f=model_save_path))

        # Define an optimizer for the new model
        optimizer = engine.build_optimizer(network, learning_rate)

        loss_fn = losses.get_loss(loss, kappa, device)

        finetune_losses = []
        test_losses = []

        # Get the train and test dataloaders
        finetune_dataloader, test_dataloader, y_task_test_raw = data_setup.build_test_task(
            test_task_data,
            sample_batch_size,
            data_config,
            target_dir_name
        )

        # Fine-tune model with task's training data and evaluate with task's test data
        for _ in range(num_epochs):
            finetune_loss = engine.train_step(network,
                                              finetune_dataloader,
                                              loss_fn,optimizer,
                                              device)
            finetune_loss /= len(finetune_dataloader)
            finetune_losses.append(finetune_loss)

            test_loss, _ = engine.evaluate(network,
                                           test_dataloader,
                                           loss_fn,
                                           device)
            test_losses.append(test_loss)

        # Learning curve for each fine-tuned model
        utils.plot_learning_curve(finetune_losses, test_losses, target_dir_name, loss)

        # Prediction plots
        y_test = data_setup.get_task_test_set(test_dataloader)
        _, y_preds = engine.evaluate(network, test_dataloader, loss_fn, device)
        utils.plot_predictions(y_test, y_preds, target_dir_name)

        # MAPE calculation on original scale
        y_preds_raw = data_setup.unstandardized_preds(torch.tensor(y_preds), target_dir_name)
        mape = losses.MAPE(y_preds_raw, y_task_test_raw)

        task_log = pd.DataFrame({'loss_type': loss,
                                 'loss_value': [test_losses[num_epochs-1]],
                                 'length': [len(finetune_dataloader)*7],
                                 'MAPE': [mape]})

        # Save logs as .csv file
        utils.save_validation_logs(task_log, target_dir_name)


def main():
    """End-to-end logic to run the whole experiment.

    Initially, the given command line arguments are parsed and the training and test tasks as
    well as the experiment's configuration are defined. Hyperparameter tuning is then performed
    to determine the optimal hyperparameters. Based on these, the final model is fine-tuned for each
    test task and then evaluated.
    """

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

    # Create results directory
    results_dir_name = './task_invariant_lstm/results/'
    if not os.path.exists(results_dir_name):
        os.makedirs(results_dir_name)

    # Set CUDA environment variables for reproducibility purposes
    # So that the LSTMs show deterministic behavior
    utils.set_cuda_reproducibility()

    # Hyperparameter tuning
    n_trials = config['n_trials']
    ht_config = config['ht_config']
    data_config = config['data_config']

    opt_config = hyperparameter_tuning(n_trials,
                                       results_dir_name,
                                       ht_config,
                                       data_config,
                                       train_filenames)

    # Train optimal model
    train_optimal(opt_config, data_config, train_filenames, results_dir_name)

    # Fine-tune and evaluate optimal model on each test time series
    finetune_optimal(opt_config, data_config, test_filenames, results_dir_name)


if __name__ == "__main__":
    main()
