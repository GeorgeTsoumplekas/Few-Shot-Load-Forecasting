"""Pipeline for training/evaluating a Recurrent AutoEncoder and creating the task embeddings.

At first, the time series used for training and testing are defined based on the given
directories. Then a hyperparameter optimization is performed using cross-validation
on the tasks of the training set. After the optimal hyperparameters have been determined,
a model is trained and then evaluated based on these. Finally, the embeddings of the tasks are
created and saved. The optimal model is saved as well as its learning curve and other plots relative
to its hyperparameter tuning and visualizeng the task embeddings.

Typical usage example:
python3 train.py --train_dir "path/to/train_dir" \
                 --test_dir "path/to/train_dir" \
                 --config "path/to/config.yaml"
"""

import argparse
import json
import os
from time import time
import yaml

import pandas as pd
from matplotlib import pyplot as plt
import optuna
from sklearn.model_selection import KFold, train_test_split
import torch

from data_setup import build_tasks_set, build_task
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
        trial: An optuna trial object, handled internally by optuna.
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
        'train_epochs': ht_config['train_epochs'],
        'loss': ht_config['loss'],
        'kappa': trial.suggest_float('kappa',
                                     ht_config['kappa']['lower_bound'],
                                     ht_config['kappa']['upper_bound']),
        'learning_rate': trial.suggest_float('learning_rate',
                                             float(ht_config['learning_rate']['lower_bound']),
                                             float(ht_config['learning_rate']['upper_bound']),
                                             log=ht_config['learning_rate']['log']),
        'embedding_ratio': trial.suggest_float('embedding_ratio',
                                                ht_config['embedding_ratio']['lower_bound'],
                                                ht_config['embedding_ratio']['upper_bound'],
                                                step=ht_config['embedding_ratio']['step'])
    }

    mean_fold_val_losses = {}  # Contains the mean validation loss of each fold

    device = utils.set_device()
    k_folds = config['kfolds']
    num_epochs = config['train_epochs']
    task_batch_size = config['task_batch_size']
    sample_batch_size = config['sample_batch_size']
    output_shape = data_config['week_num']*7*data_config['day_measurements']
    lstm_hidden_units = round(config['embedding_ratio']*output_shape)
    learning_rate = config['learning_rate']
    loss = config['loss']
    kappa = config['kappa']

    # Cross-validation splits - they should be random since we are splitting the tasks
    # (non-sequential) and not a single timeseries (sequential)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_idx,val_idx) in enumerate(kfold.split(data_filenames)):
        print(f"Fold {fold+1}|{k_folds}")

        fold_start = time()

        # Tasks used for training within this fold
        train_filenames = [data_filenames[idx] for idx in train_idx]

        # Tasks used for validation within this fold
        val_filenames = [data_filenames[idx] for idx in val_idx]

        # Get the tasks dataloaders, model, optimizer and loss function
        train_tasks_dataloader = build_tasks_set(train_filenames,
                                                 data_config,
                                                 task_batch_size)
        val_tasks_dataloader = build_tasks_set(val_filenames,
                                               data_config,
                                               task_batch_size)
        network = model_builder.build_network(1, output_shape, lstm_hidden_units, device)
        optimizer = engine.build_optimizer(network, learning_rate)
        loss_fn = losses.get_loss(loss, kappa, device)

        # Train model using the train tasks
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}|{num_epochs}")
            epoch_start = time()
            _ = engine.train_epoch(network,
                                   train_tasks_dataloader,
                                   loss_fn,
                                   optimizer,
                                   device,
                                   sample_batch_size,
                                   data_config)
            epoch_end = time()
            print(f"Epoch elapsed time: {(epoch_end-epoch_start):.3f}s")

        # Evaluate on validation tasks
        val_loss = engine.evaluate(network,
                                   val_tasks_dataloader,
                                   sample_batch_size,
                                   data_config,
                                   loss_fn,
                                   device)

        # Total validation loss for the specific fold
        mean_fold_val_losses[fold] = val_loss

        fold_end = time()
        print(f"Fold elapsed time: {(fold_end-fold_start):.3f}s")

    val_loss_sum = 0.0
    for _, value in mean_fold_val_losses.items():
        val_loss_sum += value
    mean_val_loss = val_loss_sum / len(mean_fold_val_losses.items())

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
        'train_epochs': ht_config['train_epochs'],
        'max_epochs': ht_config['max_epochs'],
        'patience': ht_config['patience'],
        'min_delta': ht_config['min_delta'],
        'loss': ht_config['loss'],
        'kappa': best_trial['params_kappa'].values[0],
        'learning_rate': best_trial['params_learning_rate'].values[0],
        'embedding_ratio': best_trial['params_embedding_ratio'].values[0]
    }

    return opt_config


def train_optimal(opt_config, data_config, data_filenames, results_dir_name):
    """Training process of the optimal model.

    The model is trained based on the optimal hyperparameters determined from the previously
    done hyperparameter tuning. The train and validation losses are calculated for each epoch and
    after training is done, the optimal model is saved. In order to prevent overfitting, early
    stopping is applied.

    Args:
        opt_config: A dictionary that contains the optimal hyperparameter values.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        data_filenames: A list of strings that contains the paths to the training tasks.
        results_dir_name: A string with the name of the directory the results will be saved.
    Returns:
        A list that contains the training loss of each training epoch and a list that contains
        the validation loss of each training epoch.
    """

    device = utils.set_device()
    task_batch_size = opt_config['task_batch_size']
    sample_batch_size = opt_config['sample_batch_size']
    num_epochs = opt_config['max_epochs']
    learning_rate = opt_config['learning_rate']
    output_shape = data_config['week_num']*7*data_config['day_measurements']
    lstm_hidden_units = round(opt_config['embedding_ratio']*output_shape)
    patience =  opt_config['patience']
    min_delta = opt_config['min_delta']
    loss = opt_config['loss']
    kappa = opt_config['kappa']

    # Split meta-train set to meta-train and meta-validation sets
    train_filenames, val_filenames = train_test_split(data_filenames,
                                                      test_size=0.2,
                                                      random_state=42)

    # Create dataloader for the training tasks
    train_tasks_dataloader = build_tasks_set(train_filenames,
                                             data_config,
                                             task_batch_size)

    # Create dataloader for the validation tasks
    val_tasks_dataloader = build_tasks_set(val_filenames,
                                           data_config,
                                           task_batch_size)

    # Get the model, optimizer, early stopper and loss function
    network = model_builder.build_network(input_shape=1,
                                          output_shape=output_shape,
                                          hidden_units=lstm_hidden_units,
                                          device=device)
    optimizer = engine.build_optimizer(network, learning_rate)
    early_stopper = engine.build_early_stopper(network, patience, min_delta)
    loss_fn = losses.get_loss(loss, kappa, device)

    train_losses = []
    val_losses = []

    # Model training using training tasks / evaluation using validation tasks
    for _ in range(num_epochs):
        epoch_start = time()

        # Train model
        train_loss = engine.train_epoch(network,
                                   train_tasks_dataloader,
                                   loss_fn,
                                   optimizer,
                                   device,
                                   sample_batch_size,
                                   data_config)
        train_losses.append(train_loss)

        # Evaluate model
        val_loss = engine.evaluate(network,
                                   val_tasks_dataloader,
                                   sample_batch_size,
                                   data_config,
                                   loss_fn,
                                   device)
        val_losses.append(val_loss)

        epoch_end = time()
        print(f"Epoch elapsed time: {(epoch_end-epoch_start):.3f}s")

        # Check if early stopping needs to be applied
        if early_stopper.early_stop(val_loss, network):
            break

    # Save best model
    utils.save_model(network_state_dict=early_stopper.get_best_model(),
                     results_dir_name=results_dir_name)

    return train_losses, val_losses


def evaluate_optimal(opt_config,
                     data_config,
                     data_filenames,
                     task_set_type,
                     results_dir_name):
    # TODO: Update docstring
    """Evaluate the trained optimal model on the given set of tasks.

    The optimal model that occured after training using the optimal hyperparameters is loaded
    and then evaluated on each given task. The total test loss is calculated while for each task
    prediction plots based on the reconstructions are created and saved.

    Args:
        opt_config: A dictionary that contains the optimal hyperparameters for the model.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        data_filenames: A list of strings that contains the paths to the tasks.
        task_set_type: A string that defines the type of the given tasks set (meta-train
            or meta-test).
        results_dir_name: A string with the name of the directory the results will be saved.
    Return:
        A float that is the mean reconstruction loss of the test tasks.
    """

    device = utils.set_device()
    task_batch_size = opt_config['task_batch_size']
    sample_batch_size = opt_config['sample_batch_size']
    output_shape = data_config['week_num']*7*data_config['day_measurements']
    lstm_hidden_units = round(opt_config['embedding_ratio']*output_shape)
    loss = opt_config['loss']
    kappa = opt_config['kappa']
    model_save_path = results_dir_name + 'optimal_trained_model.pth'

    # Create subfolder for tasks' results
    target_dir = results_dir_name + task_set_type + '_tasks/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Create tasks set dataloader
    tasks_dataloader = build_tasks_set(data_filenames,
                                       data_config,
                                       task_batch_size)

    # Load optimal model and define loss function
    network = model_builder.build_network(input_shape=1,
                                          output_shape=output_shape,
                                          hidden_units=lstm_hidden_units,
                                          device=device)
    network.load_state_dict(torch.load(f=model_save_path))
    loss_fn = losses.get_loss(loss, kappa, device)

    # DataFrame that holds the info about the inference in each task
    task_logs = pd.DataFrame(columns=['timeseries_code', 'loss_type', 'loss_value', 'length'])

    total_loss = 0.0

    # Evaluation on each task
    for task_data, timeseries_code in tasks_dataloader:

        # Create subdirectory for the specific task
        task_specific_dir = target_dir + timeseries_code[0] + '/'
        if not os.path.exists(task_specific_dir):
            os.makedirs(task_specific_dir)

        support_set_dataloader, _ = build_task(task_data, sample_batch_size, data_config)

        task_loss, y_pred, y_true = engine.evaluate_task(network,
                                                         support_set_dataloader,
                                                         loss_fn,
                                                         device)
        total_loss += task_loss

        # Logs for the task
        task_log = pd.DataFrame({'timeseries_code': timeseries_code,
                                 'loss_type': loss,
                                 'loss_value': [task_loss],
                                 'length': [7*data_config['week_num']*y_pred.shape[0]]})
        task_logs = pd.concat([task_logs, task_log], ignore_index=True)

        # Create a pred plot for each task
        utils.plot_predictions(torch.flatten(y_true).tolist(),
                               torch.flatten(y_pred).tolist(),
                               task_specific_dir)

        # Create distribution plots for each task
        utils.plot_distributions(torch.flatten(y_true),
                                 torch.flatten(y_pred),
                                 task_specific_dir)

    # Save logs as .csv file
    utils.save_validation_logs(task_logs, target_dir)

    total_loss /= len(tasks_dataloader)
    return total_loss


def embed_task_set(opt_config, data_config, data_filenames, results_dir_name):
    """Generates the embeddings for a specific tasks set.

    The optimal trained model is loaded and then the embedding for each task in the set of tasks
    is generated.

    Args:
        opt_config: A dictionary that contains the optimal hyperparameters for the model.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        data_filenames: A list of strings that contains the paths to the tasks.
        results_dir_name: A string with the name of the directory the results will be saved.
    Returns:
        A dictionary that contains the embedding of each task in tasks set.
    """

    device = utils.set_device()
    task_batch_size = opt_config['task_batch_size']
    sample_batch_size = opt_config['sample_batch_size']
    output_shape = data_config['week_num']*7*data_config['day_measurements']
    lstm_hidden_units = round(opt_config['embedding_ratio']*output_shape)
    model_save_path = results_dir_name + 'optimal_trained_model.pth'

    # Contains the embedding of each task in the examined set of tasks
    all_embeddings = {}

    # Load optimal model
    network = model_builder.build_network(input_shape=1,
                                          output_shape=output_shape,
                                          hidden_units=lstm_hidden_units,
                                          device=device)
    network.load_state_dict(torch.load(f=model_save_path))

    # Create tasks set dataloader
    tasks_dataloader = build_tasks_set(data_filenames,
                                       data_config,
                                       task_batch_size)

    for task_data, timeseries_code in tasks_dataloader:
        train_dataloader, _ = build_task(task_data, sample_batch_size, data_config)

        # Embed task
        task_embedding = engine.embed_task(network, train_dataloader, device)

        # The task embedding is trasformed to a list before being saved
        all_embeddings[timeseries_code[0]] = task_embedding.tolist()

    return all_embeddings


def main():
    """End-to-end logic to run the whole experiment.

    Initially, the given command line arguments are parsed and the training and test tasks as
    well as the experiment's configuration are defined. Hyperparameter tuning is then performed
    to determine the optimal hyperparameters and based on these, the optimal model is trained and
    then evaluated. Finally, the embeddings of both train and test tasks are generated, stored and
    visualized.
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
    results_dir_name = './task_embedding/results/'
    if not os.path.exists(results_dir_name):
        os.makedirs(results_dir_name)

    # Set CUDA environment variables for reproducibility purposes
    # So that the LSTMs show deterministic behavior
    utils.set_cuda_reproducibility()

    # Set random seeds for reproducibility purposes
    utils.set_random_seeds(config['seed'])

    # Hyperparameter tuning
    n_trials = config['n_trials']
    ht_config = config['ht_config']
    data_config = config['data_config']

    opt_config = hyperparameter_tuning(n_trials,
                                       results_dir_name,
                                       ht_config,
                                       data_config,
                                       train_filenames)

    # Train optimal task-embedding model
    train_losses, val_losses = train_optimal(opt_config,
                                             data_config,
                                             train_filenames,
                                             results_dir_name)
    utils.plot_learning_curve(train_losses, val_losses, results_dir_name, opt_config['loss'])

    # Load optimal task-embedding model and evaluate it on train tasks
    mean_train_loss = evaluate_optimal(opt_config,
                                        data_config,
                                        train_filenames,
                                        'train',
                                        results_dir_name)
    print(f"Mean meta-train set loss: {mean_train_loss}")

    # Load optimal task-embedding model and evaluate it on test tasks
    mean_test_loss = evaluate_optimal(opt_config,
                                      data_config,
                                      test_filenames,
                                      'test',
                                      results_dir_name)
    print(f"Mean meta-test set loss: {mean_test_loss}")

    # Get embeddings for train tasks
    embed_start = time()
    train_tasks_embeddings = embed_task_set(opt_config,
                                            data_config,
                                            train_filenames,
                                            results_dir_name)
    embed_end = time()
    print(f"Train set embedding elapsed time: {(embed_end-embed_start):.3f}s")

    # Save embeddings as a json file
    target_file = train_dir[:-11] + 'embeddings.json'
    with open(target_file, 'w', encoding='utf8') as outfile:
        json.dump(train_tasks_embeddings, outfile)

    # Get embeddings for test tasks
    embed_start = time()
    test_tasks_embeddings = embed_task_set(opt_config,
                                           data_config,
                                           test_filenames,
                                           results_dir_name)
    embed_end = time()
    print(f"Test set embedding elapsed time: {(embed_end-embed_start):.3f}s")

    # Save embeddings as a json file
    target_file = test_dir[:-11] + 'embeddings.json'
    with open(target_file, 'w', encoding='utf8') as outfile:
        json.dump(test_tasks_embeddings, outfile)

    # Visualize embeddings
    output_shape = data_config['week_num']*7*data_config['day_measurements']
    embedding_size = round(opt_config['embedding_ratio']*output_shape)

    utils.visualize_embeddings(train_tasks_embeddings,
                               test_tasks_embeddings,
                               embedding_size,
                               results_dir_name,
                               config['tsne_perplexity'])


if __name__ == "__main__":
    main()

# TODO: Add documentation where missing
