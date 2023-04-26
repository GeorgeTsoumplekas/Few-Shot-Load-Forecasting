import argparse
import json
import os
import yaml

from matplotlib import pyplot as plt
import optuna
from sklearn.model_selection import KFold, train_test_split
import torch
from torch import nn

from data_setup import build_tasks_set, build_task
import engine
import model_builder
import utils


def objective(trial, ht_config, data_config, data_filenames):

    # Hyperparameter search space
    config = {
        'kfolds': ht_config['k_folds'],
        'task_batch_size': ht_config['task_batch_size'],
        'sample_batch_size': ht_config['sample_batch_size'],
        'train_epochs': ht_config['train_epochs'],
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

    # Cross-validation splits - they should be random since we are splitting the tasks
    # (non-sequential) and not a single timeseries (sequential)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_idx,val_idx) in enumerate(kfold.split(data_filenames)):
        print(f"Fold {fold+1}|{k_folds}")

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
        loss_fn = nn.MSELoss()

        print("Training...")

        # Train model using the train tasks
        for _ in range(num_epochs):
            # train_epoch function
            _ = engine.train_epoch(network,
                                   train_tasks_dataloader,
                                   loss_fn,
                                   optimizer,
                                   device,
                                   sample_batch_size,
                                   data_config)

        print("Evaluating...")

        # Total validation loss for the specific fold
        val_loss = engine.evaluate(network,
                                   val_tasks_dataloader,
                                   sample_batch_size,
                                   data_config,
                                   loss_fn,
                                   device)

        mean_fold_val_losses[fold] = val_loss

        print()

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

    opt_config = {
        'task_batch_size': ht_config['task_batch_size'],
        'sample_batch_size': ht_config['sample_batch_size'],
        'train_epochs': ht_config['train_epochs'],
        'max_epochs': ht_config['max_epochs'],
        'patience': ht_config['patience'],
        'min_delta': ht_config['min_delta'],
        'learning_rate': best_trial['params_learning_rate'].values[0],
        'embedding_ratio': best_trial['params_embedding_ratio'].values[0]
    }

    return opt_config


def train_optimal(opt_config, data_config, data_filenames, results_dir_name):

    device = utils.set_device()
    task_batch_size = opt_config['task_batch_size']
    sample_batch_size = opt_config['sample_batch_size']
    num_epochs = opt_config['max_epochs']
    learning_rate = opt_config['learning_rate']
    output_shape = data_config['week_num']*7*data_config['day_measurements']
    lstm_hidden_units = round(opt_config['embedding_ratio']*output_shape)
    patience =  opt_config['patience']
    min_delta = opt_config['min_delta']

    train_filenames, val_filenames = train_test_split(data_filenames,
                                                      test_size=0.2,
                                                      random_state=42)

    train_tasks_dataloader = build_tasks_set(train_filenames,
                                             data_config,
                                             task_batch_size)

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
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}|{num_epochs}")
        # train_epoch
        train_loss = engine.train_epoch(network,
                                   train_tasks_dataloader,
                                   loss_fn,
                                   optimizer,
                                   device,
                                   sample_batch_size,
                                   data_config)
        train_losses.append(train_loss)

        # evaluate epoch
        val_loss = engine.evaluate(network,
                                   val_tasks_dataloader,
                                   sample_batch_size,
                                   data_config,
                                   loss_fn,
                                   device)
        val_losses.append(val_loss)

        # check for early stop
        if early_stopper.early_stop(val_loss, network):
            print(f"Training stopped at epoch {epoch+1}.")
            break

    # Save best model
    utils.save_model(network_state_dict=early_stopper.get_best_model(),
                     results_dir_name=results_dir_name)

    return train_losses, val_losses


def evaluate_optimal(opt_config, data_config, test_filenames, results_dir_name):

    device = utils.set_device()
    task_batch_size = opt_config['task_batch_size']
    sample_batch_size = opt_config['sample_batch_size']
    output_shape = data_config['week_num']*7*data_config['day_measurements']
    lstm_hidden_units = round(opt_config['embedding_ratio']*output_shape)
    model_save_path = results_dir_name + 'optimal_trained_model.pth'

    test_tasks_dataloader = build_tasks_set(test_filenames,
                                            data_config,
                                            task_batch_size)

    # Load optimal model
    network = model_builder.build_network(input_shape=1,
                                          output_shape=output_shape,
                                          hidden_units=lstm_hidden_units,
                                          device=device)
    network.load_state_dict(torch.load(f=model_save_path))

    loss_fn = nn.MSELoss()

    test_loss = 0.0
    for test_task_data, test_timeseries_code in test_tasks_dataloader:
        test_dataloader, _ = build_task(test_task_data, sample_batch_size, data_config)

        test_task_loss, y_preds, y_true = engine.evaluate_task(network,
                                                               test_dataloader,
                                                               loss_fn,
                                                               device)
        test_loss += test_task_loss

        # Create a pred plot for each test task
        y_preds = torch.flatten(y_preds).tolist()
        y_true = torch.flatten(y_true).tolist()
        utils.plot_predictions(y_true, y_preds, results_dir_name, test_timeseries_code)

    test_loss /= len(test_tasks_dataloader)
    return test_loss


def embed_task_set(opt_config, data_config, data_filenames, results_dir_name):

    device = utils.set_device()
    task_batch_size = opt_config['task_batch_size']
    sample_batch_size = opt_config['sample_batch_size']
    output_shape = data_config['week_num']*7*data_config['day_measurements']
    lstm_hidden_units = round(opt_config['embedding_ratio']*output_shape)
    model_save_path = results_dir_name + 'optimal_trained_model.pth'

    all_embeddings = {}

    # Load optimal model
    network = model_builder.build_network(input_shape=1,
                                          output_shape=output_shape,
                                          hidden_units=lstm_hidden_units,
                                          device=device)
    network.load_state_dict(torch.load(f=model_save_path))

    tasks_dataloader = build_tasks_set(data_filenames,
                                       data_config,
                                       task_batch_size)

    for task_data, timeseries_code in tasks_dataloader:
        train_dataloader, _ = build_task(task_data, sample_batch_size, data_config)

        # embed task function here
        task_embedding = engine.embed_task(network, train_dataloader, device)

        # add to dictionary
        all_embeddings[timeseries_code[0]] = task_embedding.tolist()

    return all_embeddings


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

    print(f"Optimal configuration: {opt_config}")

    # Train optimal task-embedding model
    train_losses, val_losses = train_optimal(opt_config,
                                             data_config,
                                             train_filenames,
                                             results_dir_name)
    utils.plot_learning_curve(train_losses, val_losses, results_dir_name)

    # Load optimal task-embedding model and evaluate it on test tasks
    test_loss = evaluate_optimal(opt_config,
                     data_config,
                     test_filenames,
                     results_dir_name)
    print(f"Test loss: {test_loss}")

    # Get embeddings for both train and test tasks
    train_tasks_embeddings = embed_task_set(opt_config,
                                            data_config,
                                            train_filenames,
                                            results_dir_name)
    target_file = train_dir[:-11] + 'embeddings.json'
    with open(target_file, 'w', encoding='utf8') as outfile:
        json.dump(train_tasks_embeddings, outfile)

    test_tasks_embeddings = embed_task_set(opt_config,
                                           data_config,
                                           test_filenames,
                                           results_dir_name)
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
