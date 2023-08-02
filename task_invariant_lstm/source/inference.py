""" Pipeline for evaluating an already trained task-invariant LSTM.

For each time series in the tasks test set, the model is fine-tuned on some of its data and
evaluated on the rest of it. The optimal models are saved as well as the corresponding learning
curves and other plots relative to its hyperparameter tuning.

python3 ./inference.py --test_dir "path/to/test_dir" \
                       --weights_dir "path/to/weights_dir" \
                       --config "path/to/config.yaml"
"""

#!/usr/bin/env python3

import argparse
import os
from time import time
import yaml

import pandas as pd
import torch

import data_setup
import engine
import losses
import model_builder
import utils


def finetune_optimal(opt_config, data_config, test_filenames,weights_dir_name, results_dir_name):
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
    model_save_path = weights_dir_name + 'optimal_trained_model.pth'
    loss = opt_config['loss']
    num_lin_layers = opt_config['linear_layers']

    # Get the dataloader for the set of test tasks
    test_tasks_dataloader = data_setup.build_tasks_set(test_filenames,
                                                       data_config,
                                                       task_batch_size,
                                                       False)

    # Fine-tuning process for each test task
    print()
    for i, (test_task_data, test_timeseries_code) in enumerate(test_tasks_dataloader):
        task_start = time()

        # Each task has an assigned directory to save relative results
        target_dir_name = results_dir_name + test_timeseries_code[0] + '/'
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        # Each fine-tuned network is derived from the same trained model
        # but is different for each validation task, since it's fine-tuned in different data.
        network = model_builder.build_network(input_shape=1,
                                              output_shape=output_shape,
                                              hidden_units=lstm_hidden_units,
                                              num_lin_layers=num_lin_layers,
                                              device=device)
        network.load_state_dict(torch.load(f=model_save_path))

        # Define an optimizer for the new model
        optimizer = engine.build_optimizer(network, learning_rate)

        loss_fn = losses.get_loss(loss)

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
        y_preds_raw = data_setup.denormalized_preds(torch.tensor(y_preds), target_dir_name)
        mape = losses.MAPE(y_preds_raw, y_task_test_raw)
        malpe = losses.MALPE(y_preds_raw, y_task_test_raw)

        task_log = pd.DataFrame({'loss_type': loss,
                                 'loss_value': [test_losses[num_epochs-1]],
                                 'length': [len(finetune_dataloader)*7],
                                 'MAPE': [mape],
                                 'MALPE': [malpe]})

        # Save logs as .csv file
        utils.save_validation_logs(task_log, target_dir_name)

        task_end = time()
        print(f"Task {i+1}|{len(test_tasks_dataloader)} / elapsed time {task_end-task_start:.2f}s")


def main():
    """End-to-end logic to run the whole experiment.

    Initially, the given command line arguments are parsed and the training and test tasks as
    well as the experiment's configuration are defined. Hyperparameter tuning is then performed
    to determine the optimal hyperparameters. Based on these, the final model is fine-tuned for each
    test task and then evaluated.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', dest='test_dir')
    parser.add_argument('--weights_dir', dest='weights_dir')
    parser.add_argument('--config', '-C', dest='config_filepath')
    args = parser.parse_args()

    test_dir = args.test_dir
    weights_dir_name = args.weights_dir
    config_filepath = args.config_filepath
    with open(config_filepath, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

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
    opt_config = config['opt_config']
    data_config = config['data_config']

    # Fine-tune and evaluate optimal model on each test time series
    finetune_optimal(opt_config, data_config, test_filenames, weights_dir_name, results_dir_name)


if __name__ == "__main__":
    main()
