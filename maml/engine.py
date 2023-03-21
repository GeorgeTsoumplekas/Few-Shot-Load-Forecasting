import numpy as np
import torch
from torch import nn

import data_setup
import model_builder
import optimizers
import utils


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MetaLearner(nn.Module):
    pass


def meta_train(opt_config, data_config, data_filenames):

    # Set random seed for reproducibility purposes
    torch.manual_seed(opt_config['seed'])
    torch.cuda.manual_seed(opt_config['seed'])

    device = utils.set_device()
    num_epochs = opt_config['train_epochs']
    task_batch_size = opt_config['task_batch_size']
    sample_batch_size = opt_config['sample_batch_size']
    lstm_hidden_units = opt_config['lstm_hidden_units']
    learning_rate = opt_config['learning_rate']
    T_max = opt_config['T_max']
    eta_min = opt_config['eta_min']
    output_shape = data_config['pred_days']*data_config['day_measurements']
    num_inner_epochs = opt_config['num_inner_epochs']

    # Create dataloader for the training tasks
    train_tasks_dataloader = data_setup.build_tasks_set(data_filenames, 
                                                        data_config,
                                                        task_batch_size)
    
    meta_optimizer = optimizers.build_meta_optimizer()
    meta_scheduler = optimizers.build_meta_scheduler(meta_optimizer, T_max, eta_min)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        # Iterate through every task in the train tasks set
        for train_task_data, _ in train_tasks_dataloader:
            train_dataloader, test_dataloader = data_setup.build_task(train_task_data,
                                                                      sample_batch_size,
                                                                      data_config)
            
            # Get per step importance vector

            # Get inner loop parameters dict

            # Inner loop steps
            for inner_epochs in range(num_inner_epochs):
                # Net forward on support set

                # Apply inner loop update
                
                # Net forward on query set
                pass

            # Accumulate losses from all training tasks on their query sets

            # Meta update

            # Get new lr from scheduler

            # Zero grads
