"""Module that contains the low-level functions to train a model.

Specifically, a method that creates the desired optimizer is defined here, as well a method
that contains the training loop within an epoch, a method that contains the training loop for 
each step in an epoch and a method used for prediction of test data. These functions are defined
in a separate module since they can be reused for the hyperparameter tuning process and the
training and fine-tuning of the optimal model.
"""

import torch

from data_setup import build_train_task


def build_optimizer(network, learning_rate):
    """Create an Adam optimizer with the specified settings.

    Args:
        network: A custom LSTM model object to be trained.
        learning_rate: A float that defines the learning rate used by the Adam optimizer.
    Returns:
        optimizer: A pytorch Adam optimizer object with the desired settings.
    """

    optimizer = torch.optim.Adam(params=network.parameters(), lr=learning_rate)
    return optimizer


def train_step(network, train_dataloader, loss_fn, optimizer, device):
    """The training loop of the model for a specific task in an epoch.

    During an epoch step, the model makes predictions on all samples of a specific training task,
    the loss is calculated and then back-propagation is performed based on the given optimizer
    to update the model's weights.

    Args:
        network: A custom LSTM model object to be trained.
        train_dataloader: A torch DataLoader object that contains the train set.
        loss_fn: A function that defines the loss function based on which the training will be done.
        optimizer: A torch optimizer object that defines the optimizer used in the training process.
        device: A string that defines the device on which calculations should take place.
    Returns:
        A float that represents the model loss on the training set for the specific epoch.
    """

    network.train()
    train_step_loss = 0.0

    # Reset model state prior to seeing a new time series
    network.reset_states()

    for x_sample, y_sample in train_dataloader:
        x_sample, y_sample = x_sample.to(device), y_sample.to(device)
        y_pred = network(x_sample)
        loss = loss_fn(y_pred, y_sample)
        train_step_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_step_loss


def train_epoch(
    network,
    tasks_dataloader,
    loss_fn,
    optimizer,
    device,
    sample_batch_size,
    data_config,
    ):
    """The training loop of the model for a specific epoch.

    During an epoch, the model loads each training task and performs a train step in each one
    of them. The loss is calculated as the sum of the losses of each step divided by the total
    number of sequences seen by the model during the epoch.

    Args:
        network: A custom LSTM model object to be trained.
        tasks_dataloader: A torch Dataloader that contains the training tasks.
        loss_fn: A function that defines the loss function based on which the training will be done.
        optimizer: A torch optimizer object that defines the optimizer used in the training process.
        device: A string that defines the device on which calculations should take place.
        sample_batch_size: An integer that is the number of subsequences in each batch.
        data_config: A dictionary that contains various user-configured values.
    Returns:
        A float that represents the model loss on the training set for the specific epoch.
    """

    total_epoch_loss = 0.0
    total_seqs = 0

    # For each training task - load the task and train the model on it
    for train_task_data, _ in tasks_dataloader:
        train_dataloader = build_train_task(train_task_data, sample_batch_size, data_config)
        task_train_loss = train_step(network,
                                     train_dataloader,
                                     loss_fn,
                                     optimizer,
                                     device)
        total_epoch_loss += task_train_loss
        total_seqs += len(train_dataloader)

    total_epoch_loss /= total_seqs
    return total_epoch_loss


def evaluate(network, val_dataloader, loss_fn, device):
    """Evaluates the model on a validation/test set.

    The model makes predictions for all samples of the validation/test set and then the loss
    is calculated based on the provided loss function.

    Args:
        network: A custom LSTM model object to be evaluated.
        val_dataloader: A torch Dataloader object that contains the val/test set.
        loss_fn: A function that defines the loss function based on which the evaluation
             will be done.
        device: A string that defines the device on which calculations should take place.
    Returns:
        A float that represents the model loss on the given set.
    """

    network.eval()
    val_loss = 0.0
    y_preds = []
    with torch.no_grad():
        for x_sample, y_sample in val_dataloader:
            x_sample, y_sample = x_sample.to(device), y_sample.to(device)
            y_pred = network(x_sample)
            val_loss += loss_fn(y_pred, y_sample).item()
            y_preds.append(y_pred.tolist())

        val_loss /= len(val_dataloader)
        y_preds = [item for sublist in y_preds for item in sublist]

    return val_loss, y_preds
