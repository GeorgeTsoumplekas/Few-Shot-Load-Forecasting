"""Module that contains the low-level functions to train a model.

Specifically, a method that creates the desired optimizer is defined here, as well a method
that contains the training loop within an epoch and a method used for prediction of test data.
These functions are defined in a separate module since they can be reused both for hyperparameter
tuning process and the training of the optimal model.
"""

import torch


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


def build_scheduler(optimizer, factor, patience, threshold):
    """Create a learning rate scheduler with the specified settings.

    The scheduler reduces the learning rate whenever training reaches a plateau (train loss
    in the learning curve becomes flat).

    Args:
        optimizer: A pytorch optimizer object on which we apply the scheduler.
        factor: A float that determines the factor by which the learning rate will be reduced.
        patience: An integer that defines the number of epochs with no improvement after which
            learning rate will be reduced.
        threshold: A float that determines the threshold for measuring the new optimum,
            to only focus on significant changes

    Returns:
        scheduler: A pytorch learning rate scheduler object with the desired settings.
    """

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode='min',
                                                           factor=factor,
                                                           patience=patience,
                                                           threshold=threshold)
    return scheduler


def train_epoch(network, train_dataloader, optimizer, loss_fn, device):
    """The training loop of the model for a specific epoch.

    During the epoch, the model makes predictions on all samples of the training set, the loss is
    calculated and then back-propagation is performed based on the given optimizer to update the
    model's weights.

    Args:
        network: A custom LSTM model object to be trained.
        train_dataloader: A torch DataLoader object that contains the train set.
        optimizer: A torch optimizer object that defines the optimizer used in the training process.
        loss_fn: A function that defines the loss function based on which the training will be done.
        device: A string that defines the device on which calculations should take place.
    Returns:
        A float that represents the model loss on the training set for the specific epoch.
    """

    network.train()
    train_loss = 0.0

    # Reset model state prior to seeing the time series again
    network.reset_states()

    for x_sample, y_sample in train_dataloader:
        x_sample, y_sample = x_sample.to(device), y_sample.to(device)
        y_pred = network(x_sample)
        loss = loss_fn(y_pred, y_sample)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)

    return train_loss


def evaluate(network, val_dataloader, loss_fn, device):
    """Evaluates the model on a validation/test set.

    The model makes predictions for all samples of the validation/test set and then the loss
    is calculated based on the provided loss function.

    Args:
        network: A custom LSTM model object to be evaluated.
        val_dataloader: A torch DataLoader object that contains the val/test set.
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
