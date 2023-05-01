"""Module that contains the low-level functionalities to train the model.

Specifically, a class that implements early stopping is included and builder functions for it and
the optimizer. Moreover, the main model's step function, a function that contains the training loop
for a single epoch, functions to evaluate and embed a certain task and a function to evaluate
all tasks of a meta-set are also included. These functions are defined in a separate module since
they can be reused for the hyperparameter tuning process and the training and fine-tuning of the
optimal model.
"""

import numpy as np
import torch

import data_setup


class EarlyStopper:
    """Custom early stopping to prematurely end training when necessary.

    The Early Stopper monitors the validation loss of the model in each epoch and terminates
    training when validation loss starts increasing instead of decreasing (meaning overfitting
    starts to appear). The model with the best generalization capability (its instance in the epoch
    where minimum validation loss was achieved). is saved internally.

    Attributes:
        patience: An integer that is the maximum number of epochs the training should go on
            without achieving a better loss than the best recorded.
        min_delta: A float that is the acceptable difference to not consider an epoch loss bigger
            the lowest recorded.
        counter: An integer that represents the number of epochs that have passed since the
            best recorded epoch.
        min_validation_loss: A float that is the minimum validation loss achieved so far by the
            trained model.
        best_model: The state dictionary of the trained model in the current best epoch
    """

    def __init__(self, network, patience, min_delta):
        """Init EarlyStopper with specified network to monitor and configurations for stopping."""

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_model = network.state_dict()


    def early_stop(self, validation_loss, network):
        """Decide whether to stop training or not based in given epoch validation loss.

        If the epoch's validation loss is worse than the best validation loss recorded and more
        epochs have passed than the accepted number of consecutive worse epochs, then training
        should stop. Otherwise training should go on. Additionally, whenever a new best validation
        loss occurs, the model's state dict at that epoch (optimal model) is saved.

        Args:
            validation_loss: A float that is the validation loss for the specific epoch.
            network: The Recurrent AutoEncoder model object that is trained.
        Returns:
            True if training should stop, otherwise False.
        """

        # If a new best epoch is found, reset the early stopping criteria and save the new loss
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model = network.state_dict()  # Save best model up to this point
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # If more 'worse' epochs have passed than patience alows, then training stops
            if self.counter >= self.patience:
                return True
        return False  # In all other cases it should go on


    def get_best_model(self):
        """Get the state dict of the model that achieved the minimum validation loss."""

        return self.best_model


def build_early_stopper(network, patience, min_delta):
    """Initialize the custom early stopper model.

    Args:
        network: A custom Recurrent AutoEncoder model object to be trained.
        patience: An integer that is the maximum number of epochs the training should go on
            without achieving a lower loss than the lowest recorded.
        min_delta: A float that is the acceptable difference to not consider an epoch loss bigger
            the lowest recorded.
    Returns:
        An initialized custom early stopper object.
    """

    return EarlyStopper(network, patience, min_delta)


def build_optimizer(network, learning_rate):
    """Create an Adam optimizer with the specified settings.

    Args:
        network: A custom Recurrent AutoEncoder model object to be trained.
        learning_rate: A float that defines the learning rate used by the Adam optimizer.
    Returns:
        A pytorch Adam optimizer object with the desired settings.
    """

    return torch.optim.Adam(params=network.parameters(), lr=learning_rate)


def model_step(network, dataloader, device):
    """
    
    """

    # Reset encoder states
    network.encoder_reset_states()

    x_sample = None
    for x_sample, _ in dataloader:
        x_sample = x_sample.to(device)
        enc_output, h_n, c_n = network.encoder_forward(x_sample)

        # TODO: Test which one is better and then move it inside the encoder's
        # forward method
        enc_output = enc_output[-1].unsqueeze(dim=0)
        # enc_output = torch.unsqueeze(torch.mean(enc_output, axis=0), dim=0)

    # Set decoder state from the last encoder state
    network.decoder_set_states(h_n, c_n)

    # Initial decoder input
    dec_input = torch.zeros_like(x_sample).to(device)

    # Contains the decoder output for all subsequences
    dec_output = torch.tensor([], device=device)

    for _ in dataloader:
        dec_input = network.decoder_forward(dec_input)
        dec_output = torch.cat([dec_input, dec_output], dim=0)

    return dec_output


def train_epoch(
    network,
    tasks_dataloader,
    loss_fn,
    optimizer,
    device,
    sample_batch_size,
    data_config,
    ):
    """
    
    """

    network.train()
    total_epoch_loss = 0.0

    # For each training task - load the task and train the model on it
    for train_task_data, _ in tasks_dataloader:
        train_dataloader, _ = data_setup.build_task(train_task_data,
                                                    sample_batch_size,
                                                    data_config)

        dec_output = model_step(network, train_dataloader, device)
        true_output = data_setup.get_full_train_set(train_dataloader).to(device)

        loss = loss_fn(dec_output, true_output)
        total_epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_epoch_loss /= len(tasks_dataloader)
    return total_epoch_loss


def evaluate_task(network, val_dataloader, loss_fn, device):
    """
    
    """

    network.eval()
    with torch.no_grad():
        dec_output = model_step(network, val_dataloader, device)
        true_output = data_setup.get_full_train_set(val_dataloader).to(device)

        val_task_loss = loss_fn(dec_output, true_output).item()

    return val_task_loss, dec_output, true_output


def evaluate(network, val_tasks_dataloader, sample_batch_size, data_config, loss_fn, device):
    """
    
    """

    val_loss = 0.0
    for val_task_data, _ in val_tasks_dataloader:
        val_dataloader, _ = data_setup.build_task(val_task_data, sample_batch_size, data_config)

        val_task_loss, _, _ = evaluate_task(network,
                                            val_dataloader,
                                            loss_fn,
                                            device)
        val_loss += val_task_loss

    val_loss /= len(val_tasks_dataloader)

    return val_loss


def embed_task(network, dataloader, device):
    """
    
    """

    # Contains the encoder output for all subsequences
    enc_outputs = torch.tensor([], device=device)

    # Reset encoder states
    network.encoder_reset_states()

    for x_sample, _ in dataloader:
        x_sample = x_sample.to(device)
        enc_output, _, _ = network.encoder_forward(x_sample)

        # TODO: Test which one is better and then move it inside the encoder's
        # forward method
        enc_output = enc_output[-1].unsqueeze(dim=0)
        # enc_output = torch.unsqueeze(torch.mean(enc_output, axis=0), dim=0)

        enc_outputs = torch.cat([enc_outputs, enc_output], dim=0)

    return torch.mean(enc_outputs, dim=0)
