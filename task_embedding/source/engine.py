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
    """Perform a full forward pass of the recurrent autoencoder on the given data set.

    This function is used when training or evaluating the model to get the reconstruction of the
    given dataset and is not necessary for inference (getting the task embeddings). First the all
    samples are passed through the encoder and the last state of the encoder is used as the initial
    state of the decoder. Then the decoder produces the reconstructed dataset in reverse order.

    Args:
        network: A custom RAE model object to be trained/validated.
        dataloader: A torch DataLoader object that contains the dataset.
        device: A string that defines the device on which calculations should take place.
    Returns:
        A torch tensor that contains the reconstructed data set.
    """

    # Reset encoder states
    network.encoder_reset_states()

    x_sample = None
    for x_sample, _ in dataloader:
        x_sample = x_sample.to(device)
        _, h_n, c_n = network.encoder_forward(x_sample)

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
    """The training loop of the model for a specific epoch.

    During an epoch, the model loads each training task and performs a train step in each one
    of them. The loss is calculated as the sum of the losses of each task divided by the total
    number of tasks seen by the model during the epoch.

    Args:
        network: A custom RAE model object to be trained.
        tasks_dataloader: A torch Dataloader that contains the training tasks.
        loss_fn: A function that defines the loss function based on which the training will be done.
        optimizer: A torch optimizer object that defines the optimizer used in the training process.
        device: A string that defines the device on which calculations should take place.
        sample_batch_size: An integer that is the number of subsequences in each batch.
        data_config: A dictionary that contains various user-configured values.
    Returns:
        A float that represents the model loss on the training set for the specific epoch.
    """

    network.train()
    total_epoch_loss = 0.0

    # For each training task - load the task and train the model on it
    for train_task_data, _ in tasks_dataloader:
        train_dataloader, _ = data_setup.build_task(train_task_data,
                                                    sample_batch_size,
                                                    data_config)

        # Reconstructed support set
        dec_output = model_step(network, train_dataloader, device)

        # Original support set
        true_output = data_setup.get_full_train_set(train_dataloader).to(device)

        loss = loss_fn(dec_output, true_output)
        total_epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_epoch_loss /= len(tasks_dataloader)
    return total_epoch_loss


def evaluate_task(network, val_dataloader, loss_fn, device):
    """Evaluates the model on a single task of the validation/test task set.

    The model reconstructs the support set of the given task and the reconstruction loss is
    calculated based on the provided loss function.

    Args:
        network: A custom RAE model object to be evaluated.
        val_dataloader: A torch Dataloader object that contains the support set of the
            validation/test task.
        loss_fn: A function that defines the loss function based on which the evaluation
             will be done.
        device: A string that defines the device on which calculations should take place.
    Returns:
        A float that is the reconstruction loss for the specific task, a tensor that contains
        the reconstructed support set and a tensor that contains the original support set.
    """

    network.eval()
    with torch.no_grad():
        # Reconstructed support set
        dec_output = model_step(network, val_dataloader, device)

        # Original support set
        true_output = data_setup.get_full_train_set(val_dataloader).to(device)

        val_task_loss = loss_fn(dec_output, true_output).item()

    return val_task_loss, dec_output, true_output


def evaluate(network, val_tasks_dataloader, sample_batch_size, data_config, loss_fn, device):
    """Evaluate model on the meta-validation/test set.

    The model reconstructs the support set of all tasks in the meta-validation/test set and then
    the loss is calculated as the mean loss of all individual tasks.

    Args:
        network: A custom RAE model object to be evaluated.
        val_tasks_dataloader: A torch Dataloader that contains the validation/test tasks.
        sample_batch_size: An integer that is the number of subsequences in each batch.
        data_config: A dictionary that contains various user-configured values.
        loss_fn: A function that defines the loss function based on which the evaluation
             will be done.
        device: A string that defines the device on which calculations should take place.
    Returns:
        A float that is the mean reconstruction loss for all tasks in the meta-set.
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
    """Forward pass of the recurrent autoencoder that yields the embedding of a task.

    First, the embedding of each subsequence of the task is created and the final embedding
    of the whole task is the mean of all subsequence embeddings.

    Args:
        network: A custom RAE embedding model object.
        dataloader: A torch Dataloader that contains the task to be embedded.
        device: A string that defines the device on which calculations should take place.
    Returns:
        A torch tensor that represents the embedding of the task.
    """

    # Contains the encoder output for all subsequences
    enc_outputs = torch.tensor([], device=device)

    # Reset encoder states
    network.encoder_reset_states()

    for x_sample, _ in dataloader:
        x_sample = x_sample.to(device)
        enc_output, _, _ = network.encoder_forward(x_sample)

        enc_outputs = torch.cat([enc_outputs, enc_output], dim=0)

    return torch.mean(enc_outputs, dim=0)
