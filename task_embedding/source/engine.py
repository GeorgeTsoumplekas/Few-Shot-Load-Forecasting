import numpy as np
import torch

import data_setup


class EarlyStopper:

    def __init__(self, network, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_model = network.state_dict()


    def early_stop(self, validation_loss, network):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model = network.state_dict()  # Save best model up to this point
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


    def get_best_model(self):
        return self.best_model


def build_early_stopper(network, patience, min_delta):
    return EarlyStopper(network, patience, min_delta)


def build_optimizer(network, learning_rate):
    return torch.optim.Adam(params=network.parameters(), lr=learning_rate)


def model_step(network, dataloader, device):

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

    network.eval()
    with torch.no_grad():
        dec_output = model_step(network, val_dataloader, device)
        true_output = data_setup.get_full_train_set(val_dataloader).to(device)

        val_task_loss = loss_fn(dec_output, true_output).item()

    return val_task_loss, dec_output, true_output


def evaluate(network, val_tasks_dataloader, sample_batch_size, data_config, loss_fn, device):

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
