import torch

import data_setup


def build_optimizer(network, learning_rate):

    optimizer = torch.optim.Adam(params=network.parameters(), lr=learning_rate)
    return optimizer


def model_step(network, train_dataloader, device):

    network.train()
    train_step_loss = 0.0

    # Reset encoder states
    network.encoder_reset_states()

    for x_sample in train_dataloader:
        x_sample = x_sample.to(device)
        enc_output, h_n, c_n = network.encoder_forward(x_sample)

        # Test to see which one is better and then move it inside the encoder's forward method
        enc_output = enc_output[-1].unsqueeze(dim=0)
        # enc_output = torch.unsqueeze(torch.mean(enc_output, axis=0), dim=0)

    # Set decoder state from the last encoder state
    network.decoder_set_states(h_n, c_n)

    # Initial decoder input
    dec_input = torch.zeros_like(x_sample)

    # Contains the decoder output for all subsequences
    dec_output = torch.Tensor([], device=device)

    for _ in train_dataloader:
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

    total_epoch_loss = 0.0

    # For each training task - load the task and train the model on it
    for train_task_data, _ in tasks_dataloader:
        train_dataloader, _ = data_setup.build_task(train_task_data,
                                                    sample_batch_size,
                                                    data_config)

        dec_output = model_step(network, train_dataloader, device)
        true_output = data_setup.get_full_train_set(train_dataloader)

        loss = loss_fn(dec_output, true_output)
        total_epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_epoch_loss /= len(tasks_dataloader)
    return total_epoch_loss


# def evaluate(network, val_dataloader, loss_fn, device):
#     network.eval()
#     val_loss = 0.0

#     with torch.no_grad():
#         for x_sample, y_sample in val_dataloader:
