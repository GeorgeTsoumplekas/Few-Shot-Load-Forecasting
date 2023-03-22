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
    def __init__(self, opt_config, data_config):
        self.device = utils.set_device()
        self.num_epochs = opt_config['train_epochs']
        self.task_batch_size = opt_config['task_batch_size']
        self.sample_batch_size = opt_config['sample_batch_size']
        self.lstm_hidden_units = opt_config['lstm_hidden_units']
        self.init_learning_rate = opt_config['init_learning_rate']
        self.meta_learning_rate = opt_config['meta_learning_rate']
        self.T_max = opt_config['T_max']
        self.eta_min = opt_config['eta_min']
        self.data_config = data_config
        self.output_shape = data_config['pred_days']*data_config['day_measurements']
        self.num_inner_steps = opt_config['num_inner_steps']
        self.seed = opt_config['seed']
        self.rng = set_torch_seed(self.seed)
        self.multi_step_loss_num_epochs = opt_config['multi_step_loss_num_epochs']
        self.second_order = opt_config['second_order']
        self.second_to_first_order_epoch = opt_config['second_to_first_order_epoch']

        self.network = model_builder.build_network(input_shape=1,
                                                   output_shape=self.output_shape,
                                                   hidden_units=self.lstm_hidden_units,
                                                   device=self.device,
                                                   meta_classifier=True)
        
        inner_loop_optimizer = optimizers.build_LSLR_optimizer(self.device,
                                                               self.num_inner_steps,
                                                               True,
                                                               self.init_learning_rate)
        names_weights = self.get_inner_loop_params(params=self.network.named_parameters())
        inner_loop_optimizer.initialise(names_weights)

        print("Inner Loop parameters:")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.to(self.device)

        print("Outer Loop parameters:")
        for param in self.trainable_parameters():
            print(param)

        self.meta_optimizer = optimizers.build_meta_optimizer(
            params=self.trainable_parameters(),
            learning_rate=self.meta_learning_rate
            )
        self.meta_scheduler = optimizers.build_meta_scheduler(
            meta_optimizer=self.meta_optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min)
        
        # Maybe include this here?
        self.loss_fn = nn.MSELoss()


    def get_per_step_loss_importance(self, epoch):

        # Initially uniform weights
        loss_weights = (1.0 / self.num_inner_steps) * np.ones(shape=(self.num_inner_steps))
        decay_rate = 1.0 / self.num_inner_steps / self.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.num_inner_steps # TODO: check constant

        for i in range(len(loss_weights)-1):
            curr_value = np.maximum(
                loss_weights[i] - (epoch * decay_rate),
                min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (epoch * (self.num_inner_steps - 1) * decay_rate),
            1.0 - ((self.num_inner_steps - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights


    def get_inner_loop_params(self, params):
        inner_loop_params = {
            name: param.to(device=self.device)
            for name, param in params 
            if param.requires_grad
            }
        return inner_loop_params
    

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def inner_loop_forward(self, dataloader, weights):
        train_step_loss = 0.0
        for x_sample, y_sample in dataloader:
            x_sample, y_sample = x_sample.to(self.device), y_sample.to(self.device)

            y_pred = self.network.forward(x_sample=x_sample,
                                          params=weights)
            loss = self.loss_fn(y_pred, y_sample)
            train_step_loss += loss.item()

        return train_step_loss


    def inner_loop_update(self,
                          inner_epoch_loss,
                          names_weights_copy,
                          use_second_order,
                          inner_epoch):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """

        self.network.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(inner_epoch_loss,
                                    names_weights_copy.values(),
                                    create_graph=use_second_order,
                                    allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = self.inner_loop_optimizer.update_params(
            names_weights_dict=names_weights_copy,
            names_grads_wrt_params_dict=names_grads_copy,
            num_step=inner_epoch)

        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [1] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}

        return names_weights_copy
        

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()


    def meta_train(self, data_filenames):

        # Create dataloader for the training tasks
        train_tasks_dataloader = data_setup.build_tasks_set(data_filenames, 
                                                            self.data_config,
                                                            self.task_batch_size)
        
        for epoch in range(self.num_epochs):
            epoch = int(epoch)
            self.meta_scheduler.step(epoch=epoch)
            use_second_order = self.second_order and (epoch > self.second_to_first_order_epoch)
            total_losses = []

            # Classifier zero grad
            self.network.zero_grad()

            # Get per step importance vector
            per_step_loss_importance = self.get_per_step_loss_importance(epoch)

            # Iterate through every task in the train tasks set
            for train_task_data, _ in train_tasks_dataloader:
                train_dataloader, test_dataloader = data_setup.build_task(
                    task_data=train_task_data,
                    sample_batch_size=self.sample_batch_size,
                    data_config=self.data_config
                    )
                
                task_losses = []

                # Get inner loop parameters dict
                names_weights_copy = self.get_inner_loop_params(
                    params=self.network.named_parameters()
                )

                names_weights_copy = {
                    name.replace('module.', ''): value.unsqueeze(0).repeat(
                        [1] + [1 for _ in range(len(value.shape))]) for
                    name, value in names_weights_copy.items()}

                # Inner loop steps
                for inner_epoch in range(self.num_inner_epochs):
                    # Net forward on support set
                    inner_epoch_support_loss = self.inner_loop_forward(
                        dataloader=train_dataloader,
                        weights=names_weights_copy
                        )

                    # Apply inner loop update
                    names_weights_copy = self.inner_loop_update(
                        inner_epoch_loss=inner_epoch_support_loss,
                        names_weights_copy=names_weights_copy,
                        use_second_order=use_second_order,
                        inner_epoch=inner_epoch
                    )

                    # Net forward on query set
                    inner_epoch_query_loss = self.inner_loop_forward(
                        dataloader=test_dataloader,
                        weights=names_weights_copy
                    )

                    task_losses.append(
                        per_step_loss_importance[inner_epoch] * inner_epoch_query_loss
                        )

                # Accumulate losses from all training tasks on their query sets
                task_losses = torch.sum(torch.stack(task_losses))
                total_losses.append(task_losses)

                # Meta update
                self.meta_update(loss=losses['loss'])

                # Get new lr from scheduler
                losses['learning_rate'] = self.scheduler.get_lr()[0]
        
                # Zero grads
                self.optimizer.zero_grad()
                self.zero_grad()
