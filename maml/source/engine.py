import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
    def __init__(self, args, data_config):
        super(MetaLearner, self).__init__()

        self.device = utils.set_device()
        self.num_epochs = args['train_epochs']
        self.task_batch_size = args['task_batch_size']
        self.sample_batch_size = args['sample_batch_size']
        self.lstm_hidden_units = args['lstm_hidden_units']
        self.init_learning_rate = args['init_learning_rate']
        self.meta_learning_rate = args['meta_learning_rate']
        self.T_max = self.num_epochs
        self.eta_min = args['eta_min']
        self.data_config = data_config
        self.output_shape = data_config['pred_days']*data_config['day_measurements']
        self.num_inner_steps = args['num_inner_steps']
        self.multi_step_loss_num_epochs = self.num_inner_steps

        self.network = model_builder.build_network(input_shape=1,
                                                   output_shape=self.output_shape,
                                                   hidden_units=self.lstm_hidden_units,
                                                   device=self.device,
                                                   meta_classifier=True)

        self.inner_loop_optimizer = optimizers.build_LSLR_optimizer(self.device,
                                                               self.num_inner_steps,
                                                               True,
                                                               self.init_learning_rate)

        # Keeps the true weights of the base model (not the copied ones used during the inner
        # loop optimization)
        self.names_weights = self.get_inner_loop_params(state_dict=self.network.state_dict())
        self.inner_loop_optimizer.initialise(self.names_weights)

        print(f"Names weights: {self.names_weights}")
        self.to(self.device)

        print("Inner Loop optimizer parameters:")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.meta_optimizer = optimizers.build_meta_optimizer(
            params=self.trainable_parameters(),
            learning_rate=self.meta_learning_rate
            )
        self.meta_scheduler = optimizers.build_meta_scheduler(
            meta_optimizer=self.meta_optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min
            )


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


    def get_inner_loop_params(self, state_dict):
        names_weights = {}

        for name, param in state_dict.items():
            names_weights[name] = nn.Parameter(param)

        return names_weights


    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """

        params = {}

        for name, param in self.names_weights.items():
            params[name] = param

        for name, param in self.inner_loop_optimizer.named_parameters():
            params[name] = param

        print("Outer Loop parameters:")
        for name, param in params.items():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        for _, param in params.items():
            if param.requires_grad:
                yield param


    def inner_loop_forward(self, dataloader, weights, is_query_set):

        self.network.train()
        train_step_loss = 0.0

        for x_sample, y_sample in dataloader:
            x_sample, y_sample = x_sample.to(self.device), y_sample.to(self.device)

            y_pred = self.network.forward(x_sample=x_sample,
                                          params=weights,
                                          is_query_set=is_query_set)
            loss = F.mse_loss(y_pred, y_sample)
            train_step_loss += loss

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
        # print(f"Names weights copy before: {names_weights_copy}") 
        # for key, value in names_weights_copy.items():
        #     print(key)
        #     print(value.shape)
        #     print()

        # print(f"Names weights copy values: {names_weights_copy.values()}")

        grads = torch.autograd.grad(inner_epoch_loss,
                                    names_weights_copy.values(),
                                    create_graph=False,  # this causes the crash - change to use_second_order when having the available resources
                                    allow_unused=True)

        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
        # print(f"Names grads copy: {names_grads_copy}")

        # names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}
        # print(f"Names weights copy after: \n")
        # for key, value in names_weights_copy.items():
        #     print(key)
        #     print(value.shape)
        #     print()

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = self.inner_loop_optimizer.update_params(
            names_weights_dict=names_weights_copy,
            names_grads_wrt_params_dict=names_grads_copy,
            num_step=inner_epoch)

        # names_weights_copy = {
        #     name.replace('module.', ''): value.unsqueeze(0).repeat(
        #         [1] + [1 for i in range(len(value.shape))]) for
        #     name, value in names_weights_copy.items()}

        return names_weights_copy
        

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()


    def get_across_task_loss_metrics(self, total_losses):
        losses = {'loss': torch.mean(torch.stack(total_losses))}

        return losses
    

    def build_summary_dict(self, epoch_losses, phase, train_losses=None):
        """
        Builds/Updates a summary dict directly from the metric dict of the current iteration.
        :param total_losses: Current dict with total losses (not aggregations) from experiment
        :param phase: Current training phase
        :param summary_losses: Current summarised (aggregated/summarised) losses stats means, stdv etc.
        :return: A new summary dict with the updated summary statistics information.
        """
        if train_losses is None:
            train_losses = {}

        for key in epoch_losses:
            train_losses["{}_{}_mean".format(phase, key)] = np.mean(epoch_losses[key])
            train_losses["{}_{}_std".format(phase, key)] = np.std(epoch_losses[key])

        return train_losses


    def meta_train(self, data_filenames):

        print("Meta-Train started.\n")

        # Create dataloader for the training tasks
        train_tasks_dataloader = data_setup.build_tasks_set(data_filenames, 
                                                            self.data_config,
                                                            self.task_batch_size)
        
        print("Created the task dataloaders.\n")

        epoch_losses = {}
        train_losses = None

        for epoch in range(self.num_epochs):
            epoch = int(epoch)
            print(f"\nEpoch {epoch+1} of {self.num_epochs}.")
            use_second_order = self.second_order and (epoch < self.second_to_first_order_epoch)

            print(f"Use second order: {use_second_order}")
            total_losses = []

            self.train()

            # Get per step importance vector
            per_step_loss_importance = self.get_per_step_loss_importance(epoch)
            print(f"Per step loss importances: {per_step_loss_importance}")

            # Iterate through every task in the train tasks set
            for task_idx, (train_task_data, _) in enumerate(train_tasks_dataloader):
                print(f"\nTrain task {task_idx+1} / {len(train_tasks_dataloader)}\n")

                train_dataloader, test_dataloader = data_setup.build_task(
                    task_data=train_task_data,
                    sample_batch_size=self.sample_batch_size,
                    data_config=self.data_config
                    )
                
                task_query_losses = []

                # Get a copy of the inner loop parameters 
                names_weights_copy = self.get_inner_loop_params(self.names_weights)
                
                # print(f"Names weights copy : {names_weights_copy}")

                # names_weights_copy = {
                #     name.replace('module.', ''): value.unsqueeze(0).repeat(
                #         [1] + [1 for _ in range(len(value.shape))]) for
                #     name, value in names_weights_copy.items()}
                
                # print(f"Names weights copy after: {names_weights_copy}")

                # Inner loop steps
                for inner_epoch in range(self.num_inner_steps):
                    print(f"Inner epoch {inner_epoch+1} / {self.num_inner_steps}")

                    self.network.reset_states()

                    # Net forward on support set
                    inner_epoch_support_loss = self.inner_loop_forward(
                        dataloader=train_dataloader,
                        weights=names_weights_copy,
                        is_query_set=False
                        )

                    print(f"Support set loss: {inner_epoch_support_loss}")

                    # print(f"Names weights copy shape before update:")
                    # for weight in names_weights_copy.values():
                    #     print(weight.shape)
                    # print()

                    # Apply inner loop update
                    names_weights_copy = self.inner_loop_update(
                        inner_epoch_loss=inner_epoch_support_loss,
                        names_weights_copy=names_weights_copy,
                        use_second_order=use_second_order,
                        inner_epoch=inner_epoch
                    )

                    # print(f"Names weights copy shape after update:")
                    # for weight in names_weights_copy.values():
                    #     print(weight.shape)
                    # print()

                    # print(f"Query set!")

                    # Net forward on query set
                    inner_epoch_query_loss = self.inner_loop_forward(
                        dataloader=test_dataloader,
                        weights=names_weights_copy,
                        is_query_set=True
                    )

                    print(f"Query set loss: {inner_epoch_query_loss}\n")

                    task_query_losses.append(
                        per_step_loss_importance[inner_epoch] * inner_epoch_query_loss
                        )

                # Accumulate losses from all training tasks on their query sets
                task_query_losses = torch.sum(torch.stack(task_query_losses))
                print(f"Task query set losses: {task_query_losses}")
                total_losses.append(task_query_losses)

            print(f"\nTotal losses: {total_losses}")

            # Losses is the double sum (eq.4 page 5 in How to train Your MAML)
            losses = self.get_across_task_loss_metrics(total_losses=total_losses)

            for i, item in enumerate(per_step_loss_importance):
                losses['loss_importance_vector_{}'.format(i)] = item.detach().cpu().numpy()

            # Meta update
            self.meta_update(loss=losses['loss'])

            # Get new lr from scheduler
            losses['meta_learning_rate'] = self.meta_scheduler.get_lr()[0]

            print(f"\nLosses: {losses}")

            self.meta_scheduler.step()

            for key, value in zip(list(losses.keys()), list(losses.values())):
                if key not in epoch_losses:
                    epoch_losses[key] = [float(value)]
                else:
                    epoch_losses[key].append(float(value))

            print(f"\nEpoch losses: {epoch_losses}")

            train_losses = self.build_summary_dict(epoch_losses=epoch_losses,
                                                   phase="train",
                                                   train_losses=train_losses)

        return train_losses


def build_meta_learner(opt_config, data_config):
    meta_learner = MetaLearner(opt_config=opt_config,
                               data_config=data_config)
    return meta_learner
