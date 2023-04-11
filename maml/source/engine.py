import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import data_setup
import model_builder
import optimizers
import utils


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
        self.second_order = args['second_order']
        self.second_to_first_order_epoch = args['second_to_first_order_epoch']

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
        self.names_weights = self.get_inner_loop_params(state_dict=self.network.state_dict(),
                                                        is_copy=False)
        self.inner_loop_optimizer.initialise(self.names_weights)

        self.to(self.device)

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


    def get_inner_loop_params(self, state_dict, is_copy):
        names_weights = {}

        if is_copy:
            for name, param in state_dict.items():
                names_weights[name] = param
        else:
            for name, param in state_dict.items():
                names_weights[name] = nn.Parameter(param)

        return names_weights
    

    def load_optimal_inner_loop_params(self, results_dir_name):
        model_save_path = results_dir_name + 'optimal_trained_model.pth'
        loaded_state_dict = torch.load(f=model_save_path)

        # Create initial weights (weights_names attribute)
        init_names_weights_dict = {}
        init_inner_loop_lr = {}

        for name in loaded_state_dict.keys():
            prefix = name.split("-")[0]
            if prefix == "layer_dict":
                new_name = str(name).replace("-", ".")
                init_names_weights_dict[new_name] = loaded_state_dict[name]
            elif prefix == "names_learning_rates_dict":
                new_name = str(name).split("-")[1:]
                new_name = "-".join(new_name)
                init_inner_loop_lr[new_name] = loaded_state_dict[name]

        self.names_weights = self.get_inner_loop_params(state_dict=init_names_weights_dict,
                                                        is_copy=False)

        self.inner_loop_optimizer.get_learned_lr(inner_loop_lr_dict=init_inner_loop_lr)


    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """

        params = {}

        for name, param in self.names_weights.items():
            params[name] = param

        for name, param in self.inner_loop_optimizer.named_parameters():
            params[name] = param

        for _, param in params.items():
            if param.requires_grad:
                yield param


    def named_params(self):
        params = {}

        for name, param in self.names_weights.items():
            params[name] = param

        for name, param in self.inner_loop_optimizer.named_parameters():
            params[name] = param

        return params
    

    def save_parameters(self, results_dir_name):
        named_params = self.named_params()

        for name, param in named_params.items():
            name = str(name).replace(".", "-")
            self.register_buffer(name, param, persistent=True)

        target_file = results_dir_name + 'optimal_trained_model.pth'
        torch.save(obj=self.state_dict(), f=target_file)


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

        grads = torch.autograd.grad(inner_epoch_loss,
                                    names_weights_copy.values(),
                                    create_graph=use_second_order,  # this causes the crash - change to use_second_order when having the available resources
                                    allow_unused=True)

        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = self.inner_loop_optimizer.update_params(
            names_weights_dict=names_weights_copy,
            names_grads_wrt_params_dict=names_grads_copy,
            num_step=inner_epoch)

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

        # print("Meta-Train started.\n")

        # Create dataloader for the training tasks
        train_tasks_dataloader = data_setup.build_tasks_set(data_filenames, 
                                                            self.data_config,
                                                            self.task_batch_size)

        # print("Created the task dataloaders.\n")

        epoch_losses = {}
        train_losses = None

        epoch_mean_support_losses = []
        epoch_mean_query_losses = []

        for epoch in range(self.num_epochs):
            epoch = int(epoch)
            # print(f"\nEpoch {epoch+1} of {self.num_epochs}.")

            use_second_order = self.second_order and (epoch < self.second_to_first_order_epoch)

            total_losses = []

            self.train()

            # Get per step importance vector
            per_step_loss_importance = self.get_per_step_loss_importance(epoch)
            # print(f"Per step loss importances: {per_step_loss_importance}")

            tasks_mean_support_losses = []
            tasks_mean_query_losses = []

            # Iterate through every task in the train tasks set
            for task_idx, (train_task_data, _) in enumerate(train_tasks_dataloader):
                # print(f"\nTrain task {task_idx+1} / {len(train_tasks_dataloader)}\n")

                train_dataloader, test_dataloader = data_setup.build_task(
                    task_data=train_task_data,
                    sample_batch_size=self.sample_batch_size,
                    data_config=self.data_config
                    )

                task_query_losses = []

                # Get a copy of the inner loop parameters 
                names_weights_copy = self.get_inner_loop_params(state_dict=self.names_weights,
                                                                is_copy=True)

                # Inner loop steps
                for inner_epoch in range(self.num_inner_steps):
                    # print(f"Inner epoch {inner_epoch+1} / {self.num_inner_steps}")

                    self.network.reset_states()

                    # Net forward on support set
                    inner_epoch_support_loss = self.inner_loop_forward(
                        dataloader=train_dataloader,
                        weights=names_weights_copy,
                        is_query_set=False
                        )

                    # print(f"Support set loss: {inner_epoch_support_loss}")

                    if inner_epoch == self.num_inner_steps-1:
                        with torch.no_grad():
                            tasks_mean_support_losses.append(
                                inner_epoch_support_loss.item()/len(train_dataloader))

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
                        weights=names_weights_copy,
                        is_query_set=True
                    )

                    # print(f"Query set loss: {inner_epoch_query_loss}\n")

                    if inner_epoch == self.num_inner_steps-1:
                        with torch.no_grad():
                            tasks_mean_query_losses.append(
                                inner_epoch_query_loss.item()/len(test_dataloader))

                    task_query_losses.append(
                        per_step_loss_importance[inner_epoch] * inner_epoch_query_loss
                        )

                # Accumulate losses from all training tasks on their query sets
                task_query_losses = torch.sum(torch.stack(task_query_losses))
                # print(f"Task query set losses: {task_query_losses}")
                total_losses.append(task_query_losses)

            # print(f"\nTotal losses: {total_losses}")

            epoch_mean_support_losses.append(np.mean(tasks_mean_support_losses))
            epoch_mean_query_losses.append(np.mean(tasks_mean_query_losses))

            # Losses is the double sum (eq.4 page 5 in How to train Your MAML)
            losses = self.get_across_task_loss_metrics(total_losses=total_losses)

            for i, item in enumerate(per_step_loss_importance):
                losses['loss_importance_vector_{}'.format(i)] = item.detach().cpu().numpy()

            # Meta update
            self.meta_update(loss=losses['loss'])

            # Get new lr from scheduler
            losses['meta_learning_rate'] = self.meta_scheduler.get_lr()[0]

            # print(f"\nLosses: {losses}")

            self.meta_scheduler.step()

            for key, value in zip(list(losses.keys()), list(losses.values())):
                if key not in epoch_losses:
                    epoch_losses[key] = [float(value)]
                else:
                    epoch_losses[key].append(float(value))

            # print(f"\nEpoch losses: {epoch_losses}")

            train_losses = self.build_summary_dict(epoch_losses=epoch_losses,
                                                   phase="train",
                                                   train_losses=train_losses)

        return train_losses, epoch_mean_support_losses, epoch_mean_query_losses
    

    def hp_tuning_meta_train(self, data_filenames):

        # Create dataloader for the training tasks
        train_tasks_dataloader = data_setup.build_tasks_set(data_filenames, 
                                                            self.data_config,
                                                            self.task_batch_size)

        for epoch in range(self.num_epochs):
            epoch = int(epoch)

            total_losses = []

            use_second_order = self.second_order and (epoch < self.second_to_first_order_epoch)

            self.train()

            # Get per step importance vector
            per_step_loss_importance = self.get_per_step_loss_importance(epoch)

            # Iterate through every task in the train tasks set
            for _, (train_task_data, _) in enumerate(train_tasks_dataloader):

                train_dataloader, test_dataloader = data_setup.build_task(
                    task_data=train_task_data,
                    sample_batch_size=self.sample_batch_size,
                    data_config=self.data_config
                    )

                task_query_losses = []

                # Get a copy of the inner loop parameters 
                names_weights_copy = self.get_inner_loop_params(state_dict=self.names_weights,
                                                                is_copy=True)

                # Inner loop steps
                for inner_epoch in range(self.num_inner_steps):

                    self.network.reset_states()

                    # Net forward on support set
                    inner_epoch_support_loss = self.inner_loop_forward(
                        dataloader=train_dataloader,
                        weights=names_weights_copy,
                        is_query_set=False
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
                        weights=names_weights_copy,
                        is_query_set=True
                    )

                    task_query_losses.append(
                        per_step_loss_importance[inner_epoch] * inner_epoch_query_loss
                        )

                # Accumulate losses from all training tasks on their query sets
                task_query_losses = torch.sum(torch.stack(task_query_losses))
                total_losses.append(task_query_losses)

            # Losses is the double sum (eq.4 page 5 in How to train Your MAML)
            losses = self.get_across_task_loss_metrics(total_losses=total_losses)

            # Meta update
            self.meta_update(loss=losses['loss'])

            # Get new lr from scheduler
            self.meta_scheduler.step()


    def evaluate(self, dataloader, weights, is_query_set=True):

        self.network.eval()
        val_task_loss = 0.0

        y_preds = []
        
        with torch.no_grad():
            for x_sample, y_sample in dataloader:
                x_sample, y_sample = x_sample.to(self.device), y_sample.to(self.device)

                y_pred = self.network.forward(x_sample=x_sample,
                                              params=weights,
                                              is_query_set=is_query_set)
                loss = F.mse_loss(y_pred, y_sample)
                val_task_loss += loss
                y_preds.append(y_pred.tolist())

            y_preds = [item for sublist in y_preds for item in sublist]

        return val_task_loss, y_preds


    def hp_tuning_meta_evaluate(self, data_filenames):

        # Create dataloader for the validation tasks
        val_tasks_dataloader = data_setup.build_tasks_set(data_filenames,
                                                          self.data_config,
                                                          self.task_batch_size)
        
        val_task_losses = []

        # Fine-tuning and evaluation on each task
        for val_task_data, _ in val_tasks_dataloader:

            # Get support and query set dataloaders
            train_dataloader, test_dataloader = data_setup.build_task(
                    task_data=val_task_data,
                    sample_batch_size=self.sample_batch_size,
                    data_config=self.data_config
                    )

            # Get a copy of the inner loop parameters 
            names_weights_copy = self.get_inner_loop_params(state_dict=self.names_weights,
                                                            is_copy=True)

            # Fine-tuning first (equivalent to inner loop optimization)
            self.train()

            # Inner loop steps
            for inner_epoch in range(self.num_inner_steps):

                self.network.reset_states()

                # Net forward on support set
                inner_epoch_support_loss = self.inner_loop_forward(
                    dataloader=train_dataloader,
                    weights=names_weights_copy,
                    is_query_set=False
                )

                # Apply inner loop update
                names_weights_copy = self.inner_loop_update(
                    inner_epoch_loss=inner_epoch_support_loss,
                    names_weights_copy=names_weights_copy,
                    use_second_order=False,
                    inner_epoch=inner_epoch
                )

            # Then evaluate on the query set
            self.eval()

            val_task_loss, _ = self.evaluate(dataloader=test_dataloader,
                                          weights=names_weights_copy,
                                          is_query_set=True)
            
            val_task_losses.append(val_task_loss)

        mean_fold_val_loss = torch.mean(torch.stack(val_task_losses))
        return mean_fold_val_loss
        

    def meta_test_optimal(self, data_filenames, results_dir_name):
        # Create dataloader for the test tasks
        test_tasks_dataloader = data_setup.build_tasks_set(data_filenames,
                                                           self.data_config,
                                                           self.task_batch_size)

        for test_task_data, test_timeseries_code in test_tasks_dataloader:
            # print("Names weights:")
            # for name, param in self.names_weights.items():
            #     print(name, param)

            support_set_losses = []
            query_set_losses = []

            # Get support and query set dataloaders
            train_dataloader, test_dataloader = data_setup.build_task(
                    task_data=test_task_data,
                    sample_batch_size=self.sample_batch_size,
                    data_config=self.data_config
                    )

            # Get a copy of the inner loop parameters 
            names_weights_copy = self.get_inner_loop_params(state_dict=self.names_weights,
                                                            is_copy=True)

            # Fine-tuning first (equivalent to inner loop optimization)
            self.train()

            # Inner loop steps
            for inner_epoch in range(self.num_inner_steps):

                self.network.reset_states()

                # Net forward on support set
                inner_epoch_support_loss = self.inner_loop_forward(
                    dataloader=train_dataloader,
                    weights=names_weights_copy,
                    is_query_set=False
                )

                with torch.no_grad():
                    support_set_losses.append(inner_epoch_support_loss.item()/len(train_dataloader))

                # Apply inner loop update
                names_weights_copy = self.inner_loop_update(
                    inner_epoch_loss=inner_epoch_support_loss,
                    names_weights_copy=names_weights_copy,
                    use_second_order=False,
                    inner_epoch=inner_epoch
                )

                # Evaluation on query set
                inner_epoch_query_loss, _ = self.evaluate(dataloader=test_dataloader,
                                                          weights=names_weights_copy,
                                                          is_query_set=True)

                with torch.no_grad():
                    query_set_losses.append(inner_epoch_query_loss.item()/len(test_dataloader))

            # Learning curve for each fine-tuned model
            utils.plot_learning_curve(support_set_losses,
                                      query_set_losses,
                                      results_dir_name,
                                      test_timeseries_code)

            # Prediction plots
            self.eval()
            _, y_preds = self.evaluate(dataloader=test_dataloader,
                                       weights=names_weights_copy,
                                       is_query_set=True)
            y_test = utils.get_task_test_set(test_dataloader)
        
            utils.plot_predictions(y_test, y_preds, results_dir_name, test_timeseries_code)

            # Save optimal fine-tuned model
            target_dir_name = results_dir_name + test_timeseries_code[0] + '/'
            self.save_parameters(target_dir_name)


def build_meta_learner(args, data_config):
    meta_learner = MetaLearner(args=args, data_config=data_config)
    return meta_learner
