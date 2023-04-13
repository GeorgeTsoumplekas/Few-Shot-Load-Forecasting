"""Module that contains the implentation of the meta-learner.

This class typically serves as a wrapper that contains the low-level functions to train a MAML++
meta-learner. A function to build the meta-learner is also defined. The class handles the
appropriate copying and passing around of the inner and outer loop parameters as well as the
meta-training and meta-evaluation of the model.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import data_setup
import model_builder
import optimizers
import utils


class MetaLearner(nn.Module):
    """Custom MAML(++) model that contains the base network and the parameters associated with
        the meta-learner.

    It's worth noticing that the internal parameters of this module are never used. On the contrary,
    both inner loop and outer loop parameters are stored separately and used in functional calls
    of the base model whenever necessary.

    Attributes:
        device: A string that defines the device on which the model should be loaded onto.
        num_epochs: An integer which is the total number of training epochs.
        task_batch_size:  An integer that is the number of tasks in each meta-batch.
        sample_batch_size: An integer that is the number of subsequences in each batch.
        lstm_hidden_units: An integer that defines the number of units in the LSTM layer.
        init_learning_rate: A positive float with the initial value to scale gradient updates
            to the parameters by.
        meta_learning_rate: A positive float that is the learning rate of the meta-optimizer.
        eta_min: A float that represents the minimum meta-learning rate.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
        output_shape: An integer that defines the number of measurements in the predicted
            subsequence.
        num_inner_steps: An integer that is the total number of inner loop steps in each epoch.
        multi_step_loss_num_epochs: An integer that is the number of epochs to be taken account of
            when calculating the multi-step loss coefficients.
        second_order: A boolean that defines whether to use second order derivatives during
            back-propagation.
        second_to_first_order_epoch: An integer that defines the epoch on which we transit from
            calculating second order derivatives to using a first-order approximation.
        network: A BaseModel object that is the base netowrk of the meta-learner.
        inner_loop_optimizer: A custom optimizer object that is used in the inner loop steps.
        names_weights: A dictionary that contains the initial parameters of the meta-learner.
        meta_optimizer: A torch optimizer object used during outer-loop optimization.
        meta_scheduler: A torch scheduler object used to define the learning rate of the
            meta-optimizer in each epoch.
    """

    def __init__(self, args, data_config):
        """Init MetaLearner with specified configuration."""

        super().__init__()

        self.device = utils.set_device()
        self.num_epochs = args['train_epochs']
        self.task_batch_size = args['task_batch_size']
        self.sample_batch_size = args['sample_batch_size']
        self.lstm_hidden_units = args['lstm_hidden_units']
        self.init_learning_rate = args['init_learning_rate']
        self.meta_learning_rate = args['meta_learning_rate']
        self.eta_min = args['eta_min']
        self.data_config = data_config
        self.output_shape = data_config['pred_days']*data_config['day_measurements']
        self.num_inner_steps = args['num_inner_steps']
        self.multi_step_loss_num_epochs = self.num_inner_steps
        self.second_order = args['second_order']
        self.second_to_first_order_epoch = args['second_to_first_order_epoch']

        # Base model
        self.network = model_builder.build_network(input_shape=1,
                                                   output_shape=self.output_shape,
                                                   hidden_units=self.lstm_hidden_units,
                                                   device=self.device)

        self.inner_loop_optimizer = optimizers.build_LSLR_optimizer(
            self.device,
            self.num_inner_steps,
            True,
            self.init_learning_rate)

        # Keeps the true weights of the base model (not the copied ones used during the inner
        # loop optimization)
        self.names_weights = self.get_inner_loop_params(state_dict=self.network.state_dict(),
                                                        is_copy=False)

        # Create the learnable learning rate of the inner loop optimizer
        self.inner_loop_optimizer.initialise(self.names_weights)

        self.to(self.device)

        self.meta_optimizer = optimizers.build_meta_optimizer(
            params=self.trainable_parameters(),
            learning_rate=self.meta_learning_rate
            )

        self.meta_scheduler = optimizers.build_meta_scheduler(
            meta_optimizer=self.meta_optimizer,
            num_epochs=self.num_epochs,
            eta_min=self.eta_min
            )


    def get_inner_loop_params(self, state_dict, is_copy):
        """Create a copy of the inner loop parameters.

        Args:
            state_dict: A dictionary that contains the inner loop parameters.
            is_copy: A boolean that defines whether the created dictionary is a copy or the
                original.
        Returns:
            A dictionary that contains the copied inner loop parameters to be optimized.
        """

        names_weights = {}

        # Just a simple copy, otherwise gradients may not be computed correctly.
        if is_copy:
            for name, param in state_dict.items():
                names_weights[name] = param
        # But when first defined, they should be defined as parameters, to enable gradient
        # computations.
        else:
            for name, param in state_dict.items():
                names_weights[name] = nn.Parameter(param)

        return names_weights


    def load_optimal_inner_loop_params(self, results_dir_name):
        """Load the learned optimal initial parameters and learning rates.

        Useful during meta-evaluation where the initial parameters and inner loop learning rates
        have already been learned during meta-training and just have to be loaded.

        Args:
            results_dir_name: A string that defines the path to where the meta-trained model has
                been saved.
        """

        model_save_path = results_dir_name + 'optimal_trained_model.pth'
        loaded_state_dict = torch.load(f=model_save_path)

        # Create initial weights (weights_names attribute) and learned inner-loop optimizer
        # learning rates.
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

        # Set optimal learned initial weights
        self.names_weights = self.get_inner_loop_params(state_dict=init_names_weights_dict,
                                                        is_copy=False)

        # Set optimal learned inner loop learning rates
        self.inner_loop_optimizer.get_learned_lr(inner_loop_lr_dict=init_inner_loop_lr)


    def trainable_parameters(self):
        """Create an iterator over the trainable parameters of the meta-learner.

        These contain the parameters of the base model and the learnable learning rates of the
        inner loop optimizer.

        Returns:
            An iterator over the trainable parameters of the meta-learner.
        """

        params = {}

        # Base model parameters
        for name, param in self.names_weights.items():
            params[name] = param

        # Learnable inner loop optimizer learning rates
        for name, param in self.inner_loop_optimizer.named_parameters():
            params[name] = param

        for _, param in params.items():
            if param.requires_grad:
                yield param


    def named_params(self):
        """Gather together the trainable parameters of the meta-learner.

        These contain the parameters of the base model and the learnable learning rates of the
        inner loop optimizer.

        Returns:
            A dictionary that contains the trainable parameters of the meta-learner.
        """

        params = {}

        # Base model parameters
        for name, param in self.names_weights.items():
            params[name] = param

        # Learnable inner loop optimizer learning rates
        for name, param in self.inner_loop_optimizer.named_parameters():
            params[name] = param

        return params


    def save_parameters(self, results_dir_name):
        """Save the parameters of the meta-learner.

        Since the true parameters are not the internal ones and are just tensors, we have to
        register them as model buffers in order to be saved in the state dictionary of the model.

        Args:
            results_dir_name: A string with the name of the directory the results will be saved.
        """

        named_params = self.named_params()

        # Register the true meta-learner parameters
        for name, param in named_params.items():
            name = str(name).replace(".", "-")  # Because dots are not allowed in the name
            self.register_buffer(name, param, persistent=True)

        target_file = results_dir_name + 'optimal_trained_model.pth'
        torch.save(obj=self.state_dict(), f=target_file)


    def get_across_task_loss_metrics(self, total_losses):
        """Calculate the mean loss for all train tasks seen during an epoch.

        Args:
            total_losses: A list that contains the losses of a specific task's query set during
                inner loop optimization.
        Returns:
            A dictionary that contains the mean loss of all tasks.
        """

        losses = {'loss': torch.mean(torch.stack(total_losses))}

        return losses


    def build_summary_dict(self, epoch_losses, phase, train_losses=None):
        """Build/Update a summary dict directly from the metric dict of the current iteration.

        Args:
            epoch_losses: Current dict with total losses (not aggregations) from experiment
            phase: A string with the current training phase.
            train_losses: Current summarised (aggregated/summarised) losses stats means,
                stdv etc.
        Returns:
            A new summary dict with the updated summary statistics information.
        """

        if train_losses is None:
            train_losses = {}

        for key in epoch_losses:
            train_losses[f"{phase}_{key}_mean"] = np.mean(epoch_losses[key])
            train_losses[f"{phase}_{key}_std"] = np.std(epoch_losses[key])

        return train_losses


    def plot_predictions(
            self,
            test_dataloader,
            names_weights_copy,
            results_dir_name,
            test_timeseries_code):

        """Plot predicted vs true values of the given query set.

        Both true and predicted values are normalized and the plot is saved as a png file.

        Args:
            test_dataloader: A torch Dataloader object that corresponds to a task's query set.
            names_weights_copy: A dictionary that contains the inner-loop optimized weights of
                the base model.
            results_dir_name: A string with the name of the directory the results will be saved.
            test_timeseries_code: A list with a string that is the id of the examined timeseries.
        """

        self.eval()

        # Predicted values
        _, y_preds = self.evaluate(dataloader=test_dataloader,
                                weights=names_weights_copy,
                                is_query_set=True)

        # True values
        y_test = utils.get_task_test_set(test_dataloader)

        utils.plot_predictions(y_test, y_preds, results_dir_name, test_timeseries_code)


    def get_per_step_loss_importance(self, epoch):
        """
        TODO
        """

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


    # inner loop forward
    def forward(self, dataloader, weights, is_query_set):
        """The training loop of the model for a specific task in an inner-loop step.

        During a training step, the model makes predictions on all support/query set samples of
        a specific task and an aggregated loss is calculated.

        Args:
            dataloader: A torch DataLoader object that contains the examined set.
            weights: A dictionary that contains the state dict of the base model.
            is_query_set: A boolean that defines whether the given input sample belongs to the
                query set of the task.
        Returns:
            A float that is the sum of losses after seeing all samples in the given set.
        """

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
        """Apply an inner loop update of the base model's weights.

        Args:
            inner_epoch_loss: A float that is the total loss for a specific inner loop step.
            names_weights_copy: A dictionary with names to parameters to update.
            use_second_order: A boolean flag of whether to use second order derivatives.
            inner_epoch: An integer that indicates current step's index.
        Returns:
            A dictionary with the update inner-loop parameters.
        """

        self.network.zero_grads(params=names_weights_copy)

        # Using second order causes the crash - change to use_second_order when having
        # the available resources
        grads = torch.autograd.grad(inner_epoch_loss,
                                    names_weights_copy.values(),
                                    create_graph=use_second_order,
                                    allow_unused=True)

        # Dictionary with weights and their corresponding gradients
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
        """Apply an outer loop update on the meta-parameters of the model.

        Args:
            loss: A float that is the total loss value to be minimized in the outer loop updates.
        """

        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()


    def meta_train(self, data_filenames, optimal_mode):
        """
        
        """

        # print("Meta-Train started.\n")

        # Create dataloader for the training tasks
        train_tasks_dataloader = data_setup.build_tasks_set(data_filenames,
                                                            self.data_config,
                                                            self.task_batch_size)

        # print("Created the task dataloaders.\n")

        # Metrics calculated only when meta-training the optimal model
        if optimal_mode:
            epoch_losses = {}
            train_losses = None
            epoch_mean_support_losses = []
            epoch_mean_query_losses = []

        # Meta-Train loop
        for epoch in range(self.num_epochs):
            epoch = int(epoch)
            # print(f"\nEpoch {epoch+1} of {self.num_epochs}.")

            use_second_order = self.second_order and (epoch < self.second_to_first_order_epoch)
            total_losses = []

            self.train()

            # Get per step importance vector
            per_step_loss_importance = self.get_per_step_loss_importance(epoch)
            # print(f"Per step loss importances: {per_step_loss_importance}")

            # Additional metrics calculated only when meta-training the optimal model
            if optimal_mode:
                tasks_mean_support_losses = []
                tasks_mean_query_losses = []

            # Iterate through every task in the train tasks set
            for train_task_data, _ in train_tasks_dataloader:
                # print(f"\nTrain task {task_idx+1} / {len(train_tasks_dataloader)}\n")

                # Create support and query sets dataloaders
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
                    inner_epoch_support_loss = self.forward(
                        dataloader=train_dataloader,
                        weights=names_weights_copy,
                        is_query_set=False
                        )

                    # print(f"Support set loss: {inner_epoch_support_loss}")

                    # Simple (no-multi step) loss used in learning curve plots
                    if optimal_mode and (inner_epoch == (self.num_inner_steps-1)):
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

                    self.network.reset_states()

                    # Net forward on query set
                    inner_epoch_query_loss = self.forward(
                        dataloader=test_dataloader,
                        weights=names_weights_copy,
                        is_query_set=True
                    )

                    # print(f"Query set loss: {inner_epoch_query_loss}\n")

                    # Simple (no-multi step) loss used in learning curve plots
                    if optimal_mode and (inner_epoch == (self.num_inner_steps-1)):
                        with torch.no_grad():
                            tasks_mean_query_losses.append(
                                inner_epoch_query_loss.item()/len(test_dataloader))

                    # Multiply loss with the multi-step loss coefficient
                    task_query_losses.append(
                        per_step_loss_importance[inner_epoch] * inner_epoch_query_loss
                        )

                # Accumulate losses from all training tasks on their query sets
                task_query_losses = torch.sum(torch.stack(task_query_losses))
                # print(f"Task query set losses: {task_query_losses}")
                total_losses.append(task_query_losses)

            # print(f"\nTotal losses: {total_losses}")

            if optimal_mode:
                epoch_mean_support_losses.append(np.mean(tasks_mean_support_losses))
                epoch_mean_query_losses.append(np.mean(tasks_mean_query_losses))

            # Losses is the double sum (eq.4 page 5 in How to train Your MAML)
            losses = self.get_across_task_loss_metrics(total_losses=total_losses)

            # Meta update
            self.meta_update(loss=losses['loss'])

            # print(f"\nLosses: {losses}")

            self.meta_scheduler.step()

            if optimal_mode:
                for key, value in zip(list(losses.keys()), list(losses.values())):
                    if key not in epoch_losses:
                        epoch_losses[key] = [float(value)]
                    else:
                        epoch_losses[key].append(float(value))

                # print(f"\nEpoch losses: {epoch_losses}")

                # Summary statistics
                train_losses = self.build_summary_dict(epoch_losses=epoch_losses,
                                                    phase="train",
                                                    train_losses=train_losses)

        if optimal_mode:
            return train_losses, epoch_mean_support_losses, epoch_mean_query_losses


    def evaluate(self, dataloader, weights, is_query_set=True):
        """
        
        """

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


    def meta_test(self, data_filenames, optimal_mode, results_dir_name=None):
        """
        
        """

        # Create dataloader for the test tasks
        test_tasks_dataloader = data_setup.build_tasks_set(data_filenames,
                                                           self.data_config,
                                                           self.task_batch_size)

        # Used to define the objective value during hyperparameter tuning
        # The sum of losses of the query set of each task used for validation
        val_task_losses = []

        # Evaluate separately on each test task
        for test_task_data, test_timeseries_code in test_tasks_dataloader:
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
                inner_epoch_support_loss = self.forward(
                    dataloader=train_dataloader,
                    weights=names_weights_copy,
                    is_query_set=False
                )

                # Simple (no-multi step) loss used in learning curve plots
                if optimal_mode:
                    with torch.no_grad():
                        support_set_losses.append(
                            inner_epoch_support_loss.item()/len(train_dataloader)
                            )

                # Apply inner loop update
                names_weights_copy = self.inner_loop_update(
                    inner_epoch_loss=inner_epoch_support_loss,
                    names_weights_copy=names_weights_copy,
                    use_second_order=False,
                    inner_epoch=inner_epoch
                )

                # Evaluation on query set
                self.eval()

                inner_epoch_query_loss, _ = self.evaluate(dataloader=test_dataloader,
                                                          weights=names_weights_copy,
                                                          is_query_set=True)

                # Simple (no-multi step) loss used in learning curve plots
                if optimal_mode:
                    with torch.no_grad():
                        query_set_losses.append(
                            inner_epoch_query_loss.item()/len(test_dataloader)
                            )
                else:
                    val_task_losses.append(inner_epoch_query_loss)

            if optimal_mode:
                # Learning curve for each fine-tuned model
                utils.plot_learning_curve(support_set_losses,
                                          query_set_losses,
                                          results_dir_name,
                                          test_timeseries_code)

                # Prediction plots
                self.plot_predictions(test_dataloader,
                                      names_weights_copy,
                                      results_dir_name,
                                      test_timeseries_code)

                # Save optimal fine-tuned model
                target_dir_name = results_dir_name + test_timeseries_code[0] + '/'
                self.save_parameters(target_dir_name)
            else:
                mean_fold_val_loss = torch.mean(torch.stack(val_task_losses))
                return mean_fold_val_loss


def build_meta_learner(args, data_config):
    """Initialize the custom meta-learner network.

    Args:
        args: A dictionary that contains all configurations to be passed to the meta-learner.
        data_config: A dictionary that contains various user-configured values that define
            the splitting and preprocessing of the data.
    Returns:
        An initialized custom meta-learner network ready to be trained.
    """

    meta_learner = MetaLearner(args=args, data_config=data_config)
    return meta_learner
