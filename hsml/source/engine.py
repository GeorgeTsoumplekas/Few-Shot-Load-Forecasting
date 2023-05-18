import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

import data_setup
import losses
import model_builder
import optimizers
import utils

class MetaLearner(nn.Module):

    def __init__(self, args, data_config):
        #TODO: Add comments where needed and remove print statements after done testing the fucntions
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
        self.num_levels = args['num_levels']
        self.num_centers = args['num_centers']
        self.sigma = args['sigma']
        self.embedding_size = args['embedding_size']
        self.loss = args['loss']
        self.kappa = args['kappa']

        # Base Learner
        self.network = model_builder.build_base_network(input_shape=1,
                                                        output_shape=self.output_shape,
                                                        hidden_units=self.lstm_hidden_units,
                                                        device=self.device)

        self.inner_loop_optimizer = optimizers.build_LSLR_optimizer(
            self.device,
            self.num_inner_steps,
            True,
            self.init_learning_rate
        )

        # Keeps the true weights of the base model (not the copied ones used during the inner
        # loop optimization)
        self.names_weights = self.get_inner_loop_params(state_dict=self.network.state_dict(),
                                                        is_copy=False)

        self.total_inner_loop_params = self.get_inner_loop_params_number()
        print("Base Network Parameters")
        for name, key in self.names_weights.items():
            print(name, key.shape, np.prod(key.shape))
        print(f"Total Inner Loop Parameters: {self.total_inner_loop_params}")

        # Hierarchical Clustering Component
        self.clustering = model_builder.build_cluster_network(num_levels=self.num_levels,
                                                              num_centers=self.num_centers,
                                                              sigma=self.sigma,
                                                              embedding_size=self.embedding_size,
                                                              device=self.device)

        # Parameter Gate
        self.parameter_gate = model_builder.build_parameter_gate(
            input_shape=2*self.embedding_size,
            output_shape=self.total_inner_loop_params,
            device=self.device
            )

        print("\nClustering Network Parameters")
        cn_params_number = 0
        for name, param in self.clustering.named_parameters():
            cn_params_number += np.prod(param.shape)
            print(name, param.shape)
        print(f"Total Clustering Network Parameters: {cn_params_number}")

        print("\nParameter Gate Parameters")
        pg_params_number = 0
        for name, param in self.parameter_gate.named_parameters():
            pg_params_number += np.prod(param.shape)
            print(name, param.shape)
        print(f"Total Parameter Gate Parameters: {pg_params_number}")

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

        print("\nOuter Loop Parameters")
        num_outer_loop_parameters = 0
        outer_loop_parameters = self.named_params()
        for name, param in outer_loop_parameters.items():
            print(name, param.shape)
            num_outer_loop_parameters += np.prod(param.shape)
        print(f"Total outer loop parameters: {num_outer_loop_parameters}")


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


    def get_inner_loop_params_number(self):
        # TODO: Add documentation

        total_inner_loop_params = 0

        for param in self.names_weights.values():
            total_inner_loop_params += np.prod(param.shape)

        return total_inner_loop_params


    def trainable_parameters(self):
        # TODO: Add documentation

        params = {}

        # Base model parameters
        for name, param in self.names_weights.items():
            params[name] = param

        # Learnable inner loop optimizer learning rates
        for name, param in self.inner_loop_optimizer.named_parameters():
            params[name] = param

        # Hierarchical Clustering Model parameters
        for name, param in self.clustering.named_parameters():
            params[name] = param

        # Parameter Gate parameters
        for name, param in self.parameter_gate.named_parameters():
            params[name] = param

        for _, param in params.items():
            if param.requires_grad:
                yield param


    def named_params(self):
        # TODO: Add documentation

        params = {}

        # Base model parameters
        for name, param in self.names_weights.items():
            params[name] = param

        # Learnable inner loop optimizer learning rates
        for name, param in self.inner_loop_optimizer.named_parameters():
            params[name] = param

        # Hierarchical Clustering Model parameters
        for name, param in self.clustering.named_parameters():
            params[name] = param

        # Parameter Gate parameters
        for name, param in self.parameter_gate.named_parameters():
            params[name] = param

        return params
    

    def get_across_task_loss_metrics(self, total_losses):
        """Calculate the mean loss for all train tasks seen during an epoch.

        Args:
            total_losses: A list that contains the losses of a specific task's query set during
                inner loop optimization.
        Returns:
            A dictionary that contains the mean loss of all tasks.
        """

        losses_dict = {'loss': torch.mean(torch.stack(total_losses))}

        return losses_dict


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


    def plot_predictions(self, test_dataloader, names_weights_copy, target_dir_name):
        """Plot predicted vs true values of the given query set.

        Both true and predicted values are normalized and the plot is saved as a png file.

        Args:
            test_dataloader: A torch Dataloader object that corresponds to a task's query set.
            names_weights_copy: A dictionary that contains the inner-loop optimized weights of
                the base model.
            target_dir_name: A string with the name of the directory the results will be saved.
        """

        self.eval()

        # Predicted values
        _, y_preds = self.evaluate(dataloader=test_dataloader,
                                weights=names_weights_copy,
                                is_query_set=True)

        # True values
        y_test = data_setup.get_task_test_set(test_dataloader)

        utils.plot_predictions(y_test, y_preds, target_dir_name)


    def plot_learning_curve(self, train_losses, test_losses, target_dir_name, loss):
        """Plot the learning curve of the desired model for a specific test task.

        The plot contains the train loss and the test loss of the model for the number
        of inner loop steps. The plot is saved as a png file.

        Args:
            train_losses: A list that contains the support set loss of each inner loop step for a
                specific task.
            test_losses: A list that contains the query set loss of each inner loop step for a
                specific task.
            target_dir_name: A string with the name of the directory the results will be saved.
            loss: A string that is the name of the loss function used.
        """

        utils.plot_learning_curve(train_losses, test_losses, target_dir_name, loss)


    def calculate_MAPE(
            self,
            test_dataloader,
            names_weights_copy,
            y_task_query_raw,
            target_dir_name
        ):
        """Calculate MAPE metric for the specific query set of a task.

        MAPE calculation is performed on the orignal scale of the timeseries' and as a result
        first it is necessary to obtain the raw time series data and de-standardize the model's
        predictions.

        Args:
            test_dataloader: A torch Dataloader object that corresponds to a task's query set.
            names_weights_copy: A dictionary that contains the inner-loop optimized weights of
                the base model.
            y_task_query_raw: A torch tensor that includes the true output values in the original
                scale.
            target_dir_name: A string with the name of the directory the results will be saved.
        Returns:
            A float that is the MAPE metric for the given query set.
        """

        self.eval()

        # Predicted values
        _, y_preds = self.evaluate(dataloader=test_dataloader,
                                weights=names_weights_copy,
                                is_query_set=True)

        # Predicted values on original scale
        y_preds_raw = data_setup.unstandardized_preds(torch.tensor(y_preds), target_dir_name)

        return losses.MAPE(y_preds_raw, y_task_query_raw)


    def create_logs(self, query_set_losses, train_dataloader, mape, target_dir_name):
        """Create and save a .csv validation log file.

        The file contains info about the losses, metrics and type of time series.

        Args:
            query_set_losses: A list that contains a tasks's query set loss for every inner loop
                step.
            train_dataloader: A torch Dataloader object that corresponds to a task's support set.
            mape: A float that is the MAPE metric for the specific tasks's query set.
            target_dir_name: A string with the name of the directory the logs will be saved.
        """

        task_log = pd.DataFrame({
            'loss_type': self.loss,
            'loss_value': [query_set_losses[self.num_inner_steps-1]],
            'length': [len(train_dataloader)*7],
            'MAPE': [mape]
        })

        # Save logs as .csv file
        utils.save_validation_logs(task_log, target_dir_name)


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
    

    def get_per_step_loss_importance(self, epoch):
        """Calculate importance coefficients of the multi step loss for each epoch.

        At first, the coefficients are the same for all inner loop steps. As epochs proceed,
        the coefficinets of all but the last inner loop step decrease based on a specified decay
        rate. Meanwhile the coefficient of the last step increases in such a way that the sum of
        all coefficients for a specific epoch equal one.

        Args:
            epoch: An integer that is the index of the current epoch.
        Returns:
            A torch tensor with the calculated coefficients.
        """

        # Initially uniform weights
        loss_weights = (1.0 / self.num_inner_steps) * np.ones(shape=(self.num_inner_steps))

        # Coefficient by which the learning rate decays in each step
        # Depends on the number of inner loop steps times the number of epochs taken into
        # consideration when calculating the multi step loss
        decay_rate = 1.0 / self.num_inner_steps / self.multi_step_loss_num_epochs

        # Determine a minimum inner loop learning rate
        min_value_for_non_final_losses = 0.03 / self.num_inner_steps # TODO: check constant

        for i in range(len(loss_weights)-1):
            curr_value = np.maximum(
                loss_weights[i] - (epoch * decay_rate),
                min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        # As epochs proceed, more and more weight is given towards the last step of the inner loop
        curr_value = np.minimum(
            loss_weights[-1] + (epoch * (self.num_inner_steps - 1) * decay_rate),
            1.0 - ((self.num_inner_steps - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights


    def apply_weights_mask(self, names_weights_copy, weights_mask):
        # TODO: Add documentation

        names_weights_masked = {}

        start = 0
        for name, param in names_weights_copy.items():
            end = start + torch.prod(torch.tensor([*param.shape]))

            mask = weights_mask[0][start:end].view(param.shape)
            names_weights_masked[name] = torch.mul(param, mask)

            start = end

        return names_weights_masked


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
        loss_fn = losses.get_loss(self.loss, self.kappa, self.device)

        for x_sample, y_sample in dataloader:
            x_sample, y_sample = x_sample.to(self.device), y_sample.to(self.device)

            y_pred = self.network.forward(x_sample=x_sample,
                                          params=weights,
                                          is_query_set=is_query_set)
            loss = loss_fn(y_pred, y_sample)
            train_step_loss += loss

        return train_step_loss


    def inner_loop_update(self,
                          inner_epoch_loss,
                          names_weights_masked,
                          use_second_order,
                          inner_epoch):
        """Apply an inner loop update of the base model's weights.

        Args:
            inner_epoch_loss: A float that is the total loss for a specific inner loop step.
            names_weights_masked: A dictionary with names to parameters to update.
            use_second_order: A boolean flag of whether to use second order derivatives.
            inner_epoch: An integer that indicates current step's index.
        Returns:
            A dictionary with the update inner-loop parameters.
        """

        self.network.zero_grads(params=names_weights_masked)

        # Using second order causes the crash - change to use_second_order when enough
        # resources are available
        grads = torch.autograd.grad(inner_epoch_loss,
                                    names_weights_masked.values(),
                                    create_graph=use_second_order,
                                    allow_unused=True)

        # Dictionary with weights and their corresponding gradients
        names_grads_copy = dict(zip(names_weights_masked.keys(), grads))

        for key, _ in names_grads_copy.items():
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_masked = self.inner_loop_optimizer.update_params(
            names_weights_dict=names_weights_masked,
            names_grads_wrt_params_dict=names_grads_copy,
            num_step=inner_epoch)

        return names_weights_masked
    

    def meta_update(self, loss):
        """Apply an outer loop update on the meta-parameters of the model.

        Args:
            loss: A float that is the total loss value to be minimized in the outer loop updates.
        """

        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()


    def meta_train(self, data_filenames, embeddings, optimal_mode):
        # TODO: Add documentation and comments if necessary

        # Create dataloader for the training tasks
        train_tasks_dataloader = data_setup.build_tasks_set(data_filenames,
                                                            self.data_config,
                                                            self.task_batch_size,
                                                            embeddings)

        # Metrics calculated only when meta-training the optimal model
        if optimal_mode:
            epoch_losses = {}
            train_losses = None
            epoch_mean_support_losses = []
            epoch_mean_query_losses = []

        # Meta-Train loop
        for epoch in range(self.num_epochs):
            epoch = int(epoch)

            use_second_order = self.second_order and (epoch < self.second_to_first_order_epoch)
            total_losses = []

            self.train()

            # Get per step importance vector
            per_step_loss_importance = self.get_per_step_loss_importance(epoch)

            # Additional metrics calculated only when meta-training the optimal model
            if optimal_mode:
                tasks_mean_support_losses = []
                tasks_mean_query_losses = []

            # Iterate through every task in the train tasks set
            for train_task_data, _, train_task_embedding in train_tasks_dataloader:

                # Create support and query sets dataloaders
                train_dataloader, test_dataloader, _ = data_setup.build_task(
                    task_data=train_task_data,
                    sample_batch_size=self.sample_batch_size,
                    data_config=self.data_config
                )

                # Compute h_i using HierarchicalClustering module
                train_task_embedding = train_task_embedding.to(self.device)
                train_task_cluster_embedding = self.clustering(train_task_embedding)

                # Compute o_i using the ParameterGate module
                weights_mask = self.parameter_gate(train_task_embedding, train_task_cluster_embedding)

                # Get a copy of the inner loop parameters
                names_weights_copy = self.get_inner_loop_params(state_dict=self.names_weights,
                                                                is_copy=True)

                # Update initial inner loop parameters for this task based on o_i
                names_weights_masked = self.apply_weights_mask(names_weights_copy, weights_mask)

                task_query_losses = []

                # Inner loop steps
                for inner_epoch in range(self.num_inner_steps):

                    self.network.reset_states()

                    # Net forward on support set
                    inner_epoch_support_loss = self.forward(
                        dataloader=train_dataloader,
                        weights=names_weights_masked,
                        is_query_set=False
                        )

                    # Simple (no-multi step) loss used in learning curve plots
                    if optimal_mode and (inner_epoch == (self.num_inner_steps-1)):
                        with torch.no_grad():
                            tasks_mean_support_losses.append(
                                inner_epoch_support_loss.item()/len(train_dataloader))

                    # Apply inner loop update
                    names_weights_masked = self.inner_loop_update(
                        inner_epoch_loss=inner_epoch_support_loss,
                        names_weights_masked=names_weights_masked,
                        use_second_order=use_second_order,
                        inner_epoch=inner_epoch
                    )

                    self.network.reset_states()

                    # Net forward on query set
                    inner_epoch_query_loss = self.forward(
                        dataloader=test_dataloader,
                        weights=names_weights_masked,
                        is_query_set=True
                    )

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
                total_losses.append(task_query_losses)

            if optimal_mode:
                epoch_mean_support_losses.append(np.mean(tasks_mean_support_losses))
                epoch_mean_query_losses.append(np.mean(tasks_mean_query_losses))

            # Losses is the double sum (eq.4 page 5 in How to train Your MAML)
            losses_dict = self.get_across_task_loss_metrics(total_losses=total_losses)

            # Meta update
            self.meta_update(loss=losses_dict['loss'])

            self.meta_scheduler.step()

            if optimal_mode:
                for key, value in zip(list(losses_dict.keys()), list(losses_dict.values())):
                    if key not in epoch_losses:
                        epoch_losses[key] = [float(value)]
                    else:
                        epoch_losses[key].append(float(value))

                # Summary statistics
                train_losses = self.build_summary_dict(epoch_losses=epoch_losses,
                                                    phase="train",
                                                    train_losses=train_losses)

        if optimal_mode:
            return train_losses, epoch_mean_support_losses, epoch_mean_query_losses


    def evaluate(self, dataloader, weights, is_query_set=True):
        """Evaluate the model on the query set of a specific task.

        The model makes predictions for all samples of the validation/test set and then the loss
        is calculated based on the provided loss function.

        Args:
            dataloader: A torch DataLoader object that contains the examined set.
            weights: A dictionary that contains the state dict of the base model.
            is_query_set: A boolean that defines whether the given input sample belongs to the
                query set of the task.
        Returns:
            A float that represents the model loss on the given set and a list that contains
            the predicted values for the given set.
        """

        self.network.eval()
        val_task_loss = 0.0

        y_preds = []
        loss_fn = losses.get_loss(self.loss, self.kappa, self.device)

        with torch.no_grad():
            for x_sample, y_sample in dataloader:
                x_sample, y_sample = x_sample.to(self.device), y_sample.to(self.device)

                y_pred = self.network.forward(x_sample=x_sample,
                                              params=weights,
                                              is_query_set=is_query_set)
                loss = loss_fn(y_pred, y_sample)
                val_task_loss += loss
                y_preds.append(y_pred.tolist())

            y_preds = [item for sublist in y_preds for item in sublist]

        return val_task_loss, y_preds


    def meta_test(self, data_filenames, embeddings, optimal_mode, results_dir_name=None):
        # TODO: Add documentation

        # Create dataloader for the test tasks
        test_tasks_dataloader = data_setup.build_tasks_set(data_filenames,
                                                           self.data_config,
                                                           self.task_batch_size,
                                                           embeddings)

        # Used to define the objective value during hyperparameter tuning
        # The sum of losses of the query set of each task used for validation
        val_task_losses = []

        # Evaluate separately on each test task
        for test_task_data, test_timeseries_code, test_task_embedding in test_tasks_dataloader:
            support_set_losses = []
            query_set_losses = []

            # If evaluating the optimal model, create a corresponding results directory
            # for each task
            if optimal_mode:
                target_dir_name = results_dir_name + test_timeseries_code[0] + '/'
                if not os.path.exists(target_dir_name):
                    os.makedirs(target_dir_name)
            else:
                target_dir_name = None

            # Get support and query set dataloaders
            train_dataloader, test_dataloader, y_task_query_raw = data_setup.build_task(
                    task_data=test_task_data,
                    sample_batch_size=self.sample_batch_size,
                    data_config=self.data_config,
                    target_dir_name=target_dir_name
            )

            # Compute h_i using HierarchicalClustering module
            test_task_embedding = test_task_embedding.to(self.device)
            test_task_cluster_embedding = self.clustering(test_task_embedding)

            # Compute o_i using the ParameterGate module
            weights_mask = self.parameter_gate(test_task_embedding, test_task_cluster_embedding)

            # Get a copy of the inner loop parameters
            names_weights_copy = self.get_inner_loop_params(state_dict=self.names_weights,
                                                            is_copy=True)

            # Update initial inner loop parameters for this task based on o_i
            names_weights_masked = self.apply_weights_mask(names_weights_copy, weights_mask)

            # Inner loop steps
            for inner_epoch in range(self.num_inner_steps):

                # Fine-tuning first (equivalent to inner loop optimization)
                self.train()

                self.network.reset_states()

                # Net forward on support set
                inner_epoch_support_loss = self.forward(
                    dataloader=train_dataloader,
                    weights=names_weights_masked,
                    is_query_set=False
                )

                # Simple (no-multi step) loss used in learning curve plots
                if optimal_mode:
                    with torch.no_grad():
                        support_set_losses.append(
                            inner_epoch_support_loss.item()/len(train_dataloader)
                            )

                # Apply inner loop update
                names_weights_masked = self.inner_loop_update(
                    inner_epoch_loss=inner_epoch_support_loss,
                    names_weights_masked=names_weights_masked,
                    use_second_order=False,
                    inner_epoch=inner_epoch
                )

                # Evaluation on query set
                self.eval()

                inner_epoch_query_loss, _ = self.evaluate(dataloader=test_dataloader,
                                                          weights=names_weights_masked,
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
                self.plot_learning_curve(support_set_losses,
                                          query_set_losses,
                                          target_dir_name,
                                          self.loss)

                # Prediction plots
                self.plot_predictions(test_dataloader, names_weights_masked, target_dir_name)

                # Save optimal fine-tuned model
                self.save_parameters(target_dir_name)

                # MAPE calculation on original scale
                mape = self.calculate_MAPE(
                    test_dataloader,
                    names_weights_masked,
                    y_task_query_raw,
                    target_dir_name
                )

                # Create log file
                self.create_logs(query_set_losses, train_dataloader, mape, target_dir_name)
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
