import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import data_setup
import model_builder
import optimizers
import utils

class MetaLearner(nn.Module):

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
        self.num_levels = args['num_levels']
        self.num_centers = args['num_centers']
        self.sigma = args['sigma']
        self.embedding_size = args['embedding_size']

        # Base Learner
        self.network = model_builder.build_base_network(input_shape=1,
                                                        output_shape=self.output_shape,
                                                        hidden_units=self.lstm_hidden_units,
                                                        device=self.device)

        # Hierarchical Clustering Component
        self.clustering = model_builder.build_cluster_network(num_levels=self.num_levels,
                                                              num_centers=self.num_centers,
                                                              sigma=self.sigma,
                                                              embedding_size=self.embedding_size,
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

        self.total_inner_loop_params = self.get_inner_loop_params_number()
        print("Base Network Parameters")
        for name, key in self.names_weights.items():
            print(name, key.shape, np.prod(key.shape))
        print(f"Total Inner Loop Parameters: {self.total_inner_loop_params}")

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

        total_inner_loop_params = 0

        for param in self.names_weights.values():
            total_inner_loop_params += np.prod(param.shape)

        return total_inner_loop_params
    

    def trainable_parameters(self):

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
    

    def meta_train(self, data_filenames, optimal_mode):
        pass


    def meta_test(self, data_filenames, optimal_mode, results_dir_name=None):
        pass


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
