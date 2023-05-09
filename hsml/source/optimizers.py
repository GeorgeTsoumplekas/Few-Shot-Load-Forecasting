import torch
from torch import nn


class LSLRGradientDescentLearningRule(nn.Module):  # pylint: disable=abstract-method
    """Same with the GradientDescentLearningRule class.

    The only difference is that it allows the use of learnable learning rates.
    """

    def __init__(
                 self,
                 device,
                 total_num_inner_loop_steps,
                 use_learnable_learning_rates,
                 init_learning_rate=1e-3,
                 ):
        """Creates a new LSLR gradient descent learning rule object.

        Args:
            device: A string that defines the device on which calculations should take place.
            total_num_inner_loop_steps: Integer, the total number of inner loop steps in each epoch.
            use_learnable_learning_rates: A boolean that indicates whether to use learnable learning
                rates or not.
            init_learning_rate: A positive scalar with the initial value to scale gradient updates
                to the parameters by. This needs to be carefully set - if too large the learning
                dynamic will be unstable and may diverge, while if set too small learning will
                proceed very slowly.
        """

        super().__init__()

        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate.to(device)
        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.use_learnable_learning_rates = use_learnable_learning_rates
        self.names_learning_rates_dict = nn.ParameterDict()  # contains the learnable LRs


    def initialise(self, names_weights_dict):
        """Initialise the parameter dictionary that holds the learnable learning rates.

        There are different learning rates for each layer of the network and for each inner loop
        step.

        Args:
            names_weights_dict: A dictionary that contains the variables to be optimized in the
                inner loop steps.
        """

        for key, _ in names_weights_dict.items():
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate,
                requires_grad=self.use_learnable_learning_rates)


    def get_learned_lr(self, inner_loop_lr_dict):
        """Update the larnable learning rates with the already learned values.

        This is useful during meta-evaluation of the optimal model. The learning rates have already
        been learned during meta-training and now just have to be plugged in the parameter
        dictionary of the optimiser to be immediately used.

        Args:
            inner_loop_lr_dict: A dictionary that contains the name and value of each learnable
                learning rate learned value.
        """

        for name, param in inner_loop_lr_dict.items():
            self.names_learning_rates_dict[name] = param


    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step):
        """Applies a single gradient descent update to all parameters.

        All parameter updates are performed using in-place operations and so
        nothing is returned.

        Args:
            names_weights_dict: A dictionary that contains the variables to be optimized in the
                inner loop steps.
            names_grads_wrt_params_dict: A list of gradients of the scalar loss function with
                respect to each of the parameters passed to `initialise` previously, with this list
                expected to be in the same order.
            num_step: An integer which is the number of the current inner loop step.
        """

        return {
            key: names_weights_dict[key]
            - self.names_learning_rates_dict[key.replace(".", "-")][num_step]
            * names_grads_wrt_params_dict[key]
            for key in names_grads_wrt_params_dict.keys()
        }


def build_LSLR_optimizer(device,
                         total_num_inner_loop_steps,
                         use_learnable_learning_rates,
                         init_learning_rate):
    """Crete a custom LSLR Gradient Descent optimizer with the specified settings.

    Args:
        device: A string that defines the device on which calculations should take place.
        total_num_inner_loop_steps: Integer, the total number of inner loop steps in each epoch.
        use_learnable_learning_rates: A boolean that indicates whether to use learnable learning
            rates or not.
        init_learning_rate: A float that represents the initial learning rate to be used by the
            optimizer.
    Returns:
        A custom LSLR Gradient Descent learning rule object.
    """

    optimizer = LSLRGradientDescentLearningRule(device,
                                                total_num_inner_loop_steps,
                                                use_learnable_learning_rates,
                                                init_learning_rate)
    return optimizer


def build_meta_optimizer(params, learning_rate):
    """Create an Adam optimizer with the specified settings.

    This optimizer is used as the meta-optimizer in the meta-learning setting.

    Args:
        params: An iterable of the parameters to be optimized.
        learning_rate: A float that represents the learning rate to be used by the optimizer.
    Returns:
        A pytorch Adam optimizer object with the desired settings.
    """

    optimizer = torch.optim.Adam(params=params,
                                 lr=learning_rate,
                                 amsgrad=False)
    return optimizer


def build_meta_scheduler(meta_optimizer, num_epochs, eta_min):
    """Create a cosine annealing learning rate scheduler.

    This scheduler is used to define the learning rate of the meta-optimizer in each epoch.

    Args:
        meta_optimizer: A pytorch optimizer object whose learning rate is to be set by
            the scheduler.
        num_epochs: An integer which is the total number of training epochs.
        eta_min: A float that represents the minimum learning rate.
    Returns:
        A pytorch Cosine Annealing learning rate scheduler with the desire settings.
    """

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=meta_optimizer,
                                                           T_max=num_epochs,
                                                           eta_min=eta_min)
    return scheduler
