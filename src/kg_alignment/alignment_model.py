"""This file contains container classes for multiple models used for alignment tasks (e.g. 2 generators and
2 discriminators."""

import torch
from input_output import export_cyclegan_alignment, export_regression_alignment, save_alignment_test_results,\
    load_cyclegan_alignment, load_regression_alignment
from generator import Generator
from discriminator import Discriminator
from constants import OPTIMIZERS
from copy import deepcopy
from abc import ABC, abstractmethod


class AlignmentModel(ABC):
    """The abstract class for alignment models, which defines basic variables and functions that must be implemented by
    alignment models."""

    def __init__(self, device, config):
        """Store the torch device and the training configurations, the basic parameters for alignment training."""
        self.device = device
        self.config = config
        self.current_lr = config['learning_rate']
        self.model_copy = None
        self.metrics = []
        self.losses = []
        self.epoch_losses = None

    @abstractmethod
    def train(self):
        """Set all models to training mode by calling their "train()" function."""
        pass

    @abstractmethod
    def eval(self):
        """Set all models to evaluation mode by calling their "eval()" function."""
        pass

    @abstractmethod
    def zero_grad(self):
        """Reset the gradients of all models by calling the "zero()" function of the respective optimizers. """
        pass

    @abstractmethod
    def optimize_all(self):
        """Do the optimization step for all models by calling the "step()" function of the respective optimizers."""
        pass

    @abstractmethod
    def change_lr(self, factor):
        """Manually change the current optimizer learning rate by multiplying it with the given factor."""
        pass

    @abstractmethod
    def update_losses_batch(self, *losses):
        """Add the losses of a batch to the current epoch loss (scaled by the batch size)."""
        pass

    @abstractmethod
    def complete_epoch(self, epoch_metrics):
        """Conclude the epoch losses by calculating the mean of the summarized batch losses, add optional loss values
        (e.g. from cycle loss of the test data) and append the epoch loss and the validation metrics to the training
        history ("metrics" array and "losses" array)."""
        pass

    @abstractmethod
    def print_epoch_info(self):
        """Print the validation metrics and model losses of the latest epoch."""
        pass

    @abstractmethod
    def copy_model(self):
        """Create a copy (deepcopy) of the state dictionary of all models (e.g. when saving best model based on
        validation metrics)."""
        pass

    @abstractmethod
    def restore_model(self):
        """Load the previously saved state dictionaries back into the models to return to an earlier model (e.g. the
        best model seen during training process based on validation metrics)."""
        pass

    @abstractmethod
    def export_model(self, test_results, description=None):
        """Export the alignment architecture by saving all model state dictionaries, configs and the evaluation
        results."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, name, device):
        """Load the previously saved model state dictionaries and configurations (identified by the directory name) to
        initialize a pre-trained alignment architecture."""
        pass


class RegressionModel(AlignmentModel):
    """This class implements the alignment model for use cases with two generators (simple regression and cycle
    regression). For description of the implemented functions, refer to the alignment model."""

    def __init__(self, device, config, generator_a=None, generator_b=None):
        """Initialize two new generators using the config or use pre-trained ones and create Adam optimizers for both
        models."""
        super().__init__(device, config)
        self.epoch_losses = [0., 0.]

        if generator_a is None:
            generator_a_conf = dict(dim_1=config['dim_b'], dim_2=config['dim_a'],
                                    layer_number=config['generator_layers'],
                                    layer_expansion=config['generator_expansion'],
                                    initialize_generator=config['initialize_generator'],
                                    norm=config['gen_norm'],
                                    batch_norm=config['gen_batch_norm'],
                                    activation=config['gen_activation'],
                                    dropout=config['gen_dropout'],
                                    )
            self.generator_a = Generator(generator_a_conf, device)
            self.generator_a.to(device)
        else:
            self.generator_a = generator_a
        if 'optimizer' in config:
            self.optimizer_g_a = OPTIMIZERS[config['optimizer']](self.generator_a.parameters(), config['learning_rate'])
        elif 'optimizer_default' in config:
            if config['optimizer_default'] == 'sgd':
                self.optimizer_g_a = OPTIMIZERS[config['optimizer_default']](self.generator_a.parameters(),
                                                                             config['learning_rate'])
            else:
                self.optimizer_g_a = OPTIMIZERS[config['optimizer_default']](self.generator_a.parameters())
        else:
            self.optimizer_g_a = torch.optim.Adam(self.generator_a.parameters(), config['learning_rate'])

        if generator_b is None:
            generator_b_conf = dict(dim_1=config['dim_a'], dim_2=config['dim_b'],
                                    layer_number=config['generator_layers'],
                                    layer_expansion=config['generator_expansion'],
                                    initialize_generator=config['initialize_generator'],
                                    norm=config['gen_norm'],
                                    batch_norm=config['gen_batch_norm'],
                                    activation=config['gen_activation'],
                                    dropout=config['gen_dropout'],
                                    )
            self.generator_b = Generator(generator_b_conf, device)
            self.generator_b.to(device)
        else:
            self.generator_b = generator_b
        if 'optimizer' in config:
            self.optimizer_g_b = OPTIMIZERS[config['optimizer']](self.generator_b.parameters(), config['learning_rate'])
        elif 'optimizer_default' in config:
            if config['optimizer_default'] == 'sgd':
                self.optimizer_g_b = OPTIMIZERS[config['optimizer_default']](self.generator_b.parameters(),
                                                                             config['learning_rate'])
            else:
                self.optimizer_g_b = OPTIMIZERS[config['optimizer_default']](self.generator_b.parameters())
        else:
            self.optimizer_g_b = torch.optim.Adam(self.generator_b.parameters(), config['learning_rate'])

    def train(self):
        self.generator_a.train()
        self.generator_b.train()

    def eval(self):
        self.generator_a.eval()
        self.generator_b.eval()

    def zero_grad(self):
        self.optimizer_g_a.zero_grad()
        self.optimizer_g_b.zero_grad()

    def optimize_all(self):
        self.optimizer_g_a.step()
        self.optimizer_g_b.step()

    def change_lr(self, factor):
        self.current_lr = self.current_lr * factor
        for param_group in self.optimizer_g_a.param_groups:
            param_group['lr'] = self.current_lr
        for param_group in self.optimizer_g_b.param_groups:
            param_group['lr'] = self.current_lr

    def update_losses_batch(self, *losses):
        loss_g_a, loss_g_b = losses
        self.epoch_losses[0] += loss_g_a
        self.epoch_losses[1] += loss_g_b

    def complete_epoch(self, epoch_metrics):
        self.metrics.append(epoch_metrics + [sum(self.epoch_losses)])
        self.losses.append(self.epoch_losses)
        self.epoch_losses = [0., 0.]

    def print_epoch_info(self):
        print(f"{len(self.metrics)} ### {self.losses[-1][0]:.2f} - {self.losses[-1][1]:.2f} ### {self.metrics[-1]}")

    def copy_model(self):
        self.model_copy = deepcopy(self.generator_a.state_dict()), deepcopy(self.generator_b.state_dict())

    def restore_model(self):
        self.generator_a.load_state_dict(self.model_copy[0])
        self.generator_b.load_state_dict(self.model_copy[1])

    def export_model(self, test_results, description=None):
        if description is None:
            description = f"Regression_{self.config['evaluation']}_{self.config['subset']}"
        export_regression_alignment(description, self.config, self.generator_a, self.generator_b, self.metrics)
        save_alignment_test_results(test_results, description)
        print(f"Saved model to {description}.")

    @classmethod
    def load_model(cls, name, device):
        generator_a, generator_b, config = load_regression_alignment(name, device)
        model = cls(device, config, generator_a, generator_b)
        return model


class CycleGAN(AlignmentModel):
    """This class implements the alignment model for GAN networks with two generators and two discriminators
    (cycle GAN). For description of the implemented functions, refer to the alignment model."""

    def __init__(self, device, config, generator_a=None, generator_b=None, discriminator_a=None, discriminator_b=None):
        """Initialize two new generators and two discriminators from the config or use pre-trained ones and create Adam
        optimizers for all models."""
        super().__init__(device, config)
        self.epoch_losses = [0., 0., 0., 0.]

        if generator_a is None:
            generator_a_conf = dict(dim_1=config['dim_b'], dim_2=config['dim_a'],
                                    layer_number=config['generator_layers'],
                                    layer_expansion=config['generator_expansion'],
                                    initialize_generator=config['initialize_generator'],
                                    norm=config['gen_norm'],
                                    batch_norm=config['gen_batch_norm'],
                                    activation=config['gen_activation'],
                                    dropout=config['gen_dropout'])
            self.generator_a = Generator(generator_a_conf, device)
            self.generator_a.to(device)
        else:
            self.generator_a = generator_a
        if 'optimizer' in config:
            self.optimizer_g_a = OPTIMIZERS[config['optimizer']](self.generator_a.parameters(), config['learning_rate'])
        elif 'optimizer_default' in config:
            if config['optimizer_default'] == 'sgd':
                self.optimizer_g_a = OPTIMIZERS[config['optimizer_default']](self.generator_a.parameters(),
                                                                             config['learning_rate'])
            else:
                self.optimizer_g_a = OPTIMIZERS[config['optimizer_default']](self.generator_a.parameters())
        else:
            self.optimizer_g_a = torch.optim.Adam(self.generator_a.parameters(), config['learning_rate'])

        if generator_b is None:
            generator_b_conf = dict(dim_1=config['dim_a'], dim_2=config['dim_b'],
                                    layer_number=config['generator_layers'],
                                    layer_expansion=config['generator_expansion'],
                                    initialize_generator=config['initialize_generator'],
                                    norm=config['gen_norm'],
                                    batch_norm=config['gen_batch_norm'],
                                    activation=config['gen_activation'],
                                    dropout=config['gen_dropout'])
            self.generator_b = Generator(generator_b_conf, device)
            self.generator_b.to(device)
        else:
            self.generator_b = generator_b
        if 'optimizer' in config:
            self.optimizer_g_b = OPTIMIZERS[config['optimizer']](self.generator_b.parameters(), config['learning_rate'])
        elif 'optimizer_default' in config:
            if config['optimizer_default'] == 'sgd':
                self.optimizer_g_b = OPTIMIZERS[config['optimizer_default']](self.generator_b.parameters(),
                                                                             config['learning_rate'])
            else:
                self.optimizer_g_b = OPTIMIZERS[config['optimizer_default']](self.generator_b.parameters())
        else:
            self.optimizer_g_b = torch.optim.Adam(self.generator_b.parameters(), config['learning_rate'])

        if discriminator_a is None:
            discriminator_a_conf = dict(dim=config['dim_a'], layer_number=config['discriminator_layers'],
                                        layer_expansion=config['discriminator_expansion'],
                                        batch_norm=config['disc_batch_norm'],
                                        activation=config['disc_activation'],
                                        dropout=config['disc_dropout'])
            self.discriminator_a = Discriminator(discriminator_a_conf, device)
            self.discriminator_a.to(device)
        else:
            self.discriminator_a = discriminator_a
        if 'optimizer' in config:
            self.optimizer_d_a = OPTIMIZERS[config['optimizer']](self.discriminator_a.parameters(),
                                                                 config['learning_rate'])
        elif 'optimizer_default' in config:
            if config['optimizer_default'] == 'sgd':
                self.optimizer_d_a = OPTIMIZERS[config['optimizer_default']](self.discriminator_a.parameters(),
                                                                             config['learning_rate'])
            else:
                self.optimizer_d_a = OPTIMIZERS[config['optimizer_default']](self.discriminator_a.parameters())
        else:
            self.optimizer_d_a = torch.optim.Adam(self.discriminator_a.parameters(), config['learning_rate'])

        if discriminator_b is None:
            discriminator_b_conf = dict(dim=config['dim_b'], layer_number=config['discriminator_layers'],
                                        layer_expansion=config['discriminator_expansion'],
                                        batch_norm=config['disc_batch_norm'],
                                        activation=config['disc_activation'],
                                        dropout=config['disc_dropout'])
            self.discriminator_b = Discriminator(discriminator_b_conf, device)
            self.discriminator_b.to(device)
        else:
            self.discriminator_b = discriminator_b
        if 'optimizer' in config:
            self.optimizer_d_b = OPTIMIZERS[config['optimizer']](self.discriminator_b.parameters(),
                                                                 config['learning_rate'])
        elif 'optimizer_default' in config:
            if config['optimizer_default'] == 'sgd':
                self.optimizer_d_b = OPTIMIZERS[config['optimizer_default']](self.discriminator_b.parameters(),
                                                                             config['learning_rate'])
            else:
                self.optimizer_d_b = OPTIMIZERS[config['optimizer_default']](self.discriminator_b.parameters())
        else:
            self.optimizer_d_b = torch.optim.Adam(self.discriminator_b.parameters(), config['learning_rate'])

    def train(self):
        self.generator_a.train()
        self.generator_b.train()
        self.discriminator_a.train()
        self.discriminator_b.train()

    def eval(self):
        self.generator_a.eval()
        self.generator_b.eval()
        self.discriminator_a.eval()
        self.discriminator_b.eval()

    def zero_grad(self):
        self.optimizer_g_a.zero_grad()
        self.optimizer_g_b.zero_grad()
        self.optimizer_d_a.zero_grad()
        self.optimizer_d_b.zero_grad()

    def optimize_all(self):
        self.optimizer_g_a.step()
        self.optimizer_g_b.step()
        self.optimizer_d_a.step()
        self.optimizer_d_b.step()

    def optimize_generator(self):
        """Do the optimization step only for generators (e.g. when training generators and discriminators separately or
        in turns)."""
        self.optimizer_g_a.step()
        self.optimizer_g_b.step()

    def optimize_discriminator(self):
        """Do the optimization step only for discriminators (e.g. when training generators and discriminators separately
        or in turns)."""
        self.optimizer_d_a.step()
        self.optimizer_d_b.step()

    def change_lr(self, factor):
        self.current_lr = self.current_lr * factor
        for param_group in self.optimizer_g_a.param_groups:
            param_group['lr'] = self.current_lr
        for param_group in self.optimizer_g_b.param_groups:
            param_group['lr'] = self.current_lr

    def update_losses_batch(self, *losses):
        loss_g_a, loss_g_b, loss_d_a, loss_d_b = losses
        self.epoch_losses[0] += loss_g_a
        self.epoch_losses[1] += loss_g_b
        self.epoch_losses[2] += loss_d_a
        self.epoch_losses[3] += loss_d_b

    def complete_epoch(self, epoch_metrics):
        self.metrics.append(epoch_metrics + [sum(self.epoch_losses)])
        self.losses.append(self.epoch_losses)
        self.epoch_losses = [0., 0., 0., 0.]

    def print_epoch_info(self):
        print(f"{len(self.metrics)} ### {self.losses[-1][0]:.2f} - {self.losses[-1][1]:.2f} "
              f"- {self.losses[-1][2]:.2f} - {self.losses[-1][3]:.2f} ### {self.metrics[-1]}")

    def copy_model(self):
        self.model_copy = deepcopy(self.generator_a.state_dict()), deepcopy(self.generator_b.state_dict()),\
                          deepcopy(self.discriminator_a.state_dict()), deepcopy(self.discriminator_b.state_dict())

    def restore_model(self):
        self.generator_a.load_state_dict(self.model_copy[0])
        self.generator_b.load_state_dict(self.model_copy[1])
        self.discriminator_a.load_state_dict(self.model_copy[2])
        self.discriminator_b.load_state_dict(self.model_copy[3])

    def export_model(self, test_results, description=None):
        if description is None:
            description = f"CycleGAN_{self.config['evaluation']}_{self.config['subset']}"
        export_cyclegan_alignment(description, self.config, self.generator_a,
                                  self.generator_b, self.discriminator_a, self.discriminator_b, self.metrics)
        save_alignment_test_results(test_results, description)
        print(f"Saved model to directory {description}.")

    @classmethod
    def load_model(cls, name, device):
        generator_a, generator_b, discriminator_a, discriminator_b, config = load_cyclegan_alignment(name, device)
        model = cls(device, config, generator_a, generator_b, discriminator_a, discriminator_b)
        return model
