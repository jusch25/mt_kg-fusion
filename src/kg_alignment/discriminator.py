"""The class for the discriminator model for a cycle GAN architecture."""

import torch.nn as nn
import torch
import typing
from constants import ACTIVATIONS


class Discriminator(nn.Module):
    """A discriminator is a pytorch neural network, a sequential model with multiple linear layers starting with the
    size of the input embedding dimension and ending with a binary output. The output is the classification result of
    the discriminator: if 0 then the input should have been generated, if 1 then the input is an original embedding. The
    discriminator is trained to classify the training and the generated data as accurate as possible."""

    def __init__(self, config, device):
        """Initialize a new discriminator with the parameters in the configuration dictionary and store the device used
        for training."""
        super().__init__()
        self.device = device
        self.config = config
        input_size = self.config['dim']
        self.network = nn.Sequential()
        # Create the model
        self.__create_sequential(config['layer_number'], config['layer_expansion'], input_size)
        self.criterion = nn.MSELoss()

    # Necessary so that pycharm recognizes this as pytorch module (https://github.com/pytorch/pytorch/issues/24326)
    def __call__(self, *args, **kwargs) -> typing.Any:
        return super().__call__(*args, **kwargs)

    def forward(self, train):
        """The train step is simply to feed the input to the sequential network and return the output for loss
        calculations."""
        return self.network(train).view(-1,)

    def score_gen(self, generated):
        """The score of generated data for the discriminator is the mean squared error loss of its classification and 0
        (the class value)."""
        return self.criterion(generated, torch.zeros(generated.shape[0], device=self.device))

    def score_train(self, original):
        """The score of original data (meaning that it was not generated) is the mean squared error loss of its
        classification and 1 (the class value)."""
        return self.criterion(original, torch.ones(original.shape[0], device=self.device))

    def decide(self, data):
        """Use the discriminator network to classify embeddings, but unlike the forward function the results are
        detached to remove gradients and the discriminator is not trained. Used for evaluation or when training the
        generator."""
        with torch.no_grad():
            result = self.network(data)
            return result.detach().view(-1,)

    def __create_sequential(self, layer_number, layer_expansion, dim):
        """
        Create the sequential discriminator model (pytorch module) using the parameters from the configuration.
        :param layer_number: the layer number specifies the number of linear layers of the network; if 0 then the input
        is directly classified to 0 or 1, else the layer number is the number of hidden layers
        :param layer_expansion: if <=1: the size of the hidden layers is constantly reduced (from embedding dimension to
        binary output), else: for the first half, the size of the hidden layers is constantly increasing by the
        expansion factor, then in the second half the hidden layers are constantly decreasing to match the binary output
        :param dim: the dimensionality of the input embeddings and the input size of the first linear layer
        """
        if layer_expansion > 1 and layer_number > 1:
            increasing_layers = (layer_number + 1) // 2
            decreasing_layers = layer_number - increasing_layers + 1
            layer_sizes = [1 + (layer_expansion - 1) * i / increasing_layers for i in range(0, increasing_layers + 1)]
            layer_sizes += [layer_expansion * i / decreasing_layers for i in reversed(range(1, decreasing_layers))]
        elif layer_number > 0:
            layer_sizes = [i / (layer_number + 1) for i in reversed(range(1, layer_number + 2))]
        else:
            layer_sizes = [1]
        counter = 1
        for i in range(len(layer_sizes) - 1):
            self.__add_modules(counter, int(layer_sizes[i] * dim), int(layer_sizes[i + 1] * dim))
            counter += 1
        self.network.add_module("linear_" + str(counter), nn.Linear(int(layer_sizes[-1] * dim), 1))

    def __add_modules(self, counter, input_size, output_size):
        """
        Depending on the configuration, several layers are added after every linear layer of the sequential model
        (batch norm layer, activation layer, dropout layer).
        :param counter: increasing number to name the layers
        :param input_size: input size of linear layer
        :param output_size: output size of linear layer and size of the other layers
        """
        self.network.add_module("linear_" + str(counter), nn.Linear(input_size, output_size))
        if self.config['batch_norm']:
            self.network.add_module("batch-norm_" + str(counter), nn.BatchNorm1d(output_size))
        if self.config['activation']:
            if 'activation_function' in self.config:
                if self.config['activation_function'] == 'leakyrelu2':
                    self.network.add_module("activation_" + str(counter),
                                            ACTIVATIONS[self.config['activation_function']](negative_slope=0.1))
                else:
                    self.network.add_module("activation_" + str(counter),
                                            ACTIVATIONS[self.config['activation_function']]())
            else:
                self.network.add_module("activation_" + str(counter), nn.LeakyReLU(negative_slope=0.1))
        if self.config['dropout']:
            if 'dropout_rate' in self.config:
                self.network.add_module("dropout_" + str(counter), nn.Dropout(p=self.config['dropout_rate']))
            else:
                self.network.add_module("dropout_" + str(counter), nn.Dropout(p=0.3))
