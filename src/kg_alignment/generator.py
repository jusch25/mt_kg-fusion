"""The class for the generator model for a regression or as port of a cycle GAN architecture."""

import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional
import typing
from util import Normalization
from constants import ACTIVATIONS


class Generator(nn.Module):
    """A generator is a pytorch neural network, a sequential model with multiple linear layers starting with the
    size of the input embedding dimension and ending with the output embedding dimension. The output is a transformation
    based on the input embedding. The generator is trained to transform embeddings from vector space A (defined by the
    coordinates in the vectors space) to the vector space B so that they are as close as possible to the aligned
    embeddings."""

    def __init__(self, config, device):
        """Initialize a new discriminator with the parameters in the configuration dictionary and store the device used
        for training."""
        super().__init__()
        self.device = device
        self.config = config
        self.network = nn.Sequential()
        self.__create_sequential(config['layer_number'], config['layer_expansion'], config['dim_1'], config['dim_2'],
                                 config['initialize_generator'])
        self.criterion = nn.MSELoss()
        self.margin_loss = nn.MarginRankingLoss(margin=0)

    # Necessary so that pycharm recognizes this as pytorch module (https://github.com/pytorch/pytorch/issues/24326)
    def __call__(self, *args, **kwargs) -> typing.Any:
        return super().__call__(*args, **kwargs)

    def forward(self, train):
        """The train step is simply to feed the input to the sequential network and return the output for loss
        calculations."""
        generated = self.network(train)
        return generated

    def score_gan(self, classified_generated):
        """The score of generated data for the generator is the mean squared error loss of its classification and 1
        (the class value for original data). Therefore the generator and discriminator play against each other and are
        optimized to reduce the complimentary loss."""
        return self.criterion(classified_generated, torch.ones(classified_generated.shape[0], device=self.device))

    @staticmethod
    def score_cyclic(train_1, cyclic_1, train_2, cyclic_2):
        """The cycle score is the vector difference (or the distance) between the original data and the cyclic generated
        data (first transformed into the other space and then back). The cyclic score ensures that the generators do not
        only learn to adapt the features of the other vector space but also to keep the original features when
        transforming."""
        return torch.mean(torch.abs(train_1 - cyclic_1)) + torch.mean(torch.abs(train_2 - cyclic_2))

    def score_unaligned(self, gen, gen_target):
        """The unaligned score is the vector difference (or the distance) between the generated the data and its
        provisional alignment chosen by finding the nearest embedding of the original data and using its alignment.
        Refer to the code in entity_alingment.py/__get_closest_neighbour for details."""
        negative_factor = 10  # how many other random embeddings are used as negative samples for the margin loss
        labels = torch.ones(gen.shape[0] * negative_factor, dtype=torch.float, device=gen.device) * -1
        pos_scores = torch.mean(torch.abs(gen - gen_target), dim=1).repeat(1, negative_factor).reshape(-1,)
        neg_left = gen.repeat(1, negative_factor).reshape(-1, gen.shape[1])
        neg_right = gen_target.repeat(negative_factor, 1)
        neg_scores = torch.mean(torch.abs(neg_left - neg_right), dim=1)
        return self.margin_loss(pos_scores, neg_scores, labels)

    @staticmethod
    def score_generated(train, gen):
        """The generated score is the vector difference (or the distance) between the generated train data and the
        aligned embedding from the other vector space."""
        return torch.mean(torch.abs(train - gen))

    def generate(self, data):
        """Use the generator network to transform embeddings, but unlike the forward function the results are detached
        to remove gradients and the generator is not trained. Used for evaluation or when training the discriminator."""
        with torch.no_grad():
            result = self.network(data)
            return result.detach()

    def __create_sequential(self, layer_number, layer_expansion, dim_1, dim_2, initialized):
        """
        Create the sequential generator model (pytorch module) using the parameters from the configuration.
        :param layer_number: the layer number specifies the number of linear layers of the network; if 0 then the input
        is directly transformed with one linear layer, else the layer number is the number of hidden layers
        :param layer_expansion: if <=1: the size of the hidden layers is constantly changed from dim_1 to dim_2,
        else: for the first half, the size of the hidden layers is constantly increasing by the expansion factor, then
        in the second half the hidden layers are constantly decreasing to match dim_2
        :param dim_1: the dimensionality of the input embeddings and the input size of the first linear layer
        :param dim_2: the dimensionality of the output embeddings and the output size of the final linear layer
        :param initialized: if True, then the first layer is initialized with the transformation matrix between the
        vector spaces (if more than one linear layer is used (layer_number>0) then the other layers must be adapted
        """
        if layer_number % 2 == 1:
            intermediate_layer = False
            increasing_layers = (layer_number + 1) // 2
        else:
            intermediate_layer = layer_number > 0
            increasing_layers = layer_number // 2
        layer_sizes = [1] + [layer_expansion * i for i in range(1, increasing_layers + 1)]
        counter = 1
        if initialized and layer_number > 0:
            self.__add_modules(0, dim_1, dim_2)
            self.__add_modules(1, dim_2, int(layer_sizes[1] * dim_1))
            counter += 1
            start = 1
        else:
            start = 0
        for i in range(start, increasing_layers):
            self.__add_modules(counter, int(layer_sizes[i] * dim_1), int(layer_sizes[i + 1] * dim_1))
            counter += 1
        if intermediate_layer:
            self.__add_modules(counter, int(layer_sizes[-1] * dim_1), int(layer_sizes[-1] * dim_2))
            counter += 1
            decrease_start = 1
        elif layer_number > 2:
            self.__add_modules(counter, int(layer_sizes[-1] * dim_1), int(layer_sizes[-2] * dim_2))
            counter += 1
            decrease_start = 2
        else:
            decrease_start = 1
        for i in range(decrease_start, increasing_layers):
            self.__add_modules(counter, int(layer_sizes[-i] * dim_2), int(layer_sizes[-(i + 1)] * dim_2))
            counter += 1
        if layer_number > 1:
            self.network.add_module("linear_" + str(counter), nn.Linear(int(layer_sizes[1] * dim_2),
                                                                        int(layer_sizes[0] * dim_2)))
        elif layer_number == 1:
            self.network.add_module("linear_" + str(counter), nn.Linear(int(layer_sizes[1] * dim_1),
                                                                        int(layer_sizes[0] * dim_2)))
        else:
            self.network.add_module("linear_" + str(counter), nn.Linear(dim_1, dim_2))
        if self.config['norm']:
            self.network.add_module("normalization", Normalization(p=2, dim=1))

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
