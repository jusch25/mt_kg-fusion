"""This abstract class provides basic attributes and functions for kg embedding models."""

import numpy as np
import torch
import torch.autograd
import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):

    def __init__(self, config, device):
        """
        Assign model parameters taken from the config.
        :param config: a dictionary with model parameters like learning rate, batch size, epoch number (see BASE_CONIFG
        in constants)
        :param device: torch device on which the model executed (cuda or cpu)
        """
        super().__init__()

        self.config = config

        # hyper parameter
        self.margin = self.config['margin']
        self.embedding_norm = self.config['embedding_norm']
        self.distance_norm = self.config['distance_norm']

        # basic parameters
        self.e_num = self.config['e_num']
        self.r_num = self.config['r_num']
        self.dim = self.config['dim']
        self.device = device

        self.criterion = nn.MarginRankingLoss(margin=self.margin, reduction='mean')

    def compute_loss(self, positive_scores, negative_scores):
        """
        Calculate the loss of the embeddings based on the loss criterion (e.g. MarginRankingLoss)
        :param positive_scores: scores of the training triples
        :param negative_scores: scores of the corrupted triples
        :return: the loss used for backpropagation
        """
        y = torch.ones(positive_scores.shape[0], dtype=torch.float, device=self.device) * -1
        loss = self.criterion(positive_scores, negative_scores, y)
        return loss

    def replace_entity_embedding(self, embedding):
        """Replace the entity embedding of this model with a pre-trained embedding or an embedding generated from cycle
        GAN."""
        self.e_embedding.weight.data = embedding

    def replace_relation_embedding(self, embedding):
        """Replace the relation embedding of this model with a pre-trained embedding or an embedding generated from cycle
        GAN."""
        self.r_embedding.weight.data = embedding

    @abstractmethod
    def forward(self, train_pos, train_neg):
        pass

    @abstractmethod
    def score_triples(self, triples):
        pass
