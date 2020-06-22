"""The TransE model for embedding of kg as defined in
https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

import numpy as np
import torch
import torch.autograd
import torch.nn as nn
from baseModel import BaseModel


class TransE(BaseModel):

    def __init__(self, config, device, entity_embedding=None, relation_embedding=None):
        """
        Initialize model and assign parameters in parent class based on the chosen configurations. Use pre-trained
        embeddings if provided.
        :param config: a dictionary that defines the model attributes
        :param device: the torch device on which the model is executed
        :param entity_embedding: an optional pre-trained embedding of the entities
        :param relation_embedding: an optional pre-trained embedding of the relations
        """
        # Adapt config for pre-trained embeddings
        if entity_embedding is not None and relation_embedding is not None:
            config['dim'] = entity_embedding.shape[1]
            config['e_num'] = entity_embedding.shape[0]
            config['r_num'] = relation_embedding.shape[0]

        # Assign variables in parent class constructor
        super().__init__(config, device)

        # Create Embeddings
        self.e_embedding = nn.Embedding(self.e_num, self.dim)
        self.r_embedding = nn.Embedding(self.r_num, self.dim)

        # Use pretrained embeddings
        if entity_embedding is not None and relation_embedding is not None:
            self.e_embedding.weight.data = entity_embedding
            self.r_embedding.weight.data = relation_embedding
        # Initialize new embeddings
        else:
            self.norm = 6 / np.sqrt(self.dim)
            nn.init.uniform_(
                self.e_embedding.weight.data,
                a=-self.norm,
                b=+self.norm,
            )
            nn.init.uniform_(
                self.r_embedding.weight.data,
                a=-self.norm,
                b=+self.norm,
            )
            norms = torch.norm(self.r_embedding.weight, p=self.embedding_norm, dim=1).data
            self.r_embedding.weight.data = torch.div(self.r_embedding.weight.data,
                                                     norms.view(self.r_num, 1).expand_as(self.r_embedding.weight))

    def forward(self, train_pos, train_neg):
        """
        Calculate the loss of the training data for model update (called in each training iteration).
        :param train_pos: true triples from the training set
        :param train_neg: corrupted triples generated from the true triples
        :return: the loss of this training iteration
        """
        norms = torch.norm(self.e_embedding.weight, p=self.embedding_norm, dim=1).data
        self.e_embedding.weight.data = torch.div(self.e_embedding.weight.data,
                                                 norms.view(self.e_num, 1).expand_as(self.e_embedding.weight))

        positive_scores = self.score_triples(train_pos)
        negative_scores = self.score_triples(train_neg)
        loss = self.compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def score_triples(self, triples):
        """
        Calculate the scores of the triples based on the model specific scoring function defined in the TransE paper.
        :param triples: the triples for which the score is calculated
        :return: the score of the triples
        """
        head_embeddings = self.e_embedding(triples[:, 0:1]).view(-1, self.dim)
        relation_embeddings = self.r_embedding(triples[:, 1:2]).view(-1, self.dim)
        tail_embeddings = self.e_embedding(triples[:, 2:3]).view(-1, self.dim)
        sum_res = head_embeddings + relation_embeddings - tail_embeddings
        distances = torch.norm(sum_res, dim=1, p=self.distance_norm).view(size=(-1,))
        return distances
