"""The RotatE model for embedding of kg as defined in https://arxiv.org/pdf/1902.10197.pdf"""

import torch
import torch.autograd
import torch.nn as nn
from numpy import pi
from baseModel import BaseModel


class RotatE(BaseModel):

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

        super().__init__(config, device)

        # Special hyper parameters
        self.gamma = 12.0
        self.epsilon = 2.0
        self.dim = config['dim']

        self.norm = (self.gamma + self.epsilon) / self.dim

        # Use pretrained embeddings
        if entity_embedding is not None and relation_embedding is not None:
            self.e_embedding = nn.Embedding(self.e_num, self.dim)
            self.r_embedding = nn.Embedding(self.r_num, self.dim)

            self.e_embedding.weight.data = entity_embedding
            self.r_embedding.weight.data = relation_embedding

        # Initialize new embeddings
        else:
            self.e_embedding = nn.Embedding(self.e_num, self.dim)  # *2)# 2x dimensionality for real and imaginary parts
            self.r_embedding = nn.Embedding(self.r_num, self.dim)

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
        Calculate the scores of the triples based on the model specific scoring function defined in the RotatE paper.
        :param triples: the triples for which the score is calculated
        :return: the score of the triples
        """
        head_embeddings = self.e_embedding(triples[:, 0:1])
        relation_embeddings = self.r_embedding(triples[:, 1:2])
        tail_embeddings = self.e_embedding(triples[:, 2:3])

        head_re, head_im = torch.chunk(head_embeddings, 2, dim=-1)
        tail_re, tail_im = torch.chunk(tail_embeddings, 2, dim=-1)

        relation_normed = relation_embeddings / (self.norm / pi)
        # relation_re = torch.cos(relation_normed)
        # relation_im = torch.sin(relation_normed)
        relation_re, relation_im = torch.chunk(relation_normed, 2, dim=-1)

        score_re = head_re * relation_re - head_im * relation_im - tail_re
        score_im = head_re * relation_im + head_im * relation_re - tail_im
        score = torch.stack([score_re, score_im], dim=0)
        score = torch.norm(score, dim=0, p=2)  # equals p = "fro" which is default
        score = self.gamma - score.sum(dim=2)
        return score
