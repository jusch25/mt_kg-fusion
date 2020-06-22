"""A collection of various functions used by other classes or scripts."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from constants import METRICS, CYCLE_GAN_CONFIG, REGRESSION_CONFIG
import re


def get_mapping_from_triples(triples):
    """
    Calculates the mapping from kg entities and relations to integers.
    :param triples: the kg for which the mapping is created
    :return: the mapping entity->int and relation->int in dictionary format as a tuple
    """
    entities_all = np.concatenate([triples[:, 0], triples[:, 2]], axis=0)
    relations_all = triples[:, 1]
    entities_unique = np.unique(entities_all)
    relations_unique = np.unique(relations_all)
    entity_to_id = {v: k for k, v in enumerate(entities_unique)}
    relation_to_id = {v: k for k, v in enumerate(relations_unique)}
    return entity_to_id, relation_to_id


def encode_all(train, valid, test, entity_to_id, relation_to_id):
    """
    Encode all data using the provided mapping dictionaries.
    :param train: training triples
    :param valid: validation triples
    :param test: test triples
    :param entity_to_id: dictionary for mapping kg entities to int
    :param relation_to_id: dictionary for mapping kg relations to int
    :return: the encoded data as a tuple
    """
    train = encode_triples(train, entity_to_id, relation_to_id)
    valid = encode_triples(valid, entity_to_id, relation_to_id)
    test = encode_triples(test, entity_to_id, relation_to_id)
    return train, valid, test


def encode_triples(triples, entity_to_id, relation_to_id):
    """
    Map the triples (s, p, o) to ints (int, int, int) by using the provided dictionaries. Triples that contain entities
    or relations that are not in the dictionaries are removed.
    :param triples: the triples that are to be encoded
    :param entity_to_id: dictionary for mapping kg entities to int
    :param relation_to_id: dictionary for mapping kg relations to int
    :return: the encoded triples
    """
    # Remove triples with unknown entites or relations
    triples = clean_data(triples, entity_to_id, relation_to_id, True)
    entity_map = np.vectorize(lambda x: entity_to_id[x])
    relation_map = np.vectorize(lambda x: relation_to_id[x])
    return np.concatenate([np.reshape(entity_map(triples[:, 0]), (-1, 1)),
                           np.reshape(relation_map(triples[:, 1]), (-1, 1)),
                           np.reshape(entity_map(triples[:, 2]), (-1, 1))], axis=1)


def decode_triples(data, entity_to_id, relation_to_id):
    """
    Decode the encoded integers back to kg entities and relations by reversing the mapping dictionaries.
    :param data: the encoded triples
    :param entity_to_id: dictionary for mapping kg entities to int
    :param relation_to_id: dictionary for mapping kg relations to int
    :return: the decoded triples
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    # reverse mappings
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    entity_map = np.vectorize(lambda x: id_to_entity[x])
    relation_map = np.vectorize(lambda x: id_to_relation[x])
    return np.concatenate([np.reshape(entity_map(data[:, 0]), (-1, 1)),
                           np.reshape(relation_map(data[:, 1]), (-1, 1)),
                           np.reshape(entity_map(data[:, 2]), (-1, 1))], axis=1)


def clean_data(data, label_to_id_1, label_to_id_2, triple_encoding):
    """
    Remove data that contains entries that are not in the mappings (and thus the model is not trained on them).
    :param data: the triples that have to be cleaned
    :param label_to_id_1: dictionary that contains the valid entities (or source entities for alginments)
    :param label_to_id_2: dictionary that contains the valid relations (or target entities for alignments)
    :param triple_encoding: if True then the data are triples with shape (s, p, o) else the data is an alignment of
    shape (e1, e2)
    :return: the list of filtered data
    """
    keys_1 = np.array(list(label_to_id_1.keys()))
    keys_2 = np.array(list(label_to_id_2.keys()))
    valid_1 = np.in1d(data[:, 0], keys_1)  # subjects for triples, entities_a for alignments
    valid_2 = np.in1d(data[:, 1], keys_2)  # relations for triples, entities_b for alignments
    if not triple_encoding:
        valid_alignments = np.logical_and.reduce((valid_1, valid_2))
        return data[valid_alignments]
    valid_3 = np.in1d(data[:, 2], keys_1)  # objects for triples
    valid_triples = np.logical_and.reduce((valid_1, valid_2, valid_3))
    return data[valid_triples]


def natural_sort(l):
    """Sorts an array in the natural number order, e.g. 0,1,2,10,100 instead of 0,1,10,100,11,2"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def encode_alignment(data, label_to_id_1, label_to_id_2, triple_encoding):
    """
    The DBP15k datasets are already encoded but the ids are not steady (so the maximum id is higher than the number of
    ids. This function maps the sorted ids to their position (enumeration) to a steady encoding.
    :param data: the encoded DBP15k triples
    :param label_to_id_1: first dictionary for re-encoding ids (from kg entities or source alignment)
    :param label_to_id_2: second dictionary for re-encoding relation ids (from kg relations or target alignment)
    :param triple_encoding: if True then the data are triples with shape (s, p, o) else the data is an alignment of
    shape (e1, e2)
    :return: the re-encoded triples
    """
    map_1 = np.vectorize(lambda x: label_to_id_1[x])
    map_2 = np.vectorize(lambda x: label_to_id_2[x])
    if triple_encoding:
        # the data are triples with shape (s, p, o)
        return np.concatenate([np.reshape(map_1(data[:, 0]), (-1, 1)),
                               np.reshape(map_2(data[:, 1]), (-1, 1)),
                               np.reshape(map_1(data[:, 2]), (-1, 1))], axis=1)
    # else the data is an alignment of shape (e1, e2) that must be cleaned
    data_cleaned = clean_data(data, label_to_id_1, label_to_id_2, False)
    return np.concatenate([np.reshape(map_1(data_cleaned[:, 0]), (-1, 1)),
                           np.reshape(map_2(data_cleaned[:, 1]), (-1, 1))], axis=1)


def embedding_partitioning(embeddings, train_alignment_ids, test_alignment_ids, validation_ids):
    """
    Takes the list of aligned entities to partition the embeddings into aligned and unaligned sets.
    :param embeddings: a matrix containing the embeddings
    :param train_alignment_ids: the list of aligned entities (encoded as ids) for training
    :param test_alignment_ids: the list of aligned entities (encoded as ids) for testing
    :param validation_ids: a random part of the original training alignments used for validation
    :return: the aligned embeddings for training, validation and testing as well as all other embeddings that are not
    aligned
    """
    aligned_ids = np.concatenate([train_alignment_ids, test_alignment_ids, validation_ids])
    rest_ids = np.setdiff1d(np.arange(len(embeddings)), aligned_ids)
    train = embeddings[train_alignment_ids]
    test = embeddings[test_alignment_ids]
    valid = embeddings[validation_ids]
    rest = embeddings[rest_ids]
    return train, valid, test, rest


def hash_triples(triples):
    """Hash the data with pythons native hash function. Used to compare triples efficiently."""
    return hash(tuple(triples))


def torch_isin(arr1, arr2, invert=False):
    """
    An implementation of the numpy isin() function (can be used for torch tensors).
    :param arr1: for each element in this array: test if it exists in the other array
    :param arr2: reference array
    :param invert: flip values (0 <=> 1 and True <=> False respectively)
    :return: a boolean array with the same length as arr1 with 1 for elements of arr1 that exist in arr2 and 0 otherwise
    """
    # noinspection PyUnresolvedReferences
    result = (arr1[..., None] == arr2).any(-1)
    if invert:
        return result.__invert__()
    return result


def split_dataset(data, train_size, valid_size, test_size):
    """
    Split an array into train, valid and test arrays based on the provided proportions.
    :param data: the array that has to be splitted
    :param train_size: proportion of the train data
    :param valid_size: proportion of the validation data
    :param test_size: proportion of the test data
    :return: the split data as a tuple
    """
    if train_size + valid_size + test_size != 1:
        size_sum = train_size + valid_size + test_size
        train_size = train_size / size_sum
        valid_size = valid_size / size_sum

    data_size = data.shape[0]
    train_size = int(data_size * train_size)
    valid_size = int(data_size * valid_size)

    return data[:train_size], data[train_size: train_size+valid_size], data[train_size+valid_size:]


def clean_config(config, gan_mode):
    """Remove unnecessary dictionary entries from the config."""
    if gan_mode:
        new_dict = {k: v for k, v in config.items() if k in CYCLE_GAN_CONFIG}
    else:
        new_dict = {k: v for k, v in config.items() if k in REGRESSION_CONFIG}
    return new_dict


class Plotter:
    """This class shows the validation metrics as a dynamic pyplot during training."""

    def __init__(self):
        """Create the pyplot figure and empty data arrays and show the plot."""
        self.fig = plt.figure()
        self.data = [[] for _ in range(7)]
        self.plots = []
        for i in range(7):
            self.plots.append(self.fig.add_subplot(2, 4, i+1))
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def update_plot(self, update):
        """Update the plots with the new data by appending the data to the arrays and redrawing the plots."""
        for i, x in enumerate(update):
            self.data[i].append(x)
            self.plots[i].clear()
            self.plots[i].plot(self.data[i])
            self.plots[i].set_ylim(bottom=0)
            self.plots[i].set_title(METRICS[i])
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.001)


class Normalization(torch.nn.Module):
    """Workaround to use vector normalization in a sequential model."""

    def __init__(self, p=2, dim=1):
        super(Normalization, self).__init__()
        self.norm = torch.nn.functional.normalize
        self.p = p
        self.dim = dim

    def forward(self, x):
        x = self.norm(x, p=self.p, dim=self.dim)
        return x
