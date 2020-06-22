"""This file contains functions to evaluate kg models and entity alignments to report metrics like MR, MRR and
Hits@k."""

import numpy as np
import torch
from util import hash_triples
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


def calculate_roc_auc_score(model, evaluation_data, train_data, opt_data):
    """Calculate the ROC-AUC score. See __calculate_classification_metrics for details"""
    return __calculate_classification_metrics(model, "roc-auc", evaluation_data, train_data, opt_data)


def calculate_average_precision_score(model, evaluation_data, train_data, opt_data):
    """Calculate the Average Precision score. See __calculate_classification_metrics for details"""
    return __calculate_classification_metrics(model, "avg-precision", evaluation_data, train_data, opt_data)


def __calculate_classification_metrics(model, metric, evaluation_data, train_data, opt_data):
    """
    Calculate classification metrics (ROC-AUC or Average Precision score from sklearn)
    :param model: the trained model for scoring the evaluation data
    :param metric: use one of the precision-based metrics, either "roc-auc" or "avg-precision"
    :param evaluation_data: the encoded triples that are used for evaluation (either validation or test data)
    :param train_data: the encoded triples that were used for training the model
    :param opt_data: the rest of the triples (either validation or test data)
    :return: the average metric score for the evaluation data (either average ROC-AUC or average Average Precision)
    """
    model.eval()
    with torch.no_grad():
        all_triples = np.concatenate((evaluation_data, train_data, opt_data), axis=0)
        all_hashed = np.apply_along_axis(hash_triples, 1, all_triples)

        entity_number = model.e_num
        all_entities = np.arange(entity_number).reshape(-1, 1)
        metric_sum = 0
        if model.device.type == "cuda":
            evaluation_data = tqdm(evaluation_data)
        for x in evaluation_data:
            corrupted_subjects_tail = np.repeat([x[1:]], entity_number, axis=0)
            corrupted_subjects = np.concatenate([all_entities, corrupted_subjects_tail], axis=1)
            corrupted_objects_head = np.repeat([x[:2]], entity_number, axis=0)
            corrupted_objects = np.concatenate([corrupted_objects_head, all_entities], axis=1)
            corrupted_triples = np.concatenate([corrupted_subjects, corrupted_objects], axis=0)
            # Remove duplicate occurrence of test triple x
            mask_index = np.nonzero(all_entities == x[2])[0][0]
            mask = np.ones(len(corrupted_triples), dtype=bool)
            mask[entity_number + mask_index] = False
            data = corrupted_triples[mask]

            data_hashed = np.apply_along_axis(hash_triples, 1, data)
            # existing triples should be close to 0, corrupted triples should be close to 1
            target_values = np.in1d(data_hashed, all_hashed, invert=True) * 1

            data = torch.tensor(data, dtype=torch.long, device=model.device)
            scores = model.score_triples(data).detach().flatten().cpu().numpy()
            scores = (scores - scores.min()) / (scores.max() - scores.min())  # normalize to values between 0 and 1

            if metric == "roc-auc":
                metric_sum += roc_auc_score(target_values, scores)
            elif metric == "avg-precision":
                metric_sum += average_precision_score(target_values, scores)
        return metric_sum / len(evaluation_data)


def calculate_rank_metrics(model, evaluation_data, train_data, opt_data, filtered=False):
    """
    Calculate either the filtered or unfiltered rank of each evaluation triple and use it for the popular metrics "Mean
    Rank", "Mean Reciprocal Rank" and "Hits@k".
    :param model: the trained model for scoring the evaluation data
    :param evaluation_data: the encoded triples that are used for evaluation (either validation or test data)
    :param train_data: the encoded triples that were used for training the model (only used when corrupted triples are
    filtered)
    :param opt_data: the rest of the triples (either validation or test data, only used when corrupted triples are
    filtered
    :param filtered: if True then remove all known valid triples that occur in either train, validation or test data
    from the set of corrupted triples before calculating the metrics (since valid triples may be ranked higher by the
    model than an evaluation triple)
    :return: a list of the average metrics for all evaluation data
    """
    model.eval()
    with torch.no_grad():
        all_hashed = None
        if filtered:
            all_triples = np.concatenate((train_data, opt_data, evaluation_data), axis=0)
            # hash triples to compare them efficiently (not possible with torch tensors)
            all_hashed = np.apply_along_axis(hash_triples, 1, all_triples)

        entity_number = model.e_num
        metric_results = MetricsContainer(len(evaluation_data))
        all_entities = np.arange(entity_number).reshape(-1, 1)
        if model.device.type == "cuda":
            evaluation_data = tqdm(evaluation_data)  # show progress bar for large datasets
        for x in evaluation_data:
            # corrupt evaluation triples by replacing both head and tail with all other entities
            corrupted_subjects_tail = np.repeat([x[1:]], entity_number, axis=0)
            corrupted_subjects = np.concatenate([all_entities, corrupted_subjects_tail], axis=1)
            corrupted_objects_head = np.repeat([x[:2]], entity_number, axis=0)
            corrupted_objects = np.concatenate([corrupted_objects_head, all_entities], axis=1)
            corrupted_triples = np.concatenate([corrupted_subjects, corrupted_objects], axis=0)

            if all_hashed is not None:
                data_hashed = np.apply_along_axis(hash_triples, 1, corrupted_triples)
                valid_indices = np.in1d(data_hashed, all_hashed, invert=True)
                data = corrupted_triples[valid_indices]
                # Add the test triple x which was removed by the filter
                data = np.append(data, [x], axis=0)
                index = len(data)-1
            else:
                # Remove duplicate occurrence of test triple x
                mask_index = np.nonzero(all_entities == x[2])[0][0]
                mask = np.ones(len(corrupted_triples), dtype=bool)
                mask[entity_number + mask_index] = False
                data = corrupted_triples[mask]
                index = np.nonzero(all_entities == x[0])[0][0]

            # score the evaluation triple and the corrupted triples and calculate the rank of the evaluation triple
            data = torch.tensor(data, dtype=torch.long, device=model.device)
            scores = model.score_triples(data).detach().flatten()
            _, indices = torch.sort(scores, descending=False)
            indices = indices.cpu().numpy()
            rank = np.nonzero(indices == index)[0][0] + 1
            metric_results.update(rank)
        return metric_results.get_results()


def calculate_all_cycle_gan_metrics(nat_a, nat_b, gen_a, gen_b, cyc_a, cyc_b, combined_results=True):
    """
    Calculate the evaluation metrics by aligning true data from an embedding space with generated data. There are four
    variants (see code below) that are evaluated.
    :param nat_a: the true data from embeddings space A
    :param nat_b: the true data from embeddings space B
    :param gen_a: the generated data for embedding space A, obtained from nat_b
    :param gen_b: the generated data for embedding space B, obtained from nat_a
    :param cyc_a: the cyclic generated data for embedding space A, obtained from gen_b
    :param cyc_b: the cyclic generated data for embedding space B, obtained from gen_a
    :param combined_results: if True, then the metrics of each variant are combined by averaging, else they are packed
    into a list
    :return: either the average of the metrics for all variants, or a list of the single metrics
    """
    score1 = calculate_single_cycle_gan_metrics(gen_a, nat_a)  # space A: generated vs. true data
    score2 = calculate_single_cycle_gan_metrics(gen_b, nat_b)  # space B: generated vs. true data
    score3 = calculate_single_cycle_gan_metrics(cyc_a, nat_a)  # space A: cyclic generated vs. true data
    score4 = calculate_single_cycle_gan_metrics(cyc_b, nat_b)  # space B: cyclic generated vs. true data
    if not combined_results:
        print(score1)
        print(score2)
        print(score3)
        print(score4)
        return [score1, score2, score3, score4]
    # mean_score = [(w + x + y + z) / 4 for w, x, y, z in zip(score1, score2, score3, score4)]
    mean_score = [(x + y) / 2 for x, y in zip(score1, score2)]
    return mean_score


def calculate_single_cycle_gan_metrics(source, target):
    """
    Calculate popular metrics (MR, MRR, Hits@k) for the alignment between existing (target) and generated (source)
    embeddings.
    :param source: the generated embeddings from the source kg which should match the target embeddings
    :param target: the reference embeddings from the target kg
    :return: a list of the average metrics for all embeddings
    """
    metric_results = MetricsContainer(len(source))
    for i, x in enumerate(source):
        scores = torch.mean(torch.abs(x.view(1, -1) - target), dim=1)
        _, indices = torch.sort(scores, descending=False)
        indices = indices.cpu().numpy()
        rank = np.nonzero(indices == i)[0][0] + 1
        metric_results.update(rank)
    return metric_results.get_results()


class MetricsContainer:
    """A container to store the rank metrics obtained from evaluation results."""

    def __init__(self, test_size):
        """
        Initialize the metrics.
        :param test_size: the size of the evaluated data to calculate the mean values
        """
        self.test_size = test_size

        self.mr = 0
        self.mrr = 0
        self.hits1 = 0
        self.hits3 = 0
        self.hits5 = 0
        self.hits10 = 0
        self.hits1p = 0

    def update(self, rank):
        """
        Add the metric values of a single evaluation triple.
        :param rank: the rank of the evaluation triple which is the basis for calculating the metrics
        """
        # calculate MR and MRR
        self.mr += rank
        self.mrr += 1 / rank
        # calculate Hits@k
        if rank <= 1:
            self.hits1 += 1
            self.hits3 += 1
            self.hits5 += 1
            self.hits10 += 1
        elif rank <= 3:
            self.hits3 += 1
            self.hits5 += 1
            self.hits10 += 1
        elif rank <= 5:
            self.hits5 += 1
            self.hits10 += 1
        elif rank <= 10:
            self.hits10 += 1

    def get_results(self):
        """
        After evaluating all triples, this function is called to calculate and return the average metric values.
        :return: the average metric values as a list
        """
        result = [round(self.mr / self.test_size, 1), round(self.mrr / self.test_size, 3),
                  round(self.hits1 / self.test_size, 3), round(self.hits3 / self.test_size, 3),
                  round(self.hits5 / self.test_size, 3), round(self.hits10 / self.test_size, 3)]
        return result
