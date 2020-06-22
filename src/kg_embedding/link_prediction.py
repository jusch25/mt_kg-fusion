"""The basic pipeline for training kg embedding models."""

from util import Plotter, split_dataset, encode_all
from evaluate import calculate_rank_metrics
from input_output import *
from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm
from os import path, remove


def run_train(config, device, dataset, align_dataset, save_results, align_mode, plot=False):
    """
    Train a new kg embedding model.
    :param config: the parameters for training
    :param device: the torch device on which the training is processed
    :param dataset: the dataset name or id
    :param align_dataset: 0 for DBP15k, 1 for WK3l-15k (only for align mode)
    :param save_results: if True, save model, training history and evalutation results
    :param align_mode: if True, then alignment datasets are used and embeddings are stored into files
    :param plot: shows a plot with validation metrics during training
    """
    # Load data
    if align_mode:
        dataset_name = dataset[:-1]
        target_mode = bool(int(dataset[-1]))
        if align_dataset == 0:
            triples, entity_to_id, relation_to_id = load_dbp15k_data(dataset_name, target_mode, return_maps=True)
        else:
            triples, entity_to_id, relation_to_id = load_wk3l_15k_data(dataset_name, target_mode, return_maps=True)
        train_array, valid_array, test_array = split_dataset(triples, 0.7, 0.1, 0.2)
        config['dataset'] = dataset_name
    else:
        train_raw, valid_raw, test_raw = load_all_data(dataset)
        entity_to_id, relation_to_id = get_mapping_from_triples(train_raw)
        config['dataset'] = DATASETS[dataset]
        train_array, valid_array, test_array = encode_all(train_raw, valid_raw, test_raw, entity_to_id, relation_to_id)

    # complete the config
    config['e_num'] = len(entity_to_id)
    config['r_num'] = len(relation_to_id)
    # additional information stored in config
    config['train_size'] = len(train_array)
    config['validation_size'] = len(valid_array)
    config['test_size'] = len(test_array)

    # Init the model and the optimizer
    model = MODELS[config['model']](config, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0., 0.))

    # Prepare training data: create corrupted triples by replacing either head or tail of a training triple
    train = torch.tensor(train_array, dtype=torch.long, device=device)
    train_pos = train.repeat(config['corruption_factor'], 1)
    split = train_pos.shape[0] // 2
    entities = np.arange(config['e_num'])
    corr_subj = torch.tensor(np.random.choice(entities, split), dtype=torch.long, device=device)
    corr_subj_triples = torch.cat([corr_subj.view(-1, 1), train_pos[:split, 1:2], train_pos[:split, 2:3]], dim=1)
    corr_obj = torch.tensor(np.random.choice(entities, train_pos.shape[0] - split), dtype=torch.long, device=device)
    corr_obj_triples = torch.cat([train_pos[split:, 0:1], train_pos[split:, 1:2], corr_obj.view(-1, 1)], dim=1)
    train_neg = torch.cat([corr_subj_triples, corr_obj_triples], dim=0)

    # variables used for the training loop
    max_score = 0
    counter = 0
    batch_size = config['batch_size']
    batch_num = int(np.ceil(len(train_array) / batch_size))
    metrics = []
    model_copy = None
    plotter = None
    if plot:
        plotter = Plotter()

    # training loop
    for j in tqdm(range(config['num_epochs'])):
        model.train()
        current_loss = .0
        # Random shuffle with torch
        idx = torch.randperm(len(train_pos))
        train_pos = train_pos[idx]
        idx = torch.randperm(len(train_neg))
        train_neg = train_neg[idx]
        for i in range(batch_num):
            train_pos_batch = train_pos[i * batch_size: min((i + 1) * batch_size, len(train_pos))]
            train_neg_batch = train_neg[i * batch_size: min((i + 1) * batch_size, len(train_neg))]

            optimizer.zero_grad()  # reset calculated gradients for a new iteration
            loss = model(train_pos_batch, train_neg_batch)  # call the model to execute the training and calculate loss
            current_loss += loss.item() * train_pos_batch.shape[0]
            loss.backward()  # calculate the gradients of the training step with back propagation
            optimizer.step()  # update the model based on the optimizer

        # calculate the validation metrics
        avg_loss = current_loss/train.shape[0]
        current_metric = calculate_rank_metrics(model, valid_array, train_array, test_array,
                                                config['filter_validation_triples'])
        current_metric.append(avg_loss)
        metrics.append(current_metric)
        if device.type == "cuda":
            print(current_metric)

        if plot:
            plotter.update_plot(current_metric)

        # save the best model based on validation MRR
        if max_score == 0 or current_metric[1] > max_score:
            max_score = current_metric[1]
            model_copy = deepcopy(model.state_dict())
            counter = 0
        else:
            counter = counter + 1

        # termination criteria: validation MR is not increasing for 20 epochs
        if config['early_stopping'] and counter > 20:
            model.config['iterations'] = j
            print("Early stopping (validation MR did not increase for 20 epochs)")
            break

        # Manual termination
        if path.exists("stop.stop"):
            remove("stop.stop")
            model.config['iterations'] = j
            print("Manual termination")
            break

    # Restore the best model seen during training
    model.load_state_dict(model_copy)

    if save_results:
        # Save the model, training history and evaluation results
        export_link_prediction_model(f"0_{config['model']}_{config['dataset']}", model, entity_to_id, relation_to_id,
                                     metrics=metrics)
        save_link_prediction_test_results(model, test_array, train_array, valid_array,
                                          f"0_{config['model']}_{config['dataset']}", calculate_rank_metrics)
        if align_mode:
            # additionally save the trained embeddings for alignments
            save_link_prediction_embeddings(f"0_{config['model']}_{config['dataset']}", model)
    else:
        test_metrics = calculate_rank_metrics(model, test_array, train_array, valid_array, filtered=True)
        print("Test metrics: ", test_metrics)
        print("Number of entities: {}".format(model.e_num))


def run_test(device, model_name, model_type, align_dataset, save_results, align_mode):
    """
    Load a trained model or trained embeddings and evaluate it.
    :param device: the torch device on which the training is processed
    :param model_name: the name of the model directory
    :param model_type: 0 for TransE, 1 for RotatE
    :param align_dataset: 0 for DBP15k, 1 for WK3l-15k
    :param save_results: if True, save the evaluation results to the model directory
    :param align_mode: if True, then pre-trained embeddings are loaded instead of the model
    """
    if align_mode:
        dataset_name = model_name[:-1]
        target_mode = bool(int(model_name[-1]))
        model = load_link_prediction_pretrained(model_type, align_dataset, dataset_name, device, target_mode)
        if align_dataset == 0:
            data = load_dbp15k_data(dataset_name, target_mode)
        else:
            data = load_wk3l_15k_data(dataset_name, target_mode)

        train, valid, test = split_dataset(data, 0.99, 0., 0.01)
    else:
        model, entity_to_id, relation_to_id = load_link_prediction_model(model_name, device)
        dataset = model.config['dataset']
        train_raw, valid_raw, test_raw = load_all_data(dataset)
        train, valid, test = encode_all(train_raw, valid_raw, test_raw, entity_to_id, relation_to_id)
    if save_results:
        save_link_prediction_test_results(model, test, train, valid, model_name, calculate_rank_metrics)
    else:
        test_metrics = calculate_rank_metrics(model, test, train, valid, filtered=True)
        print("Test metrics: ", test_metrics)
        print("Number of entities: {}".format(model.e_num))
