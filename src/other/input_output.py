"""This file contains functions to save and load data, models, plots and evaluation results."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import save, load
from constants import DATASETS, MODEL_PATH, DATA_PATH, METRICS, PRETRAINED_EMBEDDINGS, EMBEDDING_PATH, BASE_CONFIG,\
    MODELS, REMOTE, AXIS_VALUES
from util import encode_alignment, get_mapping_from_triples, encode_triples, natural_sort, clean_config
from generator import Generator
from discriminator import Discriminator
import json
import os


def load_train_data(name):
    """Load the training data of the dataset."""
    return __load_triples(name, 0)


def load_valid_data(name):
    """Load the validation data of the dataset."""
    return __load_triples(name, 1)


def load_test_data(name):
    """Load the test data of the dataset."""
    return __load_triples(name, 2)


def load_all_data(name):
    """Load the training, validation and test data of the dataset."""
    return __load_triples(name, 0), __load_triples(name, 1), __load_triples(name, 2)


def __load_triples(name, mode):
    """
    Load a dataset from a file and return it as numpy array.
    :param name: the name or id of the dataset
    :param mode: either 0 for training, 1 for validation or 2 for test data
    :return: the data as a numpy array
    """
    if not isinstance(name, str):
        name = DATASETS[name]
    path = Path(DATA_PATH) / name
    if mode == 0:
        path = path / "train.txt"
    elif mode == 1:
        path = path / "valid.txt"
    elif mode == 2:
        path = path / "test.txt"
    triples = np.loadtxt(path, dtype=str)
    return np.array(triples)


def load_dbp15k_data(name, target_mode, return_maps=False):
    """
    Load an encoded DBP15k dataset as a numpy array and re-encode it to get steady ids.
    :param name: name of the DBP15k subset ("fr_en", "ja_en" or "zh_en")
    :param target_mode: if True use the target KG (e.g. "en" for "fr_en") else use the source KG ("fr" for "fr_en")
    :param return_maps: if set to True, return the mapping between old encoding from JAPE and new steady encoding
    :return: the re-encoded triples of the dataset as numpy array and if specified the mappings
    """
    path = Path(DATA_PATH) / "DBP15k_JAPE" / name / "0_3"
    if target_mode:
        triples = np.loadtxt(path / "triples_2", dtype=int, encoding='utf-8')
        id_to_entity = np.loadtxt(path / "ent_ids_2", dtype=str, encoding='utf-8')
        id_to_relation = np.loadtxt(path / "rel_ids_2", dtype=str, encoding='utf-8')
    else:
        triples = np.loadtxt(path / "triples_1", dtype=int, encoding='utf-8')
        id_to_entity = np.loadtxt(path / "ent_ids_1", dtype=str, encoding='utf-8')
        id_to_relation = np.loadtxt(path / "rel_ids_1", dtype=str, encoding='utf-8')

    # build new mappings
    entity_ids = id_to_entity[:, 0].astype(int)
    relation_ids = id_to_relation[:, 0].astype(int)
    entiy_id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(entity_ids)}
    relation_id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(relation_ids)}
    if return_maps:
        return encode_alignment(triples, entiy_id_to_new_id, relation_id_to_new_id, triple_encoding=True),\
               entiy_id_to_new_id, relation_id_to_new_id
    return encode_alignment(triples, entiy_id_to_new_id, relation_id_to_new_id, triple_encoding=True)


def load_wk3l_15k_data(name, target_mode, return_maps=False):
    """
    Load a WK3l-15k dataset as a numpy array and encode it.
    :param name: name of the Wk3l-15k subset ("en_de" or "en_fr")
    :param target_mode: if True use the target KG (e.g. "fr" for "en_fr") else use the source KG ("en" for "en_fr")
    :param return_maps: if set to True, return the entity2id and relation2id mappings of the triples
    :return: the encoded triples of the dataset as numpy array and if specified the mappings
    """
    path = Path(DATA_PATH) / "WK3l-15k" / name
    filename = name[target_mode * 3: target_mode * 3 + 2]
    # different versions of the WK3l-15k subsets must be considered to correctly locate the files
    if name == "en_de":
        version = "v6"
    else:
        version = "v5"
    # IMPORTANT: Do not not use numpy.loadtxt here, since 1-2 entities will be lost this way
    with open(path / f"P_{filename}_{version}.csv", "r", encoding="utf-8") as file:
        triples = np.array([line[:-1].split(sep="@@@") for line in file.readlines()])

    # encode triples and save mapping if not existent
    if os.path.exists(path / f"{filename}_entity_to_id.json")\
            and os.path.exists(path / f"{filename}_relation_to_id.json"):
        with open(path / f"{filename}_entity_to_id.json", "r") as file:
            entity_to_id = json.load(file)
        with open(path / f"{filename}_relation_to_id.json", "r") as file:
            relation_to_id = json.load(file)
    else:
        entity_to_id, relation_to_id = get_mapping_from_triples(triples)
        with open(path / f"{filename}_entity_to_id.json", "w") as file:
            json.dump(entity_to_id, file, indent=4)
        with open(path / f"{filename}_relation_to_id.json", "w") as file:
            json.dump(relation_to_id, file, indent=4)
    triples_encoded = encode_triples(triples, entity_to_id, relation_to_id)
    if return_maps:
        return triples_encoded, entity_to_id, relation_to_id
    return triples_encoded


def load_dbp15k_alignment(name):
    """Load the train and test alignment of a DBP15k dataset and encode them."""
    path = Path(DATA_PATH) / "DBP15k_JAPE" / name / "0_3"
    train = np.loadtxt(path / "sup_ent_ids", dtype=int, encoding='utf-8')
    test = np.loadtxt(path / "ref_ent_ids", dtype=int, encoding='utf-8')

    # build new mappings
    id_to_entity_source = np.loadtxt(path / "ent_ids_1", dtype=str, encoding='utf-8')
    source_ids = id_to_entity_source[:, 0].astype(int)
    source_id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(source_ids)}
    id_to_entity_target = np.loadtxt(path / "ent_ids_2", dtype=str, encoding='utf-8')
    target_ids = id_to_entity_target[:, 0].astype(int)
    target_id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(target_ids)}

    train_reencoded = encode_alignment(train, source_id_to_new_id, target_id_to_new_id, triple_encoding=False)
    test_reencoded = encode_alignment(test, source_id_to_new_id, target_id_to_new_id, triple_encoding=False)
    return train_reencoded, test_reencoded


def load_wk3l_15k_alignment(name):
    """Load the alignment files of a WK3l-15k dataset and encode them. The alignments of triples,
    source->target kg alignment and target->source kg alignment are combined. The entity alignments contain some
    entities (around 100 alignments) for which no triples exist, so that the model did not learn embeddings for those
    entities. Therefore, those additional triples and their alignments must be removed."""
    path = Path(DATA_PATH) / "WK3l-15k" / name
    source_name = name[0: 2]
    target_name = name[3: 5]
    if name == "en_de":
        version = "v6"
    else:
        version = "v5"

    # Load aligned data
    with open(path / f"P_{name}_{version}.csv", "r", encoding="utf-8") as file:
        aligned_triples = np.array([line[:-1].split(sep="@@@") for line in file.readlines()])
    # aligned_triples = np.loadtxt(path / f"P_{name}_{version}.csv", dtype=str, delimiter="@@@", encoding='utf-8')

    with open(path / f"{source_name}2{target_name}_fk.csv", "r", encoding="utf-8") as file:
        aligned_source = np.array([line[:-1].split(sep="@@@") for line in file.readlines()])
    with open(path / f"{target_name}2{source_name}_fk.csv", "r", encoding="utf-8") as file:
        aligned_target = np.array([line[:-1].split(sep="@@@") for line in file.readlines()])

    # Reshape alignment from target to source kg
    aligned_target = aligned_target[:, [1, 0]]

    # Extract aligned entites and relations from aligned triples
    aligned_heads = np.concatenate([aligned_triples[:, 0].reshape(-1, 1), aligned_triples[:, 3].reshape(-1, 1)], axis=1)
    aligned_relations = np.concatenate([aligned_triples[:, 1].reshape(-1, 1), aligned_triples[:, 4].reshape(-1, 1)],
                                       axis=1)
    aligned_tails = np.concatenate([aligned_triples[:, 2].reshape(-1, 1), aligned_triples[:, 5].reshape(-1, 1)], axis=1)

    # Combine aligned entities
    aligned_entities = np.concatenate([aligned_source, aligned_target, aligned_heads, aligned_tails], axis=0)

    # Remove duplicates
    aligned_entities = np.unique(aligned_entities, axis=0)
    aligned_relations = np.unique(aligned_relations, axis=0)

    # Load entity and relation mappings for source kg
    if os.path.exists(path / f"{source_name}_entity_to_id.json")\
            and os.path.exists(path / f"{source_name}_relation_to_id.json"):
        with open(path / f"{source_name}_entity_to_id.json", "r") as file:
            entity_to_id_source = json.load(file)
        with open(path / f"{source_name}_relation_to_id.json", "r") as file:
            relation_to_id_source = json.load(file)
    else:
        _, entity_to_id_source, relation_to_id_source = load_wk3l_15k_data(name, False, True)

    # Load entity and relation mappings for target kg
    if os.path.exists(path / f"{target_name}_entity_to_id.json")\
            and os.path.exists(path / f"{target_name}_relation_to_id.json"):
        with open(path / f"{target_name}_entity_to_id.json", "r") as file:
            entity_to_id_target = json.load(file)
        with open(path / f"{target_name}_relation_to_id.json", "r") as file:
            relation_to_id_target = json.load(file)
    else:
        _, entity_to_id_target, relation_to_id_target = load_wk3l_15k_data(name, False, True)

    # Encode alignments
    aligned_entities_encoded = encode_alignment(aligned_entities, entity_to_id_source, entity_to_id_target,
                                                triple_encoding=False)
    aligned_relations_encoded = encode_alignment(aligned_relations, relation_to_id_source, relation_to_id_target,
                                                 triple_encoding=False)

    return aligned_entities_encoded, aligned_relations_encoded


def export_link_prediction_model(name, model, entity_to_id, relation_to_id, metrics=None):
    """
    Save a trained model for link prediction and its components.
    :param name: the name of the model directory
    :param model: the pytorch model (subclass of torch.nn.Module) for link prediction of which the state dict is saved
    :param entity_to_id: the dictionary to encode kg entities
    :param relation_to_id: the dictionary to encode kg relations
    :param metrics: the training history (metrics of the validation data of each epoch)
    """
    path = Path(MODEL_PATH) / name
    if not os.path.exists(path):
        os.makedirs(path)
    save(model.state_dict(), path / "model.pt")
    # the config contains the kg embedding model type and the torch device that was used for training which is important
    # for loading the model correctly
    with open(path / "config.json", "w") as file:
        json.dump(model.config, file, indent=4)
    with open(path / "entity_to_id.json", "w") as file:
        json.dump(entity_to_id, file, indent=4)
    with open(path / "relation_to_id.json", "w") as file:
        json.dump(relation_to_id, file, indent=4)
    if metrics is not None:
        metric_dic = dict(zip(METRICS, np.transpose(metrics).tolist()))
        with open(path / "training.txt", "w") as file:
            json.dump(metric_dic, file, indent=4)
        plot_metrics(metrics, path, f"{model.config['model']} ({model.config['dataset']} dataset)")


def export_regression_alignment(name, config, generator_a, generator_b, metrics=None):
    """
    Save trained models and components for entity alignment with regression.
    :param name: the name of the model directory
    :param config: the dictionary containing model and training parameters
    :param generator_a: the pytorch model (subclass of torch.nn.Module) to transform from space B to A
    :param generator_b: the pytorch model (subclass of torch.nn.Module) to transform from space A to B
    :param metrics: the training history (metrics of the validation data of each epoch)
    """
    path = Path(MODEL_PATH) / name
    if not os.path.exists(path):
        os.makedirs(path)
    save(generator_a.state_dict(), path / "generator_a.pt")
    save(generator_b.state_dict(), path / "generator_b.pt")
    # the model configs contain the model parameters and the torch device that was used for training which is important
    # for loading the models correctly
    with open(path / "generator_a_config.json", "w") as file:
        json.dump(generator_a.config, file, indent=4)
    with open(path / "generator_b_config.json", "w") as file:
        json.dump(generator_b.config, file, indent=4)
    # the general config contains training parameters and specifies the dataset on which was trained
    config = clean_config(config, 0)
    with open(path / "config.json", "w") as file:
        json.dump(config, file, indent=4)
    if metrics is not None:
        metric_dic = dict(zip(METRICS, np.transpose(metrics).tolist()))
        with open(path / "training.txt", "w") as file:
            json.dump(metric_dic, file, indent=4)
        plot_metrics(metrics, path, f"Regression ({config['subset']} dataset)")


def export_cyclegan_alignment(name, config, generator_a, generator_b, discriminator_a, discriminator_b, metrics=None):
    """
    Save trained models and components for entity alignment with a cycle GAN architecture.
    :param name: the name of the model directory
    :param config: the dictionary containing model and training parameters
    :param generator_a: the pytorch model (subclass of torch.nn.Module) to transform from space B to A
    :param generator_b: the pytorch model (subclass of torch.nn.Module) to transform from space A to B
    :param discriminator_a: the pytorch model (subclass of torch.nn.Module) to decide if inputs come from space A
    :param discriminator_b: the pytorch model (subclass of torch.nn.Module) to decide if inputs come from space B
    :param metrics: the training history (metrics of the validation data of each epoch)
    """
    path = Path(MODEL_PATH) / name
    if not os.path.exists(path):
        os.makedirs(path)
    save(generator_a.state_dict(), path / "generator_a.pt")
    save(generator_b.state_dict(), path / "generator_b.pt")
    save(discriminator_a.state_dict(), path / "discriminator_a.pt")
    save(discriminator_b.state_dict(), path / "discriminator_b.pt")
    # the model configs contain the model parameters and the torch device that was used for training which is important
    # for loading the models correctly
    with open(path / "generator_a_config.json", "w") as file:
        json.dump(generator_a.config, file, indent=4)
    with open(path / "generator_b_config.json", "w") as file:
        json.dump(generator_b.config, file, indent=4)
    with open(path / "discriminator_a_config.json", "w") as file:
        json.dump(discriminator_a.config, file, indent=4)
    with open(path / "discriminator_b_config.json", "w") as file:
        json.dump(discriminator_b.config, file, indent=4)
    # the general config contains training parameters and specifies the dataset on which was trained
    config = clean_config(config, 1)
    with open(path / "config.json", "w") as file:
        json.dump(config, file, indent=4)
    if metrics is not None:
        metric_dic = dict(zip(METRICS, np.transpose(metrics).tolist()))
        with open(path / "training.txt", "w") as file:
            json.dump(metric_dic, file, indent=4)
        plot_metrics(metrics, path, f"Cycle GAN ({config['subset']} dataset)")


def export_model_results(name, config, valid_metrics, test_metrics, gan_mode, directory=None):
    """
    Save only the training history (validation metrics for every epoch) and model configuration of an entity alignment
    model.
    :param name: description of the model (used hyper-parameter)
    :param config: the dictionary containing model and training parameters
    :param valid_metrics: the training history (metrics of the validation data of each epoch)
    :param test_metrics: the evaluation results of the test set
    :param gan_mode: the model type, 0 for cycle regression, 1 for cycle gan
    :param directory: optional, the directory to which the results are saved ("tests" directory is used by default)
    """
    if directory is not None:
        path = Path(MODEL_PATH) / directory
    else:
        path = Path(MODEL_PATH) / "tests"
    if not os.path.exists(path):
        os.makedirs(path)
    data = {"Generated B->A": dict(zip(METRICS[:-1], test_metrics[0])),
            "Generated A->B": dict(zip(METRICS[:-1], test_metrics[1])),
            "Cyclic A->A": dict(zip(METRICS[:-1], test_metrics[2])),
            "Cyclic B->B": dict(zip(METRICS[:-1], test_metrics[3]))}
    with open(path / f"{name}_test_metrics.txt", "w") as file:
        json.dump(data, file, indent=4)
    if gan_mode == -1 or config is None or valid_metrics is None:  # no model, only transformation matrix
        return None
    config = clean_config(config, gan_mode)
    with open(path / f"{name}_config.json", "w") as file:
        json.dump(config, file, indent=4)
    metric_dic = dict(zip(METRICS, np.transpose(valid_metrics).tolist()))
    with open(path / f"{name}_training.txt", "w") as file:
        json.dump(metric_dic, file, indent=4)


def export_ablation_details(directory, parameter, values):
    """
    Save the name of the parameter that was varied along with its values.
    :param directory: the directory name of the ablation study
    :param parameter: name of the parameter that was changed
    :param values: list with values of the vaired parameter
    """
    path = Path(MODEL_PATH) / directory
    data = {'parameter': parameter, 'values': values}
    with open(path / "parameter.json", "w") as file:
        json.dump(data, file, indent=4)


def load_ablation_details(directory, remote_dir=False):
    """Load the parameter that was changed for the ablation study and its values for plotting the results."""
    if remote_dir:
        path = Path(REMOTE) / directory
    else:
        path = Path(MODEL_PATH) / directory
    if os.path.exists(path / "parameter.json"):
        with open(path / "parameter.json", "r") as file:
            result = json.load(file)
        return result['values']
    else:
        for x in AXIS_VALUES:
            if x in directory:
                return AXIS_VALUES[x]


def load_evaluation_results(directory, remote_dir=False):
    """Load the evaluation results of the test data of a final model. The files should be named "..{i}_test_metrics.txt"
    with i being the i-th repetition of model training (final models are trained 5 times to report average and variance
    of the evaluation results, hyper-parameter studies are done with 3 repetitions)."""
    if remote_dir:
        path = Path(REMOTE) / directory
    else:
        path = Path(MODEL_PATH) / directory
    files = os.listdir(path)
    result_files = [x for x in files if x[-17:] == "_test_metrics.txt"]
    result_files = natural_sort(result_files)
    results = []
    for x in result_files:
        with open(path / x, "r") as file:
            results.append(json.load(file))
    history_files = [x for x in files if x[-13:] == "_training.txt"]
    histories = []
    for x in history_files:
        with open(path / x, "r") as file:
            data = np.array(json.load(file)['Hits@1'])
            histories.append(np.where(data == max(data))[0][0])
    print(f"Avg first epoch with Max Hits@1: {np.average(histories)}")
    return results


def load_configs(directory, remote_dir=False):
    """Load the configurations of an ablation study for plotting the results."""
    if remote_dir:
        path = Path(REMOTE) / directory
    else:
        path = Path(MODEL_PATH) / directory
    files = os.listdir(path)
    result_files = [x for x in files if x[-12:] == "_config.json"]
    result_files = natural_sort(result_files)
    results = []
    for x in result_files:
        with open(path / x, "r") as file:
            results.append(json.load(file))
    return results


def load_history(directory, remote_dir=False):
    """Load the history of a training process."""
    if remote_dir:
        path = Path(REMOTE) / directory
    else:
        path = Path(MODEL_PATH) / directory
    files = os.listdir(path)
    result_files = [x for x in files if x[-13:] == "_training.txt"]
    result_files = natural_sort(result_files)
    results = []
    for x in result_files:
        with open(path / x, "r") as file:
            results.append(json.load(file))
    return results


def load_link_prediction_model(name, device):
    """
    Load a trained model for link prediction. The model config is used to recreated the model class.
    :param name: the name of the model directory
    :param device: the current torch device, used for transferring the saved model (which was possibly trained on a
    different device) to the correct device
    :return: the link prediction model (subclass of torch.nn.Module) and its entity and relation mappings
    """
    path = Path(MODEL_PATH) / name
    with open(path / "config.json", "r") as file:
        config = json.load(file)
    with open(path / "entity_to_id.json", "r") as file:
        entity_to_id = json.load(file)
    with open(path / "relation_to_id.json", "r") as file:
        relation_to_id = json.load(file)
    model = MODELS[config['model']](config, device)
    model.load_state_dict(load(path / "model.pt", map_location=device))
    model.to(device)
    return model, entity_to_id, relation_to_id


def load_regression_alignment(name, device):
    """
    Load trained models for entity alignment with regression.
    :param name: the name of the model directory
    :param device: the current torch device, used for transferring the saved models (which were possibly trained on a
    different device) to the correct device
    :return: the generator models (subclass of torch.nn.Module) and the training configurations for entity alignment
    with regression
    """
    path = Path(MODEL_PATH) / name
    with open(path / "config.json", "r") as file:
        config = json.load(file)
    # Load Generator B->A
    with open(path / "generator_a_config.json", "r") as file:
        generator_a_config = json.load(file)
    generator_a = Generator(generator_a_config, device)
    generator_a.load_state_dict(load(path / "generator_a.pt", map_location=device))
    generator_a.to(device)
    # Load Generator A->B
    with open(path / "generator_b_config.json", "r") as file:
        generator_b_config = json.load(file)
    generator_b = Generator(generator_b_config, device)
    generator_b.load_state_dict(load(path / "generator_b.pt", map_location=device))
    generator_b.to(device)
    return generator_a, generator_b, config


def load_cyclegan_alignment(name, device):
    """
    Load trained models for entity alignment with a cycle GAN architecture.
    :param name: the name of the model directory
    :param device: the current torch device, used for transferring the saved models (which were possibly trained on a
    different device) to the correct device
    :return: the generator and discriminator models (subclasses of torch.nn.Module) and the training configurations
    for entity alignment with a cycle gan architecture
    """
    path = Path(MODEL_PATH) / name
    with open(path / "config.json", "r") as file:
        config = json.load(file)
    # Load Generator B->A
    with open(path / "generator_a_config.json", "r") as file:
        generator_a_config = json.load(file)
    generator_a = Generator(generator_a_config, device)
    generator_a.load_state_dict(load(path / "generator_a.pt", map_location=device))
    generator_a.to(device)
    # Load Generator A->B
    with open(path / "generator_b_config.json", "r") as file:
        generator_b_config = json.load(file)
    generator_b = Generator(generator_b_config, device)
    generator_b.load_state_dict(load(path / "generator_b.pt", map_location=device))
    generator_b.to(device)
    # Load Discriminator A
    with open(path / "discriminator_a_config.json", "r") as file:
        discriminator_a_config = json.load(file)
    discriminator_a = Discriminator(discriminator_a_config, device)
    discriminator_a.load_state_dict(load(path / "discriminator_a.pt", map_location=device))
    discriminator_a.to(device)
    # Load Discriminator B
    with open(path / "discriminator_b_config.json", "r") as file:
        discriminator_b_config = json.load(file)
    discriminator_b = Discriminator(discriminator_b_config, device)
    discriminator_b.load_state_dict(load(path / "discriminator_b.pt", map_location=device))
    discriminator_b.to(device)
    return generator_a, generator_b, discriminator_a, discriminator_b, config


def save_link_prediction_embeddings(name, model):
    """
    Save the embeddings of trained kg link prediction models.
    :param name: name of the model directory
    :param model: the pytorch kg link prediction model from which the embeddings are extracted
    """
    path = Path(MODEL_PATH) / name

    # extract embeddings from model
    e_embeddings = model.e_embedding.weight.data.detach().tolist()
    r_embeddings = model.r_embedding.weight.data.detach().tolist()

    embeddings = [e_embeddings, r_embeddings]
    with open(path / "embeddings.json", "w") as file:
        json.dump(embeddings, file, indent=4)


def load_pretrained_embeddings(model, dataset, subset, device, without_relations=True):
    """
    Load the two corresponding pre-trained kg embeddings trained on an aligned dataset.
    :param model: choose the link prediction model (see constants.MODELS for available models)
    :param dataset: choose the alignment dataset (see constants.PRETRAINED_EMBEDDINGS for available dataset)
    :param subset: name of the subset (e.g. "fr_en") to locate the files with the pretrained embeddings
    :param device: torch device to which the embeddings are moved
    :param without_relations: if True, only load entity embeddings
    :return: the pre-trained entity embeddings of both kg and optionally also their relation embeddings
    """
    model = list(MODELS)[model]
    dataset = PRETRAINED_EMBEDDINGS[dataset]

    # Load embeddings of source kg/kg 1/match kg
    filename_source = subset[0: 2] + "_embeddings.json"
    path_source = Path(EMBEDDING_PATH) / model / dataset / subset / filename_source
    with open(path_source, "r") as file:
        embeddings_source = json.load(file)
    e_embedding_source = torch.tensor(embeddings_source[0], device=device)

    # Load embeddings of target kg/kg 2/ref kg
    filename_target = subset[3: 5] + "_embeddings.json"
    path_target = Path(EMBEDDING_PATH) / model / dataset / subset / filename_target
    with open(path_target, "r") as file:
        embeddings_target = json.load(file)
    e_embedding_target = torch.tensor(embeddings_target[0], device=device)

    if without_relations:
        return e_embedding_source, e_embedding_target

    r_embedding_source = torch.tensor(embeddings_source[1], device=device)
    r_embedding_source = r_embedding_source[:int(0.5 * r_embedding_source.shape[0])]
    r_embedding_target = torch.tensor(embeddings_target[1], device=device)
    r_embedding_target = r_embedding_target[:int(0.5 * r_embedding_target.shape[0])]
    return e_embedding_source, r_embedding_source, e_embedding_target, r_embedding_target


def load_link_prediction_pretrained(model, dataset, subset, device, target_kg):
    """
    Load pre-trained embeddings into the link prediction model.
    :param model: choose the link prediction model (see constants.MODELS for available models)
    :param dataset: choose the alignment dataset (see constants.PRETRAINED_EMBEDDINGS for available dataset)
    :param subset: name of the subset (e.g. "fr_en") to locate the file with the pretrained embeddings
    :param device: torch device on which the model will be executed
    :param target_kg: if True use the target KG (e.g. "en" for "fr_en") else use the source KG ("fr" for "fr_en")
    :return: the TransE model with the pre-trained DBP15k embeddings
    """
    filename = subset[target_kg * 3: target_kg * 3 + 2] + "_embeddings.json"
    model = list(MODELS)[model]
    dataset = PRETRAINED_EMBEDDINGS[dataset]
    path = Path(EMBEDDING_PATH) / model / dataset / subset / filename
    with open(path, "r") as file:
        embeddings = json.load(file)
    e_embedding = torch.tensor(embeddings[0])
    r_embedding = torch.tensor(embeddings[1])
    r_embedding = r_embedding[:int(0.5 * r_embedding.shape[0])]
    model = MODELS[model](BASE_CONFIG, device, e_embedding, r_embedding)
    model.to(device)
    return model


def save_plot(x_data, y_data, path, title=None, x_label=None, y_label=None, legend=None, rotate_axis=False,
              y_data_2=None, second_y_label=None):
    """
    Create and save a single line plot with the provided information.
    :param x_data: the data for the x-axis
    :param y_data: the data for the y-axis
    :param path: the path of the while to which the plot is stored
    :param title: optional, title of the plot
    :param x_label: optional, name of the x-axis
    :param y_label: optional, name of the y-axis
    :param legend: optional, plot legend if more than one data row
    :param rotate_axis: if True, rotate the x-axis values by 45 degrees for better readability of longer values
    :param y_data_2: optional, use a second y-axis to display data of different scales (e.g. Hits@1 and iterations)
    :param second_y_label: optional, name for the second axis
    """
    fig, ax1 = plt.subplots()
    ax1.plot(x_data, y_data)
    if rotate_axis:
        plt.xticks(rotation=45)
    if title is not None:
        plt.title(title)
    if x_label is not None:
        ax1.set_xlabel(x_label)
    if y_label is not None:
        ax1.set_ylabel(y_label)
    if legend is not None:
        plt.legend(legend)  # , loc="center right" , bbox_to_anchor=(0,0.1,1,1))
    if y_data_2 is not None:
        if len(y_data.shape) > 1:
            color = '#000000'
        else:
            color = 'tab:blue'
        ax1.set_ylabel(y_label, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        if len(y_data.shape) > 1:
            color = 'tab:red'
        else:
            color = 'tab:orange'
        ax2.set_ylabel(second_y_label, color=color)
        ax2.plot(x_data, y_data_2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.clf()


def plot_metrics(metrics, directory, description):
    """
    Plots the validation metrics of a training history.
    :param metrics: the metrics of the training
    :param directory: the path (string or Path object) of the output directory
    :param description: the description of the model for the title
    """
    metrics = np.array(metrics)
    if isinstance(directory, str):
        directory = Path(directory)
    save_plot(x_data=range(len(metrics)), y_data=metrics[:, 1:-1], path=directory / "training.png",
              title="Validation metrics for " + description, x_label="Epoch", y_label="Metrics",
              legend=METRICS[1:-1])
    save_plot(x_data=range(len(metrics)), y_data=metrics[:, -1], path=directory / "training-loss.png",
              title="Model loss for " + description, x_label="Epoch", y_label="Loss")
    save_plot(x_data=range(len(metrics)), y_data=metrics[:, 0], path=directory / "training-mr.png",
              title="Validation mean rank for " + description, x_label="Epoch", y_label="Mean Rank")


def save_alignment_test_results(test_metrics, model_name):
    """
    Save the evaluation results of a trained alignment model.
    :param test_metrics: the alignment metrics for generated test entities (unidirectional and cyclic)
    :param model_name: the name of the model to locate its directory
    """
    path = Path(MODEL_PATH) / model_name
    data = {"Generated B->A": dict(zip(METRICS[:-1], test_metrics[0])),
            "Generated A->B": dict(zip(METRICS[:-1], test_metrics[1])),
            "Cyclic A->A": dict(zip(METRICS[:-1], test_metrics[2])),
            "Cyclic B->B": dict(zip(METRICS[:-1], test_metrics[3]))}
    with open(path / "test_metrics.txt", "w") as file:
        json.dump(data, file, indent=4)


def save_link_prediction_test_results(model, test, train, valid, model_name, calculate_rank_function):
    """
    Save the evaluation results of a trained link prediction model.
    :param model: the trained model
    :param test: test data for the evaluation
    :param train: train data used for filtering
    :param valid: validation data used for filtering
    :param model_name: the name of the model to locate its directory
    :param calculate_rank_function: the evaluation function used to compute the metrics
    """
    path = Path(MODEL_PATH) / model_name
    metrics_unfiltered = calculate_rank_function(model, test, train, valid, filtered=False)
    metrics_filtered = calculate_rank_function(model, test, train, valid, filtered=True)
    data = {"unfiltered": dict(zip(METRICS[:-1], metrics_unfiltered)),
            "filtered": dict(zip(METRICS[:-1], metrics_filtered))}
    with open(path / "test_metrics.txt", "w") as file:
        json.dump(data, file, indent=4)
