"""A class for hyper parameter grid search of kg embedding models with plotting of the results."""

import matplotlib.pyplot as plt
import time
from torch import cuda, device as torch_device
from input_output import load_all_data
from link_prediction import run_train
from evaluate import calculate_rank_metrics
from util import get_mapping_from_triples, encode_triples
from constants import BASE_CONFIG, METRICS
import os

# hyper parameter ranges for grid search
dims = [25, 50, 100, 200, 500]
lrs = [0.005, 0.01, 0.05, 0.1, 0.3]
corruptions = [1, 5, 10, 40, 100]
evaluation = [0, 1, 2]
models = ["RotatE", "TransE"]

# dictionaries used for result plotting
ranges = {0: dims, 1: lrs, 2: corruptions, 3: evaluation, 4: models}
modes = {0: "dim", 1: "learning_rate", 2: "corruption_factor", 3: "eval_method", 4: "model"}
metrics = METRICS[2:-1]


def hyperparam_grid_search(mode, config, datasets, device):
    """
    Train and evaluate models for each value of the selected hyper parameter (for each dataset) and save plots of the
    metrics.
    :param mode: defines which of the model parameters is searched
    :param config: a dictionary with model parameters
    :param datasets: a list of datasets that are used for training
    :param device: the torch device on which the code is executed
    """
    for x in datasets:
        hits_result = [[] for _ in range(len(metrics))]
        name = (x.replace("../../../data/git/other/", "").replace("/train.txt", ""))
        train_data, valid_data, test_data = load_all_data(x)
        entity_to_id, relation_to_id = get_mapping_from_triples(train_data)
        train = encode_triples(train_data, entity_to_id, relation_to_id)
        valid = encode_triples(valid_data, entity_to_id, relation_to_id)
        test = encode_triples(test_data, entity_to_id, relation_to_id)
        print(name)
        for y in ranges[mode]:
            print("{}: {}".format(modes[mode], y))
            config[modes[mode]] = y
            start = time.time()
            model = run_train(config, device, x, -1, False, False)
            tr_subj, tr_obj = calculate_rank_metrics(model, entity_to_id, relation_to_id, True)
            te_subj, te_obj = calculate_rank_metrics(model, entity_to_id, relation_to_id, True)
            print(time.time()-start)
            for i, metric in enumerate(metrics):
                hits_result[i].append([tr_subj[metric], tr_obj[metric], te_subj[metric], te_obj[metric]])
        for i, metric in enumerate(metrics):
            plt.plot(ranges[mode], hits_result[i])
            plt.title("{} for {}\nDim={}, lr={}, corruption={}, eval={}, model={}".format(
                metric, name, config['dim'], config['learning_rate'], config['corruption_factor'],
                config['eval_method'], config['model']))
            plt.xlabel(modes[mode])
            plt.ylabel(metric)
            plt.legend(["Train data, subjects", "Train data, objects", "Test data, subjects", "Test data, objects"])
            if not os.path.exists("../../../results/figures/" + name + "/" + modes[mode]):
                os.makedirs("../../../results/figures/" + name + "/" + modes[mode])
            plt.savefig("../../../results/figures/" + name + "/" + modes[mode] + "/" + metric + ".png")
            plt.clf()
        print()


# Execute the hyper parameter search
modified_config = dict(
    dim=20,
    margin=2,
    batch_size=32,
    num_epochs=200,
    corruption_factor=1,
    filter_validation_triples=0,
    model="TransE",
    device="cuda:1"
)
config = BASE_CONFIG.copy()
config.update(modified_config)
if cuda.is_available() and config['device'] != "cpu":
    device = torch_device(config['device'])
else:
    config['device'] = "cpu"
    device = torch_device("cpu")

hyperparam_grid_search(0, config.copy(), [0], device)
