"""The main script for training or evaluating kg embedding models."""

from link_prediction import run_train, run_test
from constants import BASE_CONFIG, MODELS
from torch import cuda, manual_seed, device as torch_device
from numpy.random import seed
import time

# Parameters
SEED = False  # use fixed seed (0) for numpy and torch random modules
TRAIN_MODEL = False  # train a new model or load a trained model
SAVE_RESULTS = False  # save trained model and/or model evaluation
ALIGN_MODE = True  # use aligned datasets or pre-trained embeddings like DBP15k

# Variables required for training
dataset = "en_fr0"  # dataset name or id
align_data_train = 1  # 0 for DBP15k, 1 for WK3l-15k (only for align mode)
model_type_train = 0  # 0 for TransE, 1 for RotatE (only for align mode)

# Variables required for testing
model_name = "fr_en0"  # name of the model or the embeddings that is loaded, also used to load the corresponding data
align_data_test = 0  # 0 for DBP15k, 1 for WK3l-15k (only for align mode)
model_type_test = 1  # 0 for TransE, 1 for RotatE (only for align mode)

# this dictionary contains the basic parameters for training the model
custom_config = dict(
    dim=1000,
    margin=10,
    batch_size=1000,
    learning_rate=0.2,
    num_epochs=200,
    corruption_factor=5,
    filter_validation_triples=False,
    early_stopping=True,
    device="cuda:1"
)
config = BASE_CONFIG.copy()
config.update(custom_config)
config['model'] = list(MODELS)[model_type_train]

if cuda.is_available() and config['device'] != "cpu":
    device = torch_device(config['device'])
else:
    config['device'] = "cpu"
    device = torch_device("cpu")
print("Using {}".format(config['device']))

if SEED:
    manual_seed(0)  # pytorch seed
    seed(0)  # numpy seed

if TRAIN_MODEL:
    start = time.time()
    run_train(config=config, device=device, dataset=dataset, align_dataset=align_data_train, save_results=SAVE_RESULTS,
              align_mode=ALIGN_MODE, plot=False)
    print("Training took {} seconds.".format(time.time() - start))
else:
    run_test(device=device, model_name=model_name, model_type=model_type_test, align_dataset=align_data_test,
             save_results=SAVE_RESULTS, align_mode=ALIGN_MODE)
