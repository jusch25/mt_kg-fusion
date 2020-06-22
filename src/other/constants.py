"""This file contains various constants used for kg embedding and alignment models."""

from transE import TransE
from rotatE import RotatE
from torch.optim.adam import Adam
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamax import Adamax
from torch.optim.sgd import SGD
from torch.optim.rmsprop import RMSprop
from torch.nn import ReLU, LeakyReLU, PReLU, ELU, Tanh


# path to the data directory
DATA_PATH = "../../../data/"

# a list of available link prediction datasets
DATASETS = ["countries_S1", "countries_S2", "countries_S3",
            "nations", "kinship", "umls",
            "WN18RR", "WN18",
            "FB15k-237", "FB15k",
            "YAGO3-10"]

# a list of available alignment datasets
AlIGN_DATASETS = ["CN3l", "WK3l-15k", "DWY100k", "DBP15k_FULL", "DBP15_JAPE", "WK3l-15k", "WK3l-120k"]

# a list of available pretrained embeddings
PRETRAINED_EMBEDDINGS = ["DBP15k", "WK3l-15k"]

# path to the trained models
MODEL_PATH = "../../../results/models/"

# path to pretrained embeddings
EMBEDDING_PATH = "../../../results/embeddings/"

# path to result plots
FIGURE_PATH = "../../../results/figures/"

# path to remote models
REMOTE = "../../../results/models/Tests"

# a list of available metrics for model evaluation
METRICS = ["MR", "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10", "Loss"]

# a list of available kg embedding models
MODELS = {"TransE": TransE, "RotatE": RotatE}

# a list of tested activation functions for the kg alignment models
ACTIVATIONS = {'relu': ReLU, 'prelu': PReLU, 'leakyrelu': LeakyReLU, 'leakyrelu2': LeakyReLU, 'elu': ELU, 'tanh': Tanh}

# a list of tested optimizers for the kg alignment models
OPTIMIZERS = {'adam': Adam, 'adadelta': Adadelta, 'adagrad': Adagrad, 'adamax': Adamax, 'rmsprop': RMSprop, 'sgd': SGD}

# a dictionary with default values of required parameters for kg embedding models
BASE_CONFIG = dict(
    embedding_norm=2,
    margin=1,
    distance_norm=1,
    filter_validation_triples=False,
    early_stopping=True,
    dim=50,
    learning_rate=0.01,
    batch_size=32,
    num_epochs=100,
    corruption_factor=1,
    model="TransE",
    device="cpu"
)

# a dictionary with default values of required parameters for kg alignment regression
REGRESSION_CONFIG = dict(
    batch_size=1000,
    batch_size_rest=1000,
    num_epochs=1000,
    learning_rate=0.003,
    gamma1=1,
    gamma2=1,
    epsilon1=1,
    epsilon2=1,
    device="cpu",
    generator_layers=3,
    generator_expansion=5,
    evaluation=None,
    dataset=None,
    subset=None,
    model_type=None,
    validation_split=0.95,
    early_stopping=1,
    adaptive_lr=1,
    initialize_generator=0,
    cyclic_loss_train=0,
    cyclic_loss_test=1,
    margin_loss=1,
    gen_norm=1,
    gen_batch_norm=0,
    gen_activation=1,
    gen_dropout=0,
    dim_a=-1,
    dim_b=-1,
    train_size=-1,
    validation_size=-1,
    test_size=-1,
    iterations=-1,
)

# a dictionary with default values of required parameters for kg alignment cycle GAN
CYCLE_GAN_CONFIG = dict(
    batch_size=1000,
    batch_size_rest=1000,
    num_epochs=1000,
    learning_rate=0.003,
    delta1=1,
    delta2=1,
    gamma1=1,
    gamma2=1,
    gamma3=1,
    epsilon1=1,
    epsilon2=1,
    epsilon3=1,
    device="cpu",
    generator_layers=3,
    generator_expansion=5,
    discriminator_layers=6,
    discriminator_expansion=3,
    evaluation=None,
    dataset=None,
    subset=None,
    model_type=None,
    validation_split=0.95,
    early_stopping=1,
    adaptive_lr=1,
    initialize_generator=0,
    initialize_discriminator=1,
    cyclic_loss_train=0,
    cyclic_loss_test=1,
    margin_loss=1,
    generator_repetition=1,
    discriminator_repetition=0,
    gen_norm=1,
    gen_batch_norm=0,
    gen_activation=1,
    gen_dropout=0,
    disc_batch_norm=0,
    disc_activation=1,
    disc_dropout=0,
    dim_a=-1,
    dim_b=-1,
    train_size=-1,
    validation_size=-1,
    test_size=-1,
    iterations=-1,
)

# a dictionary with hyper-parameters and their values for hyper-parameter evaluation
HYPERPARAM = {
    'learning_rate': [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001],
    'delta1': [0.0, 0.3, 0.5, 0.7, 1., 1.3, 1.5, 1.7, 2.],
    'delta2': [0.0, 0.3, 0.5, 0.7, 1., 1.3, 1.5, 1.7, 2.],
    'gamma1': [0.0, 0.3, 0.5, 0.7, 1., 1.3, 1.5, 1.7, 2.],
    'gamma2': [0.0, 0.3, 0.5, 0.7, 1., 1.3, 1.5, 1.7, 2.],
    'gamma3': [0.0, 0.3, 0.5, 0.7, 1., 1.3, 1.5, 1.7, 2.],
    'epsilon1': [0.0, 0.3, 0.5, 0.7, 1., 1.3, 1.5, 1.7, 2.],
    'epsilon2': [0.0, 0.3, 0.5, 0.7, 1., 1.3, 1.5, 1.7, 2.],
    'epsilon3': [0.0, 0.3, 0.5, 0.7, 1., 1.3, 1.5, 1.7, 2.],
    'activation_function': list(ACTIVATIONS.keys()),
    'dropout_rate': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
    'optimizer': list(OPTIMIZERS.keys()),
    'optimizer_default': list(OPTIMIZERS.keys()),
    'patience': [10, 20, 30, 50, 70, 100],
}

# a dictionary with sets of hyper-parameters and their values which are evaluated simultaneously via grid testing
PARAMGROUPS = {
    'gen_loss': {
        'cyclic_loss_train': [0, 1],
        'cyclic_loss_test': [0, 1],
        'margin_loss': [0, 1],
        '': [""],  # dummy entry
    },
    'gen_layers': {
        'gen_norm': [0, 1],
        'gen_batch_norm': [0, 1],
        'gen_activation': [0, 1],
        'gen_dropout': [0, 1],
    },
    'gen_size': {
        'generator_layers': [0, 2, 4, 6],
        'generator_expansion': [2, 4, 6],
        '': [""],  # dummy entry
        "_": ["_"],  # dummy entry
    },
    'disc_layers': {
        'disc_batch_norm': [0, 1],
        'disc_activation': [0, 1],
        'disc_dropout': [0, 1],
        '': [""],  # dummy entry
    },
    'disc_size': {
        'discriminator_layers': [0, 2, 4, 6],
        'discriminator_expansion': [2, 4, 6],
        '': [""],  # dummy entry
        "_": ["_"],  # dummy entry
    },
    'batch_sizes': {
        'batch_size': [500, 1000, 2000],
        'batch_size_rest': [500, 1000, 2000],
        '': [""],  # dummy entry
        "_": ["_"],  # dummy entry
    },
    'repetitions': {
        'generator_repetition': [1, 2, 3, 4],
        'discriminator_repetition': [1, 2, 3, 4],
        '': [""],  # dummy entry
        "_": ["_"],  # dummy entry
    },
}

AXIS_VALUES = {
    'cyclic_loss_train': ['___', '__N', '_U_', '_UN', 'T__', 'T_N', 'TU_', 'TUN'],
    'gen_norm': ['____', '___D', '__A_', '__AD',
                 '_B__', '_B_D', '_BA_', '_BAD',
                 'N___', 'N__D', 'N_A_', 'N_AD',
                 'NB__', 'NB_D', 'NBA_', 'NBAD'],
    'layers': ['0,2', '0,4', '0,6',
               '2,2', '2,4', '2,6',
               '4,2', '4,4', '4,6',
               '6,2', '6,4', '6,6'],
    'disc_activation': ['___', '__D', '_A_', '_AD', 'B__', 'B_D', 'BA_', 'BAD'],
    'batch_size': ['500,500', '500,1000', '500,2000',
                   '1000,500', '1000,1000', '1000,2000',
                   '2000,500', '2000,1000', '2000,2000'],
    'repetition': ['1,1', '1,2', '1,3', '1,4',
                   '2,1', '2,2', '2,3', '2,4',
                   '3,1', '3,2', '3,3', '3,4',
                   '4,1', '4,2', '4,3', '4,4'],
    'variance': range(10),
    'adaptive': range(10),
    'final': range(5),
}

# optimized configuration for DBP15k datasets based on hyper-parameter tests and ablation results on subset fr-en
FINAL_DBP_CONFIG = dict(
    batch_size=1000,
    batch_size_rest=1000,
    num_epochs=2000,
    learning_rate=0.003,
    delta1=1,  # factor for discriminator loss of train data
    delta2=1,  # factor for discriminator loss of generated data
    gamma1=1.7,  # factor for generator loss of generated data
    gamma2=1,  # factor for generator loss of cycle data
    gamma3=1,  # factor for generator loss of discriminator
    epsilon1=1,  # factor for unaligned loss of neighbour data
    epsilon2=1,  # factor for unaligned loss of cycle data
    epsilon3=1,  # factor for unaligned loss of discriminator
    generator_layers=6,
    generator_expansion=8,
    discriminator_layers=4,
    discriminator_expansion=6,
    validation_split=0.95,
    early_stopping=1,
    adaptive_lr=0,
    patience=50,
    first100=1,
    initialize_generator=0,
    initialize_discriminator=1,
    cyclic_loss_train=0,
    cyclic_loss_test=1,
    margin_loss=1,
    generator_repetition=1,
    discriminator_repetition=0,
    gen_norm=1,
    gen_batch_norm=1,
    gen_activation=1,
    gen_dropout=0,
    disc_batch_norm=0,
    disc_activation=1,
    disc_dropout=1,
)

# optimized configuration for WK3l-15k datasets based on hyper-parameter tests and ablation results on subset en-de
FINAL_WK3L_CONFIG = dict(
    batch_size=1000,
    batch_size_rest=1000,
    num_epochs=2000,
    learning_rate=0.003,
    delta1=1,  # factor for discriminator loss of train data
    delta2=1,  # factor for discriminator loss of generated data
    gamma1=1.7,  # factor for generator loss of generated data
    gamma2=1,  # factor for generator loss of cycle data
    gamma3=1,  # factor for generator loss of discriminator
    epsilon1=1,  # factor for unaligned loss of neighbour data
    epsilon2=1,  # factor for unaligned loss of cycle data
    epsilon3=1,  # factor for unaligned loss of discriminator
    generator_layers=4,
    generator_expansion=6,
    discriminator_layers=4,
    discriminator_expansion=6,
    validation_split=0.95,
    early_stopping=1,
    adaptive_lr=0,
    patience=50,
    first100=1,
    initialize_generator=0,
    initialize_discriminator=1,
    cyclic_loss_train=0,
    cyclic_loss_test=1,
    margin_loss=1,
    generator_repetition=1,
    discriminator_repetition=0,
    gen_norm=1,
    gen_batch_norm=1,
    gen_activation=1,
    gen_dropout=1,
    disc_batch_norm=0,
    disc_activation=1,
    disc_dropout=1,
)
