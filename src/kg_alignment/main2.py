"""The main script for training or evaluating kg alignment models."""

from torch import cuda, manual_seed, device as torch_device
from numpy.random import seed
from constants import FINAL_DBP_CONFIG, FINAL_WK3L_CONFIG
from entity_alignment import run_train, run_test, calculate_transformation_matrix, single_hyperparam_study, run_final,\
    multi_hyperparam_study

# Main parameters to control the mode
SEED = False  # use fixed seed (0) for numpy and torch random modules
FINAL = True  # calculate model with final parameters 5 times
HYPERPARAMS = False  # test the impact varied hyper-parameters (only results, no models saved)
TRAIN_MODEL = True  # train a new model or load a trained model
SAVE_RESULTS = False  # save not only model history and evaluation but also the model (for normal training)
GAN_MODE = 0  # 0 for regression, 1 for cycle GAN, -1 for direct transformation matrix
EVALUATION = 2  # 0 for alignment with only test entities, 1 for test+valid+exclusive entities, 2 for all entities

# Variables required for training
dataset = 0  # 0 for DBP15k, 1 for WK3l-15k
subset = "fr_en"  # dataset name or id
model_type = 0  # 0 for TransE, 1 for RotatE

# Variables required for testing
model_name = "SimpleRegression_fr_en"  # name of the model that is loaded

config = dict(
    batch_size=1000,
    batch_size_rest=1000,
    num_epochs=1000,
    learning_rate=0.001,
    delta1=1,  # factor for discriminator loss of train data
    delta2=1,  # factor for discriminator loss of generated data
    gamma1=1,  # factor for generator loss of generated data
    gamma2=1,  # factor for generator loss of cycle data
    gamma3=1,  # factor for generator loss of discriminator
    epsilon1=1,  # factor for unaligned loss of neighbour data
    epsilon2=1,  # factor for unaligned loss of cycle data
    epsilon3=1,  # factor for unaligned loss of discriminator
    device="cuda:0",
    generator_layers=3,
    generator_expansion=5,
    discriminator_layers=4,
    discriminator_expansion=2,
    evaluation=EVALUATION,
    dataset=dataset,
    subset=subset,
    model_type=model_type,
    validation_split=0.95,
    early_stopping=1,
    adaptive_lr=0,
    patience=50,
    first100=0,
    initialize_generator=0,
    initialize_discriminator=1,
    cyclic_loss_train=0,
    cyclic_loss_test=1,
    margin_loss=1,
    generator_repetition=1,
    discriminator_repetition=0,
    # generator and discriminator layer parameters for ablation study
    gen_norm=1,
    gen_batch_norm=1,
    gen_activation=1,
    gen_dropout=0,
    disc_batch_norm=0,
    disc_activation=1,
    disc_dropout=1,
)

if cuda.is_available() and config['device'] != "cpu":
    device = torch_device(config['device'])
else:
    config['device'] = "cpu"
    device = torch_device("cpu")
print("Using {}".format(config['device']))

if SEED:
    manual_seed(0)  # pytorch seed
    seed(0)  # numpy seed

if FINAL and HYPERPARAMS:
    print("Too many modes selected. please choose only one the them.")
    exit()

if not FINAL and not HYPERPARAMS:
    if TRAIN_MODEL:
        if GAN_MODE == -1:
            calculate_transformation_matrix(config, device)
        else:
            run_train(config, device, GAN_MODE, SAVE_RESULTS)
    else:
        run_test(model_name, device, GAN_MODE, SAVE_RESULTS)

if FINAL:
    config['dataset'] = 0
    config['subset'] = "fr_en"
    # calculate_transformation_matrix(config, device, final=True)
    config.update(FINAL_DBP_CONFIG)
    run_final(config, device, GAN_MODE)
    cuda.empty_cache()
    config['subset'] = "zh_en"
    # calculate_transformation_matrix(config, device, final=True)
    config.update(FINAL_DBP_CONFIG)
    run_final(config, device, GAN_MODE)
    cuda.empty_cache()
    config['subset'] = "ja_en"
    # calculate_transformation_matrix(config, device, final=True)
    config.update(FINAL_DBP_CONFIG)
    run_final(config, device, GAN_MODE)
    cuda.empty_cache()
    config['dataset'] = 1
    config['subset'] = "en_de"
    # calculate_transformation_matrix(config, device, final=True)
    config.update(FINAL_WK3L_CONFIG)
    run_final(config, device, GAN_MODE)
    cuda.empty_cache()
    config['subset'] = "en_fr"
    # calculate_transformation_matrix(config, device, final=True)
    config.update(FINAL_WK3L_CONFIG)
    run_final(config, device, GAN_MODE)

if HYPERPARAMS:  # list all parameter tests here
        # config['dataset'] = 1
        # config['subset'] = "en_de"
        config['dataset'] = 0
        config['subset'] = "fr_en"

        single_hyperparam_study(config, device, GAN_MODE, "learning_rate")
        multi_hyperparam_study(config, device, GAN_MODE, "batch_sizes")
        multi_hyperparam_study(config, device, GAN_MODE, "gen_loss")
        multi_hyperparam_study(config, device, GAN_MODE, "gen_layers")
        multi_hyperparam_study(config, device, GAN_MODE, "disc_layers")
        multi_hyperparam_study(config, device, GAN_MODE, "disc_size")
        multi_hyperparam_study(config, device, GAN_MODE, "gen_size")
        # single_hyperparam_study(config, device, GAN_MODE, "gamma1")
        # single_hyperparam_study(config, device, GAN_MODE, "gamma2")
        # single_hyperparam_study(config, device, GAN_MODE, "gamma3")
        # single_hyperparam_study(config, device, GAN_MODE, "epsilon1")
        # single_hyperparam_study(config, device, GAN_MODE, "epsilon2")
        # single_hyperparam_study(config, device, GAN_MODE, "epsilon3")
        # single_hyperparam_study(config, device, GAN_MODE, "activation_function")
        # single_hyperparam_study(config, device, GAN_MODE, "dropout_rate")
        # single_hyperparam_study(config, device, GAN_MODE, "optimizer")
        # single_hyperparam_study(config, device, GAN_MODE, "optimizer_default")
