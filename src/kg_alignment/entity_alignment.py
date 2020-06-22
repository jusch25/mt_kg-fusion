"""This file contains the main functions for entity alignment training and testing."""

import torch
from torch.utils.data import random_split
import numpy as np
from evaluate import calculate_all_cycle_gan_metrics
from input_output import load_pretrained_embeddings, load_dbp15k_alignment, save_alignment_test_results,\
    load_wk3l_15k_alignment, export_model_results, export_ablation_details
from util import embedding_partitioning
from constants import HYPERPARAM, PARAMGROUPS
from alignment_model import RegressionModel, CycleGAN
import time
from tqdm import tqdm


def calculate_transformation_matrix(config, device, final=False):
    """
    This function calculates the linear transformation matrix between the train alignment embeddings as a solution of
    the least squares problem.
    :param config: a dictionary with training and model parameters
    :param device: the torch device on which the calculations are executed
    :param final: if False (default) then a single transformation matrix is computed and the results are printed else
    the computation is repeated five times and the results are stored to obtain the baseline model
    """
    data_a, data_b = __load_data(config, device)
    train_a, valid_a, test_a, rest_a, eval_a = data_a
    train_b, valid_b, test_b, rest_b, eval_b = data_b

    train_a_copy = train_a.cpu()
    train_b_copy = train_b.cpu()

    if not final:  # only print the results, no saving
        transform_ab = np.linalg.lstsq(train_a_copy, train_b_copy, rcond=None)[0]
        transform_ba = np.linalg.lstsq(train_b_copy, train_a_copy, rcond=None)[0]
        transform_ab = torch.tensor(transform_ab, device=device)
        transform_ba = torch.tensor(transform_ba, device=device)

        gen_a = torch.matmul(test_b, transform_ba)
        gen_b = torch.matmul(test_a, transform_ab)
        cyc_a = torch.matmul(gen_b, transform_ba)
        cyc_b = torch.matmul(gen_a, transform_ab)

        print("Test metrics all")
        calculate_all_cycle_gan_metrics(eval_a, eval_b, gen_a, gen_b, cyc_a, cyc_b, combined_results=False)
        print("Test metrics -train")
        calculate_all_cycle_gan_metrics(rest_a, rest_b, gen_a, gen_b, cyc_a, cyc_b, combined_results=False)
        print("Test metrics test")
        calculate_all_cycle_gan_metrics(test_a, test_b, gen_a, gen_b, cyc_a, cyc_b, combined_results=False)
    else:  # repeat each transformation 5 times and save the test metrics for all evalutaion sets
        for i in range(5):
            transform_ab = np.linalg.lstsq(train_a_copy, train_b_copy, rcond=None)[0]
            transform_ba = np.linalg.lstsq(train_b_copy, train_a_copy, rcond=None)[0]
            transform_ab = torch.tensor(transform_ab, device=device)
            transform_ba = torch.tensor(transform_ba, device=device)

            gen_a = torch.matmul(test_b, transform_ba)
            gen_b = torch.matmul(test_a, transform_ab)
            cyc_a = torch.matmul(gen_b, transform_ba)
            cyc_b = torch.matmul(gen_a, transform_ab)

            print("Test metrics all")
            test_results_all = calculate_all_cycle_gan_metrics(eval_a, eval_b, gen_a, gen_b, cyc_a, cyc_b,
                                                               combined_results=False)
            print("Test metrics -train")
            test_results_ntrain = calculate_all_cycle_gan_metrics(rest_a, rest_b, gen_a, gen_b, cyc_a, cyc_b,
                                                                  combined_results=False)
            print("Test metrics test")
            test_results_test = calculate_all_cycle_gan_metrics(test_a, test_b, gen_a, gen_b, cyc_a, cyc_b,
                                                                combined_results=False)
            export_model_results(f"all_{i}", None, None, test_results_all, -1,
                                 f"final_{-1}_{config['subset']}")
            export_model_results(f"ntrain_{i}", None, None, test_results_ntrain, -1,
                                 f"final_{-1}_{config['subset']}")
            export_model_results(f"test_{i}", None, None, test_results_test, -1,
                                 f"final_{-1}_{config['subset']}")


def run_train(config, device, gan_mode, save_results):
    """
    The entry point when a new model is trained. The configuration specifies all training and model details and other
    functions perform the training steps
    :param config: a dictionary with training and model parameters
    :param device: the torch device on which the calculations are executed
    :param gan_mode: the model type, 0 for regression, 1 for cycle GAN
    :param save_results:
    :return:
    """
    data_a, data_b = __load_data(config, device)
    model, test_results = __train(gan_mode, device, config, data_a, data_b)

    if save_results:
        model.export_model(test_results)
    else:
        name = f"{gan_mode}_{config['evaluation']}_{config['subset']}_{config['learning_rate']}"
        export_model_results(name, config, model.metrics, test_results, gan_mode)
        print(f"Saved results to {name}.")


def multi_hyperparam_study(config, device, gan_mode, group, comment=""):
    """
    This script performs a multi-dimensional parameter study (mostly ablation studies) which test all combinations of
    the provided parameters. Every test is repeated 3 times. The training history, the configuration and the metrics
    from the evaluation are always stored but never the model.
    :param config: a dictionary with training and model parameters
    :param device: the torch device on which the calculations are executed
    :param gan_mode: the model type, 0 for regression, 1 for cycle GAN
    :param group: the name of the mutli-dimensional parameter test that shall be executed (the value must exist in
    constants.py)
    :param comment: an optional comment for the directory names to avoid overwriting older results
    """
    config['evaluation'] = 2
    data_a, data_b = __load_data(config, device)
    train_a, valid_a, test_a, rest_a, eval_a = data_a
    train_b, valid_b, test_b, rest_b, eval_b = data_b
    counter = 1

    dimensions = list(PARAMGROUPS[group].keys())
    dimension_1 = dimensions[0]
    param_1 = PARAMGROUPS[group][dimension_1]
    dimension_2 = dimensions[1]
    param_2 = PARAMGROUPS[group][dimension_2]
    dimension_3 = dimensions[2]
    param_3 = PARAMGROUPS[group][dimension_3]
    dimension_4 = dimensions[3]
    param_4 = PARAMGROUPS[group][dimension_4]
    description = f"{dimension_1}_{dimension_2}_{dimension_3}_{dimension_4}"

    for x in param_1:
        for y in param_2:
            for z in param_3:
                for w in param_4:
                    for i in range(3):
                        config_copy = dict(config)
                        config_copy[dimension_1] = x
                        config_copy[dimension_2] = y
                        config_copy[dimension_3] = z
                        config_copy[dimension_4] = w
                        model = __train(gan_mode, device, config_copy, data_a, data_b, False)
                        model.eval()
                        with torch.no_grad():
                            gen_a = model.generator_a.generate(test_b)
                            gen_b = model.generator_b.generate(test_a)
                            cyc_a = model.generator_a.generate(gen_b)
                            cyc_b = model.generator_b.generate(gen_a)
                            print("Test metrics all")
                            test_results_all = calculate_all_cycle_gan_metrics(eval_a, eval_b, gen_a, gen_b, cyc_a,
                                                                               cyc_b, combined_results=False)
                            print("Test metrics -train")
                            test_results_ntrain = calculate_all_cycle_gan_metrics(rest_a, rest_b, gen_a, gen_b, cyc_a,
                                                                                  cyc_b, combined_results=False)
                            print("Test metrics test")
                            test_results_test = calculate_all_cycle_gan_metrics(test_a, test_b, gen_a, gen_b, cyc_a,
                                                                                cyc_b, combined_results=False)
                        export_model_results(f"{counter}#{x}_{y}_{z}_{w}+{i}", config_copy, model.metrics,
                                             test_results_test, gan_mode,
                                             f"{gan_mode}_0_{config_copy['subset']}_{description}{comment}")
                        export_model_results(f"{counter}#{x}_{y}_{z}_{w}+{i}", config_copy, model.metrics,
                                             test_results_ntrain, gan_mode,
                                             f"{gan_mode}_1_{config_copy['subset']}_{description}{comment}")
                        export_model_results(f"{counter}#{x}_{y}_{z}_{w}+{i}", config_copy, model.metrics,
                                             test_results_all, gan_mode,
                                             f"{gan_mode}_2_{config_copy['subset']}_{description}{comment}")
                        counter += 1


def single_hyperparam_study(config, device, gan_mode, parameter, comment=""):
    """
    This function provides parameter tests, where the value for one parameter is changed and the results are saved for
    each evaluation set. Every test is repeated 3 times. The training history, the configuration and the metrics from
    the evaluation are always stored but never the model.
    :param config: a dictionary with training and model parameters
    :param device: the torch device on which the calculations are executed
    :param gan_mode: the model type, 0 for regression, 1 for cycle GAN
    :param parameter: the name of the single parameter that shall be tested (the value must exist in constants.py)
    :param comment: an optional comment for the directory names to avoid overwriting older results
    """
    config['evaluation'] = 2
    data_a, data_b = __load_data(config, device)
    train_a, valid_a, test_a, rest_a, eval_a = data_a
    train_b, valid_b, test_b, rest_b, eval_b = data_b
    counter = 1

    values = HYPERPARAM[parameter]

    for x in values:
        config_copy = dict(config)
        config_copy[parameter] = x
        for i in range(3):
            model = __train(gan_mode, device, config_copy, data_a, data_b, False)
            model.eval()
            with torch.no_grad():
                gen_a = model.generator_a.generate(test_b)
                gen_b = model.generator_b.generate(test_a)
                cyc_a = model.generator_a.generate(gen_b)
                cyc_b = model.generator_b.generate(gen_a)
                print("Test metrics all")
                test_results_all = calculate_all_cycle_gan_metrics(eval_a, eval_b, gen_a, gen_b, cyc_a, cyc_b,
                                                                   combined_results=False)
                print("Test metrics -train")
                test_results_ntrain = calculate_all_cycle_gan_metrics(rest_a, rest_b, gen_a, gen_b, cyc_a, cyc_b,
                                                                      combined_results=False)
                print("Test metrics test")
                test_results_test = calculate_all_cycle_gan_metrics(test_a, test_b, gen_a, gen_b, cyc_a, cyc_b,
                                                                    combined_results=False)
            export_model_results(f"{counter}#{x}+{i}", config_copy, model.metrics, test_results_test, gan_mode,
                                 f"{gan_mode}_0_{config_copy['subset']}_{parameter}{comment}")
            export_ablation_details(f"{gan_mode}_0_{config_copy['subset']}_{parameter}{comment}",
                                    parameter, values)
            export_model_results(f"{counter}#{x}+{i}", config_copy, model.metrics, test_results_ntrain, gan_mode,
                                 f"{gan_mode}_1_{config_copy['subset']}_{parameter}{comment}")
            export_ablation_details(f"{gan_mode}_1_{config_copy['subset']}_{parameter}{comment}",
                                    parameter, values)
            export_model_results(f"{counter}#{x}+{i}", config_copy, model.metrics, test_results_all, gan_mode,
                                 f"{gan_mode}_2_{config_copy['subset']}_{parameter}{comment}")
            export_ablation_details(f"{gan_mode}_2_{config_copy['subset']}_{parameter}{comment}",
                                    parameter, values)
            counter += 1


def manual_study(config, device, gan_mode, title, repetitions):
    """
    This function trains a single model configuration a customizable number of repetitions.
    :param config: a dictionary with training and model parameters
    :param device: the torch device on which the calculations are executed
    :param gan_mode: the model type, 0 for regression, 1 for cycle GAN
    :param title: the name of the manual study
    :param repetitions: the number of repititions
    """
    config['evaluation'] = 2
    data_a, data_b = __load_data(config, device)
    train_a, valid_a, test_a, rest_a, eval_a = data_a
    train_b, valid_b, test_b, rest_b, eval_b = data_b

    for i in range(repetitions):
        model = __train(gan_mode, device, config, data_a, data_b, False)
        model.eval()
        with torch.no_grad():
            gen_a = model.generator_a.generate(test_b)
            gen_b = model.generator_b.generate(test_a)
            cyc_a = model.generator_a.generate(gen_b)
            cyc_b = model.generator_b.generate(gen_a)
            print("Test metrics all")
            test_results_all = calculate_all_cycle_gan_metrics(eval_a, eval_b, gen_a, gen_b, cyc_a, cyc_b,
                                                               combined_results=False)
            print("Test metrics -train")
            test_results_ntrain = calculate_all_cycle_gan_metrics(rest_a, rest_b, gen_a, gen_b, cyc_a, cyc_b,
                                                                  combined_results=False)
            print("Test metrics test")
            test_results_test = calculate_all_cycle_gan_metrics(test_a, test_b, gen_a, gen_b, cyc_a, cyc_b,
                                                                combined_results=False)
        export_model_results(f"{i}", config, model.metrics, test_results_test, gan_mode,
                             f"{gan_mode}_0_{config['subset']}_{title}")
        export_model_results(f"{i}", config, model.metrics, test_results_ntrain, gan_mode,
                             f"{gan_mode}_1_{config['subset']}_{title}")
        export_model_results(f"{i}", config, model.metrics, test_results_all, gan_mode,
                             f"{gan_mode}_2_{config['subset']}_{title}")


def run_final(config, device, gan_mode):
    """
    Train the final models for the competitive results. Each model is trained five times and the results are computed
    and saved for all evaluation sets. The best of the five model is stored.
    :param config: a dictionary with training and model parameters
    :param device: the torch device on which the calculations are executed
    :param gan_mode: the model type, 0 for regression, 1 for cycle GAN
    """
    config['evaluation'] = 2
    data_a, data_b = __load_data(config, device)
    train_a, valid_a, test_a, rest_a, eval_a = data_a
    train_b, valid_b, test_b, rest_b, eval_b = data_b
    best_model = 0

    for i in range(5):
        model = __train(gan_mode, device, config, data_a, data_b, False)
        model.eval()
        with torch.no_grad():
            gen_a = model.generator_a.generate(test_b)
            gen_b = model.generator_b.generate(test_a)
            cyc_a = model.generator_a.generate(gen_b)
            cyc_b = model.generator_b.generate(gen_a)
            print("Test metrics all")
            test_results_all = calculate_all_cycle_gan_metrics(eval_a, eval_b, gen_a, gen_b, cyc_a, cyc_b,
                                                               combined_results=False)
            print("Test metrics -train")
            test_results_ntrain = calculate_all_cycle_gan_metrics(rest_a, rest_b, gen_a, gen_b, cyc_a, cyc_b,
                                                                  combined_results=False)
            print("Test metrics test")
            test_results_test = calculate_all_cycle_gan_metrics(test_a, test_b, gen_a, gen_b, cyc_a, cyc_b,
                                                                combined_results=False)
        if test_results_all[0][2] + test_results_all[1][2] > best_model:
            best_model = test_results_all[0][2] + test_results_all[1][2]
            model.export_model(test_results_all, f"final_{gan_mode}_{config['subset']}")
        export_model_results(f"all_{i}", config, model.metrics, test_results_all, gan_mode,
                             f"final_{gan_mode}_{config['subset']}")
        export_model_results(f"ntrain_{i}", config, model.metrics, test_results_ntrain, gan_mode,
                             f"final_{gan_mode}_{config['subset']}")
        export_model_results(f"test_{i}", config, model.metrics, test_results_test, gan_mode,
                             f"final_{gan_mode}_{config['subset']}")


def run_test(name, device, gan_mode, save_results):
    """
    This function loads a previously trained model and evaluates it. The results can be saved.
    :param name: a dictionary with training and model parameters
    :param device: the torch device on which the calculations are executed
    :param gan_mode: the model type, 0 for regression, 1 for cycle GAN
    :param save_results: If true, store the results of the model evaluation
    """
    if gan_mode:
        model = CycleGAN.load_model(name, device)
    else:
        model = RegressionModel.load_model(name, device)

    data_a, data_b = __load_data(model.config, device)
    train_a, valid_a, test_a, rest_a, eval_a = data_a
    train_b, valid_b, test_b, rest_b, eval_b = data_b

    model.eval()
    with torch.no_grad():
        gen_a = model.generator_a.generate(test_b)
        gen_b = model.generator_b.generate(test_a)
        cyc_a = model.generator_a.generate(gen_b)
        cyc_b = model.generator_b.generate(gen_a)
        print("Test metrics")
        test_results = calculate_all_cycle_gan_metrics(eval_a, eval_b, gen_a, gen_b, cyc_a, cyc_b,
                                                       combined_results=False)

    if save_results:
        save_alignment_test_results(test_results, name)


def __train_cycle_regression(model, train_a, train_b, valid_a, valid_b, rest_a, rest_b):
    """
    The script that actually executes the regression training. At first parameters are set and the neighbours are
    calculated. Then the training loop starts. In every epoch the generators are trained with the generator loss and
    optionally the cyclic loss for training data, the cyclic loss for the remaining data and the neighbour loss - based
    on the configuration. At the end of each epoch the early stopping criteria is checked, the model is tested with the
    validation data and the best model is updated if necessary.
    :param model: the alignment model with two generators
    :param train_a: the embeddings of KG 1 that are part of the train alignments
    :param train_b: the embeddings of KG 2 that are part of the train alignments
    :param valid_a: the embeddings of KG 1 that are part of the validation alignments
    :param valid_b: the embeddings of KG 2 that are part of the validation alignments
    :param rest_a: the remaining embeddings of KG 1 (test embeddings and unaligned)
    :param rest_b: the remaining embeddings of KG 2 (test embeddings and unaligned)
    """
    start = time.time()
    max_score = 0
    counter = 0
    lr_change = False
    model.config['iterations'] = model.config['num_epochs']

    # batch sizes of train embeddings and the rest
    batch_size = model.config['batch_size']
    if batch_size == 0:
        batch_size = train_a.shape[0]
    batch_num = int(np.ceil(train_a.shape[0] / batch_size))
    if model.config['batch_size_rest'] == 0:
        batch_num_rest = 1
    else:  # Potential problems if batch_size_rest < batch_num_rest
        batch_num_rest = int(np.ceil(max(rest_a.shape[0], rest_b.shape[0]) / model.config['batch_size_rest']))
    batch_size_rest_a = int(np.ceil(rest_a.shape[0] / batch_num_rest))
    batch_size_rest_b = int(np.ceil(rest_b.shape[0] / batch_num_rest))

    # calculate neighbours
    neighbours_a = None
    neighbours_b = None
    if model.config['margin_loss']:
        neighbours_a = __get_closest_neighbour(rest_a, train_a)
        neighbours_b = __get_closest_neighbour(rest_b, train_b)

    # start training
    iterator = tqdm(range(model.config['num_epochs']))
    for j in iterator:
        model.train()
        model.zero_grad()

        idx_train = torch.randperm(train_a.shape[0])
        train_a_epoch = train_a[idx_train]
        train_b_epoch = train_b[idx_train]
        idx_rest_a = torch.randperm(rest_a.shape[0])
        rest_a_epoch = rest_a[idx_rest_a]
        idx_rest_b = torch.randperm(rest_b.shape[0])
        rest_b_epoch = rest_b[idx_rest_b]

        # the generator training with the training alignments
        for i in range(batch_num):
            train_a_batch = train_a_epoch[i * batch_size: min((i + 1) * batch_size, len(train_a_epoch))]
            train_b_batch = train_b_epoch[i * batch_size: min((i + 1) * batch_size, len(train_b_epoch))]

            gen_b = model.generator_b(train_a_batch)
            gen_a = model.generator_a(train_b_batch)

            loss_b = model.generator_b.score_generated(train_b_batch, gen_b) * model.config['gamma1']
            loss_a = model.generator_a.score_generated(train_a_batch, gen_a) * model.config['gamma1']

            if model.config['cyclic_loss_train']:
                cyc_b = model.generator_b(gen_a)
                cyc_a = model.generator_a(gen_b)

                loss_cyc = (model.generator_b.score_cyclic(train_a_batch, cyc_a, train_b_batch, cyc_b) *
                            model.config['gamma2'])
                loss_a += loss_cyc / 2
                loss_b += loss_cyc / 2

            loss_total = loss_a + loss_b
            loss_total.backward()

            model.optimize_all()
            model.update_losses_batch(loss_a.item() * train_a_batch.shape[0] / train_a.shape[0],
                                      loss_b.item() * train_b_batch.shape[0] / train_b.shape[0])

        # the training of the remaining embeddings
        for i in range(batch_num_rest):
            rest_a_batch = rest_a_epoch[i * batch_size_rest_a: min((i + 1) * batch_size_rest_a, rest_a_epoch.shape[0])]
            rest_b_batch = rest_b_epoch[i * batch_size_rest_b: min((i + 1) * batch_size_rest_b, rest_b_epoch.shape[0])]

            gen_a = None
            gen_b = None
            loss_a_old = None
            loss_b_old = None
            if model.config['cyclic_loss_test']:
                gen_b = model.generator_b(rest_a_batch)
                gen_a = model.generator_a(rest_b_batch)
                cyc_b = model.generator_b(gen_a)
                cyc_a = model.generator_a(gen_b)

                loss_cyc = (model.generator_a.score_cyclic(rest_a_batch, cyc_a, rest_b_batch, cyc_b) *
                            model.config['epsilon2'])
                loss_a = loss_b = loss_cyc / 2

                if model.config['margin_loss']:
                    loss_a_old = loss_a
                    loss_b_old = loss_b
                else:
                    loss_cyc.backward()
                    model.optimize_all()
                    model.update_losses_batch(loss_a.item() * rest_a_batch.shape[0] / rest_a_epoch.shape[0],
                                              loss_b.item() * rest_b_batch.shape[0] / rest_b_epoch.shape[0])

            # neighbour loss
            if model.config['margin_loss']:
                neighbours_a_batch = neighbours_a[idx_rest_a][i * batch_size_rest_a: min((i + 1) * batch_size_rest_a,
                                                                                         rest_a_epoch.shape[0])]
                neighbours_b_batch = neighbours_b[idx_rest_b][i * batch_size_rest_b: min((i + 1) * batch_size_rest_b,
                                                                                         rest_b_epoch.shape[0])]
                gen_b_train = model.generator_b.generate(train_a)
                gen_a_train = model.generator_a.generate(train_b)

                if gen_a is None or gen_b is None:
                    gen_a = model.generator_a(rest_b_batch)
                    gen_b = model.generator_b(rest_a_batch)

                loss_a = (model.generator_a.score_unaligned(gen_a, gen_a_train[neighbours_b_batch]) *
                          model.config['epsilon1'])
                loss_b = (model.generator_b.score_unaligned(gen_b, gen_b_train[neighbours_a_batch]) *
                          model.config['epsilon1'])

                if loss_a_old is not None and loss_b_old is not None:
                    loss_a += loss_a_old
                    loss_b += loss_b_old

                loss_total = loss_a + loss_b
                loss_total.backward()

                model.optimize_all()
                model.update_losses_batch(loss_a.item() * rest_a_batch.shape[0] / rest_a_epoch.shape[0],
                                          loss_b.item() * rest_b_batch.shape[0] / rest_b_epoch.shape[0])

        # evaluation of the model with the validation data
        model.eval()
        with torch.no_grad():
            gen_a = model.generator_a.generate(valid_b)
            gen_b = model.generator_b.generate(valid_a)
            cyc_a = model.generator_a.generate(gen_b)
            cyc_b = model.generator_b.generate(gen_a)
            current_metrics = calculate_all_cycle_gan_metrics(valid_a, valid_b, gen_a, gen_b, cyc_a, cyc_b)
        model.complete_epoch(current_metrics)

        # early stopping criteria
        if max_score == 0 or current_metrics[2] > max_score:
            max_score = current_metrics[2]
            model.copy_model()
            counter = 0
            lr_change = False
        elif not (model.config['first100'] and j < 100):
            counter = counter + 1
        if model.config['early_stopping'] and counter > model.config['patience']:
            if model.config['adaptive_lr'] and counter < 80:
                if not lr_change:
                    model.restore_model()
                    model.change_lr(0.5)
                    lr_change = True
                    print(f"Changed learning rate of optimizer to: {model.current_lr}")
            else:
                model.config['iterations'] = j
                print(f"Stopped at epoch {j} due to early stopping with max Hits@1 of {max_score}.")
                iterator.close()
                break

        model.print_epoch_info()
    end = time.time()
    print(f"Training took {(end - start) / 60:.1f}min, with "
          f"{(end - start) / model.config['iterations']:.2f}s/iteration.")


def __train_cycle_gan(model, train_a, train_b, valid_a, valid_b, rest_a, rest_b):
    """
    The script that actually executes the cycle GAN training. If necessary the script starts with the initialization of
    the discriminator by training it with the direct transformation matrix of the train alignment embeddings. Both
    discriminator losses are used. Afterwards, parameters are set and the neighbours are calculated. Then the training
    loop starts. If the discriminator was not initialized then the generator and discriminator are trained in turn by
    using their specific losses. Else only the generator is trained, like in the regression model. After the basic
    generators and discriminator losses, the optional cyclic loss for training data, cyclic loss for the remaining data
    and neighbour loss are calculated - based on the configuration. At the end of each epoch the early stopping criteria
    is checked, the model is tested based on the validation data and the best model is updated if necessary.
    :param model: the alignment model with two generators and two discriminators
    :param train_a: the embeddings of KG 1 that are part of the train alignments
    :param train_b: the embeddings of KG 2 that are part of the train alignments
    :param valid_a: the embeddings of KG 1 that are part of the validation alignments
    :param valid_b: the embeddings of KG 2 that are part of the validation alignments
    :param rest_a: the remaining embeddings of KG 1 (test embeddings and unaligned)
    :param rest_b: the remaining embeddings of KG 2 (test embeddings and unaligned)
    """
    start = time.time()
    if model.config['initialize_discriminator']:
        # train discriminator first after initializing generator with transform matrix
        train_a_copy = train_a.cpu()
        train_b_copy = train_b.cpu()
        transform_ab = np.linalg.lstsq(train_a_copy, train_b_copy, rcond=None)[0]
        transform_ba = np.linalg.lstsq(train_b_copy, train_a_copy, rcond=None)[0]
        transform_ab = torch.tensor(transform_ab, device=model.device)
        transform_ba = torch.tensor(transform_ba, device=model.device)

        # initialize discriminator
        min_score = 0
        counter = 0
        iterator = tqdm(range(10000))
        for i in iterator:
            model.train()
            model.zero_grad()

            idx = torch.randperm(train_a.shape[0])
            train_a_epoch = train_a[idx]
            train_b_epoch = train_b[idx]
            gen_b_tmp = torch.matmul(train_a_epoch, transform_ab)
            gen_a_tmp = torch.matmul(train_b_epoch, transform_ba)

            dec_b = model.discriminator_b(train_b_epoch)  # + torch.rand(train_b_batch.shape) * 0.6 - 0.3)
            loss_d_b_train = model.discriminator_b.score_train(dec_b)
            dec_gen_b = model.discriminator_b(gen_b_tmp)  # + torch.rand(gen_b_tmp.shape) * 0.6 - 0.3)
            loss_d_b_gen = model.discriminator_b.score_gen(dec_gen_b)
            loss_d_b = loss_d_b_train * model.config['delta1'] + loss_d_b_gen * model.config['delta2']

            dec_a = model.discriminator_a(train_a_epoch)  # + torch.rand(train_a_batch.shape) * 0.6 - 0.3)
            loss_d_a_train = model.discriminator_a.score_train(dec_a)
            dec_gen_a = model.discriminator_a(gen_a_tmp)  # + torch.rand(gen_a_tmp.shape) * 0.6 - 0.3)
            loss_d_a_gen = model.discriminator_a.score_gen(dec_gen_a)
            loss_d_a = loss_d_a_train * model.config['delta1'] + loss_d_a_gen * model.config['delta2']

            loss_d_total = loss_d_a + loss_d_b
            loss_d_total.backward()

            model.optimize_discriminator()

            if min_score == 0 or loss_d_total.item() < min_score:
                min_score = loss_d_total.item()
                model.copy_model()
                counter = 0
            else:
                counter = counter + 1
            if counter > 15:
                print(f"Stopped discriminator training at epoch {i}. Lowest loss: {min_score}")
                iterator.close()
                break

        model.restore_model()

        # calculate neighbours
        gen_b = torch.matmul(rest_a, transform_ab)
        gen_a = torch.matmul(rest_b, transform_ba)
        cyc_b = torch.matmul(gen_a, transform_ab)
        cyc_a = torch.matmul(gen_b, transform_ba)
        print(f"D_A: Test score = {model.discriminator_a.decide(rest_a).mean()}"
              f" - Gen score = {model.discriminator_a.decide(gen_a).mean()}"
              f" - Cyc score = {model.discriminator_a.decide(cyc_a).mean()}")
        print(f"D_B: Test score = {model.discriminator_b.decide(rest_b).mean()}"
              f" - Gen score = {model.discriminator_b.decide(gen_b).mean()}"
              f" - Cyc score = {model.discriminator_b.decide(cyc_b).mean()}")

    max_score = 0
    counter = 0
    lr_change = False
    model.config['iterations'] = model.config['num_epochs']

    # batch sizes of train embeddings and the rest
    batch_size = model.config['batch_size']
    if batch_size == 0:
        batch_size = train_a.shape[0]
    batch_num = int(np.ceil(train_a.shape[0] / batch_size))
    if model.config['batch_size_rest'] == 0:
        batch_num_rest = 1
    else:  # Potential problems if batch_size_rest < batch_num_rest
        batch_num_rest = int(np.ceil(max(rest_a.shape[0], rest_b.shape[0]) / model.config['batch_size_rest']))
    batch_size_rest_a = int(np.ceil(rest_a.shape[0] / batch_num_rest))
    batch_size_rest_b = int(np.ceil(rest_b.shape[0] / batch_num_rest))

    neighbours_a = None
    neighbours_b = None
    if model.config['margin_loss']:
        neighbours_a = __get_closest_neighbour(rest_a, train_a)
        neighbours_b = __get_closest_neighbour(rest_b, train_b)

    # start the training
    iterator = tqdm(range(model.config['num_epochs']))
    for j in iterator:
        model.train()
        model.zero_grad()

        idx = torch.randperm(train_a.shape[0])
        train_a_epoch = train_a[idx]
        train_b_epoch = train_b[idx]
        idx_rest_a = torch.randperm(rest_a.shape[0])
        rest_a_epoch = rest_a[idx_rest_a]
        idx_rest_b = torch.randperm(rest_b.shape[0])
        rest_b_epoch = rest_b[idx_rest_b]

        # training based on train alignments
        for i in range(batch_num):
            train_a_batch = train_a_epoch[i * batch_size: min((i + 1) * batch_size, len(train_a_epoch))]
            train_b_batch = train_b_epoch[i * batch_size: min((i + 1) * batch_size, len(train_b_epoch))]

            # discriminator training
            for _ in range(model.config['discriminator_repetition']):
                gen_b_tmp = model.generator_b.generate(train_a_batch)
                gen_a_tmp = model.generator_a.generate(train_b_batch)

                dec_b = model.discriminator_b(train_b_batch)  # + torch.rand(train_b_batch.shape) * 0.6 - 0.3)
                loss_d_b_train = model.discriminator_b.score_train(dec_b)
                dec_gen_b = model.discriminator_b(gen_b_tmp)  # + torch.rand(gen_b_tmp.shape) * 0.6 - 0.3)
                loss_d_b_gen = model.discriminator_b.score_gen(dec_gen_b)
                loss_d_b = loss_d_b_train * model.config['delta1'] + loss_d_b_gen * model.config['delta2']

                dec_a = model.discriminator_a(train_a_batch)  # + torch.rand(train_a_batch.shape) * 0.6 - 0.3)
                loss_d_a_train = model.discriminator_a.score_train(dec_a)
                dec_gen_a = model.discriminator_a(gen_a_tmp)  # + torch.rand(gen_a_tmp.shape) * 0.6 - 0.3)
                loss_d_a_gen = model.discriminator_a.score_gen(dec_gen_a)
                loss_d_a = loss_d_a_train * model.config['delta1'] + loss_d_a_gen * model.config['delta2']

                loss_d_total = loss_d_a + loss_d_b
                loss_d_total.backward()

                model.optimize_discriminator()
                model.update_losses_batch(0., 0., loss_d_a.item() * train_a_batch.shape[0] / train_a.shape[0],
                                          loss_d_b.item() * train_b_batch.shape[0] / train_b.shape[0])

            # generator training
            for _ in range(model.config['generator_repetition']):
                gen_b = model.generator_b(train_a_batch)
                gen_a = model.generator_a(train_b_batch)

                dec_gen_b_tmp = model.discriminator_b(gen_b)
                dec_gen_a_tmp = model.discriminator_a(gen_a)

                loss_gan_a = model.generator_a.score_gan(dec_gen_a_tmp)
                loss_gan_b = model.generator_b.score_gan(dec_gen_b_tmp)
                loss_gen_a = model.generator_a.score_generated(train_b_batch, gen_b)
                loss_gen_b = model.generator_b.score_generated(train_a_batch, gen_a)

                loss_g_a = loss_gen_a * model.config['gamma1'] + loss_gan_a * model.config['gamma3']
                loss_g_b = loss_gen_b * model.config['gamma1'] + loss_gan_b * model.config['gamma3']

                if model.config['cyclic_loss_train']:
                    cyc_b = model.generator_b(gen_a)
                    cyc_a = model.generator_a(gen_b)

                    loss_cyc = (model.generator_b.score_cyclic(train_a_batch, cyc_a, train_b_batch, cyc_b) *
                                model.config['gamma2'])
                    loss_g_a += loss_cyc / 2
                    loss_g_b += loss_cyc / 2

                loss_g_total = loss_g_a + loss_g_b
                loss_g_total.backward()

                model.optimize_generator()
                model.update_losses_batch(loss_g_a.item() * train_b_batch.shape[0] / train_b.shape[0],
                                          loss_g_b.item() * train_a_batch.shape[0] / train_a.shape[0], 0., 0.)

        # training of the remaining data with cycle and neighbour loss
        for i in range(batch_num_rest):
            rest_a_batch = rest_a_epoch[i * batch_size_rest_a: min((i + 1) * batch_size_rest_a, rest_a_epoch.shape[0])]
            rest_b_batch = rest_b_epoch[i * batch_size_rest_b: min((i + 1) * batch_size_rest_b, rest_b_epoch.shape[0])]

            gen_b = model.generator_b(rest_a_batch)
            gen_a = model.generator_a(rest_b_batch)
            dec_gen_b_tmp = model.discriminator_b(gen_b)
            dec_gen_a_tmp = model.discriminator_a(gen_a)

            loss_gan_a = model.generator_a.score_gan(dec_gen_a_tmp) * model.config['epsilon3']
            loss_gan_b = model.generator_b.score_gan(dec_gen_b_tmp) * model.config['epsilon3']

            loss_a_old = None
            loss_b_old = None
            if model.config['cyclic_loss_test']:
                cyc_b = model.generator_b(gen_a)
                cyc_a = model.generator_a(gen_b)

                loss_cyc = (model.generator_a.score_cyclic(rest_a_batch, cyc_a, rest_b_batch, cyc_b) *
                            model.config['epsilon2'])
                loss_a = loss_cyc / 2 + loss_gan_a
                loss_b = loss_cyc / 2 + loss_gan_b

                if model.config['margin_loss']:
                    loss_a_old = loss_a
                    loss_b_old = loss_b
                else:
                    total_loss = loss_a + loss_b
                    total_loss.backward()
                    model.optimize_generator()
                    model.update_losses_batch(loss_a.item() * rest_a_batch.shape[0] / rest_a_epoch.shape[0],
                                              loss_b.item() * rest_b_batch.shape[0] / rest_b_epoch.shape[0], 0., 0.)

            # neighbour loss
            if model.config['margin_loss']:
                neighbours_a_batch = neighbours_a[idx_rest_a][i * batch_size_rest_a: min((i + 1) * batch_size_rest_a,
                                                                                         rest_a_epoch.shape[0])]
                neighbours_b_batch = neighbours_b[idx_rest_b][i * batch_size_rest_b: min((i + 1) * batch_size_rest_b,
                                                                                         rest_b_epoch.shape[0])]
                gen_b_train = model.generator_b.generate(train_a)
                gen_a_train = model.generator_a.generate(train_b)

                loss_a = (model.generator_a.score_unaligned(gen_a, gen_a_train[neighbours_b_batch]) *
                          model.config['epsilon1'])
                loss_b = (model.generator_b.score_unaligned(gen_b, gen_b_train[neighbours_a_batch]) *
                          model.config['epsilon1'])

                if loss_a_old is not None and loss_b_old is not None:
                    loss_a += loss_a_old
                    loss_b += loss_b_old
                else:
                    loss_a += loss_gan_a
                    loss_b += loss_gan_b

                loss_total = loss_a + loss_b
                loss_total.backward()

                model.optimize_generator()
                model.update_losses_batch(loss_a.item() * rest_a_batch.shape[0] / rest_a_epoch.shape[0],
                                          loss_b.item() * rest_b_batch.shape[0] / rest_b_epoch.shape[0], 0., 0.)

        # evaluation with validation data to find best performing model
        model.eval()
        with torch.no_grad():
            gen_a = model.generator_a.generate(valid_b)
            gen_b = model.generator_b.generate(valid_a)
            cyc_a = model.generator_a.generate(gen_b)
            cyc_b = model.generator_b.generate(gen_a)
            current_metrics = calculate_all_cycle_gan_metrics(valid_a, valid_b, gen_a, gen_b, cyc_a, cyc_b)
        model.complete_epoch(current_metrics)

        # early stopping criteria
        if max_score == 0 or current_metrics[2] > max_score:
            max_score = current_metrics[2]
            model.copy_model()
            counter = 0
            lr_change = False
        elif not (model.config['first100'] and j < 100):
            counter = counter + 1
        if model.config['early_stopping'] and counter > model.config['patience']:
            if model.config['adaptive_lr'] and counter < 80:
                if not lr_change:
                    model.restore_model()
                    model.change_lr(0.5)
                    lr_change = True
                    print(f"Changed learning rate of optimizer to: {model.current_lr}")
            else:
                model.config['iterations'] = j
                print(f"Stopped at epoch {j} due to early stopping with max Hits@1 of {max_score}.")
                iterator.close()
                break

        model.print_epoch_info()
    end = time.time()
    print(f"Training took {(end - start) / 60:.1f}min, with "
          f"{(end - start) / model.config['iterations']:.2f}s/iteration.")


def __initialize_generator(train_a, train_b, model, device):
    """
    Initialize the first layer of the generators by replacing the weights with the results from the transformation
    matrices
    :param train_a: train alignments of KG 1
    :param train_b: train alignments of KG 2
    :param model: the model, for which the generators shall be initialized
    :param device: the torch device on which the computations are done
    """
    train_a_copy = train_a.cpu()
    train_b_copy = train_b.cpu()
    transform_ab = np.linalg.lstsq(train_a_copy, train_b_copy, rcond=None)[0]
    transform_ba = np.linalg.lstsq(train_b_copy, train_a_copy, rcond=None)[0]
    transform_ab = torch.tensor(transform_ab, device=device)
    transform_ba = torch.tensor(transform_ba, device=device)
    # Replace the generator layers
    model.generator_a.network[0].bias = None
    model.generator_b.network[0].bias = None
    model.generator_a.network[0].weight.data = transform_ba.transpose(0, 1)
    model.generator_b.network[0].weight.data = transform_ab.transpose(0, 1)


def __train(gan_mode, device, config, data_a, data_b, evaluation=True):
    """
    This function is the interface to start training a model after the embeddings are loaded
    :param gan_mode: the model type, either regression or cycle GAN
    :param config: a dictionary with training and model parameters
    :param device: the torch device on which the calculations are executed
    :param data_a: the tuple with five sets from KG 1 (including all evaluation sets)
    :param data_b: the tuple with five sets from KG 2 (including all evaluation sets)
    :param evaluation: if True (default) then the model is evaluated and the results are printed
    :return: the trained model, which can be used for further test or it can be stored
    """
    train_a, valid_a, test_a, rest_a, eval_a = data_a
    train_b, valid_b, test_b, rest_b, eval_b = data_b

    if not gan_mode:
        print("##### Train new cycle regression model #####")
        model = RegressionModel(device, config)
        if config['initialize_generator']:
            __initialize_generator(train_a, train_b, model, device)
        __train_cycle_regression(model, train_a, train_b, valid_a, valid_b, rest_a, rest_b)
    else:
        print("##### Train new cycle gan model #####")
        model = CycleGAN(device, config)
        if config['initialize_generator']:
            __initialize_generator(train_a, train_b, model, device)
        __train_cycle_gan(model, train_a, train_b, valid_a, valid_b, rest_a, rest_b)

    model.restore_model()
    if evaluation:
        model.eval()
        with torch.no_grad():
            gen_a = model.generator_a.generate(test_b)
            gen_b = model.generator_b.generate(test_a)
            cyc_a = model.generator_a.generate(gen_b)
            cyc_b = model.generator_b.generate(gen_a)
            print("Test metrics")
            test_results = calculate_all_cycle_gan_metrics(eval_a, eval_b, gen_a, gen_b, cyc_a, cyc_b,
                                                           combined_results=False)
        return model, test_results
    return model


def __load_data(config, device):
    """
    This script loads the embeddings and the alignment for the subset defined in the config. Additionally, the
    alignments are split and the evaluation sets are prepared.
    :param config: a dictionary with training and model parameters
    :param device: the torch device on which the calculations are executed
    :return: returns the embeddings of KG 1 as five tuple and the same for the embeddings of KG 2
    """
    # Load embeddings
    embeddings_a, embeddings_b = load_pretrained_embeddings(config['model_type'], config['dataset'], config['subset'],
                                                            device, without_relations=True)
    config['dim_a'] = embeddings_a.shape[1]
    config['dim_b'] = embeddings_b.shape[1]

    # Load alignments
    if config['dataset'] == 0:
        train_alignment, test_alignment = load_dbp15k_alignment(config['subset'])
    else:
        aligned_entities, _ = load_wk3l_15k_alignment(config['subset'])
        split = int(0.3 * aligned_entities.shape[0])  # use the 30/70 split from DBP15k
        rand_split = random_split(aligned_entities, [split, aligned_entities.shape[0] - split])
        train_alignment = aligned_entities[rand_split[0].indices]
        test_alignment = aligned_entities[rand_split[1].indices]

    # Take random validation set from train alignments
    alignment_a = train_alignment[:, 0]
    alignment_b = train_alignment[:, 1]
    alignment_split = int(config['validation_split'] * alignment_a.shape[0])
    alignment_rand_split = random_split(alignment_a, [alignment_split, alignment_a.shape[0] - alignment_split])
    train_alignment_a = alignment_a[alignment_rand_split[0].indices]
    validation_alignment_a = alignment_a[alignment_rand_split[1].indices]
    train_alignment_b = alignment_b[alignment_rand_split[0].indices]
    validation_alignment_b = alignment_b[alignment_rand_split[1].indices]

    # Generate train, valid, test embeddings from alignments
    train_a, valid_a, test_a, exclusive_a = embedding_partitioning(embeddings_a, train_alignment_a,
                                                                   test_alignment[:, 0], validation_alignment_a)
    train_b, valid_b, test_b, exclusive_b = embedding_partitioning(embeddings_b, train_alignment_b,
                                                                   test_alignment[:, 1], validation_alignment_b)

    config['train_size'] = train_a.shape[0]
    config['validation_size'] = valid_a.shape[0]
    config['test_size'] = test_a.shape[0]

    if config['evaluation'] == 0:
        eval_a = test_a
        eval_b = test_b
    elif config['evaluation'] == 1:
        eval_a = torch.cat([test_a, valid_a, exclusive_a], dim=0)
        eval_b = torch.cat([test_b, valid_b, exclusive_b], dim=0)
    else:
        eval_a = torch.cat([test_a, valid_a, exclusive_a, train_a], dim=0)
        eval_b = torch.cat([test_b, valid_b, exclusive_b, train_b], dim=0)

    rest_a = torch.cat([test_a, valid_a, exclusive_a], dim=0)
    rest_b = torch.cat([test_b, valid_b, exclusive_b], dim=0)

    return (train_a, valid_a, test_a, rest_a, eval_a), (train_b, valid_b, test_b, rest_b, eval_b)


def __get_closest_neighbour(unaligned_source, aligned_source):
    """
    A function to search for the nearest neighbour in the aligned embeddings for each embedding in the unaligned set
    :param unaligned_source: the set of embeddings that are not aligned
    :param aligned_source: the set of embeddings that are part of a training alignment
    """
    unaligned_target_indices = []
    for x in unaligned_source:
        distances = torch.mean(torch.abs(x.view(1, -1) - aligned_source), dim=1)
        _, indices = torch.sort(distances, descending=False)
        unaligned_target_indices.append(indices[0])
    return torch.tensor(unaligned_target_indices, dtype=torch.long, device=unaligned_source.device)
