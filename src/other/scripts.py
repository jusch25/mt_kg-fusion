"""This file contains several functions for to evaluate and report the results of link prediction and alignment models
with plots and statistics."""

from input_output import load_evaluation_results, load_dbp15k_data, load_wk3l_15k_alignment, load_dbp15k_alignment,\
    load_wk3l_15k_data, save_plot, load_ablation_details, load_configs, load_history
from constants import METRICS, REMOTE, FIGURE_PATH
from pathlib import Path
import os
import numpy as np


def final_results(directory, eval_set, separated=True):
    """Calculate and print the statistics of a final model. Final models are trained and evaluated multiple times to
    report average results and standard deviation."""
    results = load_evaluation_results(directory)
    if eval_set == 0:
        results = results[10:]
    elif eval_set == 1:
        results = results[5:10]
    elif eval_set == 2:
        results = results[:5]
    keys = list(results[0].keys())
    metrics = METRICS[:-1]
    values = np.array([[list(result[x].values()) for x in result] for result in results])[:, :2]
    avg_values = np.average(values, axis=0)
    std_values = np.std(values, axis=0)
    if not separated:
        avg_values = np.average(avg_values, axis=0)
        std_values = np.average(std_values, axis=0)
    if separated:
        for i in [1, 0]:
            print(keys[i])
            print(f"{metrics[2]}: {avg_values[i, 2]*100:.2f} {std_values[i, 2]*100:.2f}")
            print(f"{metrics[5]}: {avg_values[i, 5]*100:.2f} {std_values[i, 5]*100:.2f}")
            print(f"{metrics[1]}: {avg_values[i, 1]:.3f} {std_values[i, 1]:.3f}")
            print()
        print(f"& {avg_values[1, 2]*100:.2f} & {avg_values[1, 5]*100:.2f} & {avg_values[1, 1]:.3f} & "
              f"{avg_values[0, 2]*100:.2f} & {avg_values[0, 5]*100:.2f} & {avg_values[0, 1]:.3f}")
        print(f"& $\pm$ {std_values[1, 2]*100:.2f} & $\pm$ {std_values[1, 5]*100:.2f} & $\pm$ {std_values[1, 1]:.3f} & "
              f"$\pm$ {std_values[0, 2]*100:.2f} & $\pm$ {std_values[0, 5]*100:.2f} & $\pm$ {std_values[0, 1]:.3f}")
    else:
        print(f"& {avg_values[2]*100:.2f} $\pm$ {std_values[2]*100:.2f} & {avg_values[5]*100:.2f} "
              f"$\pm$ {std_values[5]*100:.2f} & {avg_values[1]:.3f} $\pm$ {std_values[1]:.3f}")


def dataset_statistics():
    """Print the statistics of all subsets of the entity alignment datasets DBP15k and WK3l-15k (number of entities,
    relations and triples of each graph and number of aligned entities)."""
    dbp15k_subsets = ["fr_en", "ja_en", "zh_en"]
    wk3l_15k_subset = ["en_fr", "en_de"]
    print("Dataset - Subset - Graph: Triples + Entities + Relations * Alignments")
    for x in dbp15k_subsets:
        data_1, e2id_1, r2id_1 = load_dbp15k_data(x, False, True)
        data_2, e2id_2, r2id_2 = load_dbp15k_data(x, True, True)
        train_alignment, test_alignment = load_dbp15k_alignment(x)
        print(f"DBP15k - {x} - {x[0:2]}: {len(data_1)} & {len(e2id_1)} & {len(r2id_1)} & "
              f"{len(train_alignment) + len(test_alignment)}")
        print(f"DBP15k - {x} - {x[3:5]}: {len(data_2)} & {len(e2id_2)} & {len(r2id_2)} & "
              f"{len(train_alignment) + len(test_alignment)}")
    for x in wk3l_15k_subset:
        data_1, e2id_1, r2id_1 = load_wk3l_15k_data(x, False, True)
        data_2, e2id_2, r2id_2 = load_wk3l_15k_data(x, True, True)
        alignments, _ = load_wk3l_15k_alignment(x)
        print(f"WK3l-15k - {x} - {x[0:2]}: {len(data_1)} & {len(e2id_1)} & {len(r2id_1)} & {len(alignments)}")
        print(f"WK3l-15k - {x} - {x[3:5]}: {len(data_2)} & {len(e2id_2)} & {len(r2id_2)} & {len(alignments)}")


def plot_hyperparam_eval(directory_name, title, x_label, mode=None, gan_mode=0, x_data_in=None, multiple=None,
                         all_dirs=False, repetition=3):
    """
    Create a pyplot of the hyper-parameter evaluation results from a directory. A hyper-parameter evaluation (see
    kg_alignment/entity_alignment/multi_hyperparam_study and kg_alignment/entity_alignment/single_hyperparam_study)
    trains models with varying parameters to test their impact on the model performance. Each paramter constellation is
    tested multiple times to report average results.
    Example: Ablation study for model architecture - parameters: batch-norm layer used (True, False), dropout layer used
    (True, False), activation layer used (True, False) => number of combinations: 2*2*2=8 - so 8 constellations are
    trained 3 times for example, leading to 24 trained models which are evaluated and the results are stored in one
    directory (the models itself are not saved).
    :param directory_name: the name of the directory of the parameter evaluation with all test results
    :param title: the title of the plot
    :param x_label: the description of the x-axis of the plot
    :param mode: (optional, default None) specifies which data series for the y-axis are used:
        None: only plot average values (of Hits@1)
        max-average-min: plot max + average + min values (of Hits@1)
        epochs: plot average values (of Hits@1) and number of epochs required for model training
        eval_sets: plot the Hits@1 for all three evaluation sets
        history: plots training history for MR, MRR, Hits@1 and Hits@10 on two y-axis
        metrics: plots the metrics MR, MRR, Hits@1 and Hits@10 on two y-axis
    :param gan_mode: (optional, default 0) specifies which entitiy alignment method was used: 0 for cycle
    regression model, 1 for cycle gan model
    :param x_data_in: (optional, default None) if provided, use as x-axis values (e.g. range of numbers or list of
    strings)
    :param multiple: (optional, default None) if provided, load a list of evaluation results (with same number of tests)
    in one graph as multiple data series
    :param all_dirs: read all results directories and plot each one
    :param repetition: how many times each test was repeated (default 3 for ablation study and other parameter tests) to
    reshape the arrays correctly
    """
    datasets = ["en_de", "fr_en"]
    if all_dirs:
        base_names = os.listdir(REMOTE)
        base_names = [x for x in base_names if x[0] == str(gan_mode) and "variance" not in x and "patience" not in x
                      and "adaptive" not in x]
        base_names = [x[9:] for x in base_names]
        base_names = list(set(base_names))
        y_label = "Hits@1"
        legend = ["test entities", "no train entities", "all entities"]
        for base_name in base_names:
            for x in datasets:
                x_data = load_ablation_details(f"{gan_mode}_{0}_{x}{base_name}", True)
                x_data = [str(x) for x in x_data]
                y_data_all = []
                for i in range(3):
                    directory = f"{gan_mode}_{i}_{x}{base_name}"
                    y_data = load_evaluation_results(directory, True)
                    y_data = np.array([[list(result[x].values()) for x in result] for result in y_data])
                    y_data_base = y_data.reshape((-1, repetition, 4, 6))
                    y_data = np.average(y_data_base, axis=1)[:, :2]
                    y_data = np.average(y_data, axis=1)
                    y_data = y_data[:, 2]
                    y_data_all.append(y_data)

                y_data_all = np.array(y_data_all)
                y_data = y_data_all.transpose()
                path = Path(FIGURE_PATH) / f"{gan_mode}_{base_name}-{x}.pdf"
                save_plot(x_data, y_data, path, title, x_label, y_label, legend, rotate_axis=True)

    else:
        directory = directory_name
        if x_data_in is None:
            x_data = load_ablation_details(directory, True)
        else:
            x_data = x_data_in
        x_data = [str(x) for x in x_data]
        y_data_2 = None
        y_label_2 = None

        if multiple is not None:
            y_data_all = []
            for name in multiple:
                directory = name
                y_data = load_evaluation_results(directory, True)
                y_data = np.array([[list(result[x].values()) for x in result] for result in y_data])
                y_data_base = y_data.reshape((-1, repetition, 4, 6))
                y_data = np.max(y_data_base, axis=1)[:, :2]
                y_data = np.average(y_data, axis=1)
                y_data = y_data[:, 2]
                y_data_all.append(y_data)
            y_data_all = np.array(y_data_all)
            y_data = y_data_all.transpose()
            y_label = "Hits@1"
            legend = ["Alternating training", "Initialized discriminator"]

        else:
            if mode == "history":
                y_data = load_history(directory, True)
                y_data = y_data[12]
                y_data_base = np.array([y_data[key] for key in y_data])
                y_data = y_data_base.transpose()
                y_data_2 = y_data[:, 0]
                y_label_2 = "Mean Rank"
                y_data = y_data[:, [1, 2, 5]]
                y_label = "MRR, Hits@k"
                legend = ["MRR", "Hits@1", "Hits@10"]
                x_data = range(y_data.shape[0])
            else:
                y_label = "Hits@1"
                legend = None
                y_data = load_evaluation_results(directory, True)
                y_data = np.array([[list(result[x].values()) for x in result] for result in y_data])
                y_data_base = y_data.reshape((-1, repetition, 4, 6))
                y_data = np.average(y_data_base, axis=1)[:, :2]
                y_data = np.average(y_data, axis=1)
                if mode == "metrics":
                    y_data_2 = y_data[:, 0]
                    y_label_2 = "Mean Rank"
                    y_data = y_data[:, [1, 2, 5]]
                    y_label = "MRR, Hits@k"
                    legend = ["MRR", "Hits@1", "Hits@10"]
                else:
                    y_data = y_data[:, 2]

            if mode == "max-avg-min":
                y_data2 = np.max(y_data_base, axis=1)[:, :2]
                y_data2 = np.average(y_data2, axis=1)
                y_data2 = y_data2[:, 2]
                y_data3 = np.min(y_data_base, axis=1)[:, :2]
                y_data3 = np.average(y_data3, axis=1)
                y_data3 = y_data3[:, 2]
                y_data = np.concatenate([y_data2.reshape(-1, 1), y_data.reshape(-1, 1), y_data3.reshape(-1, 1)],
                                        axis=1)
                y_label = "Hits@1"
                legend = ["Maximum", "Average", "Minimum"]
            elif mode == "epochs":
                y_data2 = load_configs(directory, True)
                y_data2 = np.array([x['iterations'] for x in y_data2])
                y_data2 = y_data2.reshape((-1, repetition))
                y_data_2 = np.average(y_data2, axis=1)
                y_label_2 = "Epochs"
            elif mode == "eval_sets":
                y_data2 = load_evaluation_results(f"{gan_mode}_1_{directory[4:]}", True)
                y_data2 = np.array([[list(result[x].values()) for x in result] for result in y_data2])
                y_data2 = y_data2.reshape((-1, repetition, 4, 6))
                y_data2 = np.average(y_data2, axis=1)[:, :2]
                y_data2 = np.average(y_data2, axis=1)
                y_data2 = y_data2[:, 2]
                y_data3 = load_evaluation_results(f"{gan_mode}_2_{directory[4:]}", True)
                y_data3 = np.array([[list(result[x].values()) for x in result] for result in y_data3])
                y_data3 = y_data3.reshape((-1, repetition, 4, 6))
                y_data3 = np.average(y_data3, axis=1)[:, :2]
                y_data3 = np.average(y_data3, axis=1)
                y_data3 = y_data3[:, 2]
                y_data = np.concatenate([y_data.reshape(-1, 1), y_data2.reshape(-1, 1), y_data3.reshape(-1, 1)],
                                        axis=1)
                y_label = "Hits@1"
                legend = ["test entities", "no train entities", "all entities"]

        path = Path(FIGURE_PATH) / f"{directory}+{mode}.pdf"
        save_plot(x_data, y_data, path, title, x_label, y_label, legend, rotate_axis=True, y_data_2=y_data_2,
                  second_y_label=y_label_2)


modes = {0: None, 1: "max-avg-min", 2: "epochs", 3: "eval_sets", 4: "history", 5: "metrics"}

d = "0_0_en_de_gen_norm_gen_batch_norm_gen_activation_gen_dropout"
t = "Influence of the generator layer components"
plot_hyperparam_eval(d, t, "Layer combinations", all_dirs=False, gan_mode=0, mode=modes[2], repetition=3)
