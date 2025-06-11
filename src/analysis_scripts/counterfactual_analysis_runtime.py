from __future__ import annotations  # noqa: D100, INP001

import argparse
import sys

sys.path.append("src/")

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import Uncertainpy.src.uncertainpy.gradual as grad
from mlp_to_qbaf_converter.counterfactual_contestability_explanation import (
    CounterfactualContestabilityExplanation,
)
from mlp_to_qbaf_converter.mlp_to_qbaf import MLPToQBAF
from sparx.sparx import LocalSpArX

seed = 2025
data = pd.read_csv("data/diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

input_feature_names = list(X.columns)
output_names = ["Diabetes?"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=seed,
)

X_train, X_test, y_train, y_test = (
    X_train.to_numpy(),
    X_test.to_numpy(),
    y_train.to_numpy(),
    y_test.to_numpy(),
)

smote = SMOTE(random_state=seed, sampling_strategy="minority")
X_train, y_train = smote.fit_resample(X_train, y_train)

# Feature Scaling (Use StandardScaler instead of MinMaxScaler)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parser = argparse.ArgumentParser()


parser.add_argument(
    "--size",
    type=str,
    required=True,
    help="Size of the MLP.",
)

size = [int(s) for s in parser.parse_args().size.split("_")]


def train_mlp(
    size: tuple[int],
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
) -> MLPClassifier:
    """Train a MLPClassifier with the given size and data."""
    min_accuracy = 0.7

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    solvers = ["adam", "sgd", "lbfgs"]

    best_mlp = None
    best_score = 0
    params = {"alpha": alphas, "solver": solvers}
    grid_search = GridSearchCV(
        MLPClassifier(
            hidden_layer_sizes=size,
            activation="logistic",
            random_state=seed,
            max_iter=5000,
            early_stopping=True,
        ),
        params,
        cv=2,
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)
    best_mlp = grid_search.best_estimator_
    best_score = grid_search.best_score_

    if best_score < min_accuracy:
        msg = "Model accuracy is too low"
        raise RuntimeError(msg)

    return best_mlp


def create_new_qbaf(  # noqa:PLR0913
    original_qbaf: grad.BAG,
    sparse_qbaf: grad.BAG,
    sp: LocalSpArX,
    counterfactual_weights: dict[tuple[str, str], float],
    input_feature_names: list[str],
    output_names: list[str],
) -> dict[tuple[str, str], float]:
    """Estimate new counterfactual edge weights based on the sparse counterfactual."""
    new_weights = {}

    for arg in sparse_qbaf.arguments:
        if arg in input_feature_names:
            continue
        if arg in output_names:
            clusters = [arg]
            layer = "output"
            neuron = arg
        else:
            split_name = arg.split(" ")
            layer = int(split_name[1])
            neuron = int(split_name[3])
            clusters = sp.get_containing_neurons(layer, neuron)

        for n in clusters:
            original_name = n if layer == "output" else f"Layer {layer} Neuron {n}"

            original_arg = original_qbaf.arguments[original_name]
            original_attackers = original_arg.attackers
            original_supporters = original_arg.supporters

            num_attackers = len(original_attackers)
            num_supporters = len(original_supporters)
            num_edges = num_attackers + num_supporters

            for attacker in original_attackers:
                attacker_name = attacker.name
                if attacker_name in input_feature_names:
                    sparse_cluster_name = attacker_name
                else:
                    attacker_name_split = attacker_name.split(" ")
                    attacker_layer = int(attacker_name_split[1])
                    attacker_neuron = int(attacker_name_split[3])
                    sparse_cluster = (
                        sp.cluster_labels[attacker_layer - 1][attacker_neuron - 1] + 1
                    )
                    sparse_cluster_name = (
                        f"Layer {attacker_layer} Neuron {sparse_cluster}"
                    )

                original_relation = (attacker_name, original_name)
                sparse_relation = (sparse_cluster_name, arg)

                new_weights[original_relation] = abs(
                    counterfactual_weights[sparse_relation] / num_edges,
                )

            for supporter in original_supporters:
                supporter_name = supporter.name
                if supporter_name in input_feature_names:
                    sparse_cluster_name = supporter_name
                else:
                    supporter_name_split = supporter_name.split(" ")
                    supporter_layer = int(supporter_name_split[1])
                    supporter_neuron = int(supporter_name_split[3])
                    sparse_cluster = (
                        sp.cluster_labels[supporter_layer - 1][supporter_neuron - 1] + 1
                    )
                    sparse_cluster_name = (
                        f"Layer {supporter_layer} Neuron {sparse_cluster}"
                    )

                original_relation = (supporter_name, original_name)
                sparse_relation = (sparse_cluster_name, arg)

                new_weights[original_relation] = abs(
                    counterfactual_weights[sparse_relation] / num_edges,
                )

    return new_weights


def check_validity(
    new_qbaf: grad.BAG,
    desired_strength: float,
    delta: float,
    topic_arg_name: str,
) -> tuple[bool, float]:
    """Check if the new QBAF is valid based on the desired strength."""
    # Get the strength of the topic argument
    topic_arg = new_qbaf.arguments[topic_arg_name]
    topic_strength = topic_arg.strength

    # Check if the strength is greater than or equal to the desired strength
    dist = abs(desired_strength - topic_strength)
    return (dist < delta, dist)


# We start by checking how the average counterfactual changes by sparsifying the QBAF


X_test = np.clip(X_test, 0, 1)
train_set = np.column_stack((X_train, y_train))
sparsification_amounts = [10, 20, 30, 40, 50, 60, 70, 80, 90]

mlp = train_mlp(size, X_train, y_train)
neurons_per_layer = [
    mlp.n_features_in_,
    *list(mlp.hidden_layer_sizes),
    mlp.n_outputs_,
]

timings_original = np.zeros((len(X_test), len(sparsification_amounts)))
timings_sparse = np.zeros((len(X_test), len(sparsification_amounts)))

for example_num in tqdm(range(len(X_test)), desc="Example Num"):
    example = X_test[example_num]
    example_row = np.append(X_test[example_num], y_test[example_num])

    start_orig = time.time()

    original_qbaf = MLPToQBAF(
        neurons_per_layer,
        mlp.coefs_,
        mlp.intercepts_,
        "logistic",
        input_feature_names,
        output_names,
        example,
    ).get_qbaf()

    original_ce = CounterfactualContestabilityExplanation(
        original_qbaf,
        grad.SumAggregation(),
        grad.MLPBasedInfluence(),
        output_names[0],
        seed,
    ).get_ce_explanation()

    end_orig = time.time()

    for i, sparsification in enumerate(sparsification_amounts):
        sparse_time = time.time()

        sp = LocalSpArX(
            mlp.coefs_,
            mlp.intercepts_,
            "logistic",
            sparsification,
            example_row,
            train_set,
            np.sqrt(X_test.shape[1]) * 0.75,
        )

        sp_weights, sp_biases = sp.get_sparsified_mlp()

        sparse_qbaf = MLPToQBAF(
            sp.get_sparsified_shape(),
            sp_weights,
            sp_biases,
            "logistic",
            input_feature_names,
            output_names,
            example,
        ).get_qbaf()

        sparse_ce = CounterfactualContestabilityExplanation(
            sparse_qbaf,
            grad.SumAggregation(),
            grad.MLPBasedInfluence(),
            output_names[0],
            seed,
        ).get_ce_explanation()

        approx_weights = create_new_qbaf(
            original_qbaf,
            sparse_qbaf,
            sp,
            sparse_ce,
            input_feature_names,
            output_names,
        )

        fixed_ce = CounterfactualContestabilityExplanation(
            original_qbaf,
            grad.SumAggregation(),
            grad.MLPBasedInfluence(),
            output_names[0],
            seed,
            initial_cf_weights=approx_weights,
        ).get_ce_explanation()

        end_sparse = time.time()

        timings_sparse[example_num, i] = end_sparse - sparse_time
        timings_original[example_num, i] = end_orig - start_orig

        print(f"Runtime Sparse: {timings_sparse[example_num, i]}")
        print(f"Runtime Original: {timings_original[example_num, i]}")
        print(
            f"Sparse quicker? {
                timings_sparse[example_num, i] < timings_original[example_num, i]
            }",
        )
        print()

size_str = "_".join([str(s) for s in size])

np.savez_compressed(
    f"outputs/diabetes/ce/runtime_mlp_{size_str}.npz",
    timings_original=timings_original,
    timings_sparse=timings_sparse,
)
