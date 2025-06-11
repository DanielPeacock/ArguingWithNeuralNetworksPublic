"""Analysis script for Covertype dataset using SpArX, clustering using AAEs."""  # noqa: INP001

import sys

sys.path.append("src/")
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from mlp_to_qbaf_converter.utils import forward_pass_dataset, logistic
from sparx import sparx

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--layers",
    type=int,
    required=True,
    help="Number of hidden layers in the MLP.",
)
argparser.add_argument(
    "--neurons",
    type=int,
    required=True,
    help="Number of neurons in each hidden layer.",
)
args = argparser.parse_args()

print("Loading dataset...")
covertype = pd.read_csv("data/covtype.csv")

print("Preprocessing dataset...")

X_covertype = covertype.drop(columns=["Cover_Type"])
y_covertype = covertype[["Cover_Type"]]

encoder = LabelEncoder()
y_covertype = y_covertype.apply(encoder.fit_transform)

input_names = X_covertype.columns.tolist()
output_names = encoder.classes_.tolist()

X_covertype = X_covertype.to_numpy()
y_covertype = y_covertype.to_numpy()

X_train_covertype, X_test_covertype, y_train_covertype, y_test_covertype = (
    train_test_split(X_covertype, y_covertype, test_size=0.2, random_state=42)
)

h = 1e-8

scaler_covertype = MinMaxScaler()

scaler_covertype.fit(X_train_covertype)
X_train_covertype = scaler_covertype.transform(X_train_covertype)
X_test_covertype = np.clip(scaler_covertype.transform(X_test_covertype), h, 1 - h)

y_train_covertype = y_train_covertype.reshape(-1)
y_test_covertype = y_test_covertype.reshape(-1)


def input_output_unfaithfulness(model, sparse_model, local_dataset, example_weights):  # noqa: ANN001, ANN201, D103
    weights_original, biases_original = model
    weights_sparse, biases_sparse = sparse_model

    original_output = forward_pass_dataset(
        local_dataset[:, :-1],
        weights_original,
        biases_original,
        logistic,
    )[-1]

    sparse_output = forward_pass_dataset(
        local_dataset[:, :-1],
        weights_sparse,
        biases_sparse,
        logistic,
    )[-1]

    diff = sparse_output - original_output
    diff = np.power(diff, 2)
    total = np.sum(diff, axis=1)
    total = np.multiply(total, example_weights)
    total = np.sum(diff)

    return total / np.sum(example_weights)


def structural_unfaithfulness(model, sparse_model, X_test, sp, example_num):  # noqa: ANN001, ANN201, D103, N803
    weights_original, biases_original = model
    weights_sparse, biases_sparse = sparse_model
    preserve_percent = sp.preserve_percentage / 100
    original_activations = forward_pass_dataset(
        X_test,
        weights_original,
        biases_original,
        logistic,
    )
    sparse_activations = forward_pass_dataset(
        X_test,
        weights_sparse,
        biases_sparse,
        logistic,
    )

    structural_unfaithfulness = 0

    for i, original_activation in enumerate(original_activations):
        sparse_activation = sparse_activations[i]
        if i != 0:
            for cluster_label in range(
                int(len(biases_original[i - 1]) * preserve_percent),
            ):
                if cluster_label in sp.cluster_labels[i - 1]:
                    structural_unfaithfulness += np.sum(
                        np.abs(
                            np.mean(
                                original_activation[
                                    example_num,
                                    sp.cluster_labels[i - 1] == cluster_label,
                                ],
                            )
                            - sparse_activation[example_num, cluster_label],
                        ),
                    )
        else:
            structural_unfaithfulness += np.sum(
                np.abs(
                    sparse_activation[example_num, :]
                    - original_activation[example_num, :],
                ),
            )

    number_of_nodes = sum(sp.get_sparsified_shape()[1:])
    structural_unfaithfulness /= number_of_nodes

    return structural_unfaithfulness


sparse_percent = 80

layers = args.layers
no_neurons = args.neurons

model_name = f"classifier_covertype_{no_neurons}_neurons_{layers}_layers"
hidden_layers = tuple([no_neurons] * layers)

print("Training model with hidden layers:", hidden_layers)
if not Path(model_name).exists():
    classifier_covertype = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="logistic",
        max_iter=50,
        random_state=2025,
        solver="adam",
        learning_rate_init=0.01,
        batch_size=256,
        early_stopping=True,
    )
    classifier_covertype.fit(X_train_covertype, y_train_covertype)
    joblib.dump(classifier_covertype, model_name)
else:
    classifier_covertype = joblib.load(model_name)

print("Model trained.")

print("Evaluating model...")

y_pred_covertype = classifier_covertype.predict(X_test_covertype)
print(classification_report(y_test_covertype, y_pred_covertype))

print("Evaluating SpArX...")

train_set_full = np.column_stack(
    (
        X_train_covertype,
        y_train_covertype,
    ),
)

topic_arg = output_names[0]
kernel_size = np.sqrt(X_test_covertype.shape[1]) * 0.75
model_original = (
    classifier_covertype.coefs_,
    classifier_covertype.intercepts_,
)

results_io = []
results_struct = []


def process_example(example_num):  # noqa: ANN001, ANN201, D103
    example = X_test_covertype[example_num]
    example_row = np.append(example, y_test_covertype[example_num])

    sp = sparx.LocalSpArX(
        classifier_covertype.coefs_,
        classifier_covertype.intercepts_,
        "logistic",
        sparse_percent,
        example_row,
        train_set_full,
        kernel_size,
        input_names,
        output_names,
        topic_arg,
        sparx.ClusteringMethod.AAE_GRADIENT,
    )

    s_weights, s_biases = sp.get_sparsified_mlp()
    model_sparse = (s_weights, s_biases)
    local_dataset = sp.local_dataset
    example_weights = sp.example_weights

    io_unfaithfulness = input_output_unfaithfulness(
        model_original,
        model_sparse,
        local_dataset,
        example_weights,
    )
    struct_unfaithfulness = structural_unfaithfulness(
        model_original,
        model_sparse,
        X_test_covertype,
        sp,
        example_num,
    )

    print(f"Example {example_num} processed.", flush=True)

    return io_unfaithfulness, struct_unfaithfulness


results = Parallel(n_jobs=-1)(
    delayed(process_example)(example_num)
    for example_num in range(len(X_test_covertype))
)

results_io, results_struct = zip(*results)
sp_result_io = np.mean(results_io)
sp_result_struct = np.mean(results_struct)

print("Input-output unfaithfulness:", sp_result_io)
print("Structural unfaithfulness:", sp_result_struct)
print("Saving results...")
Path("covertype_sparx_aae_results").mkdir(parents=True, exist_ok=True)
np.savez(
    f"covertype_sparx_aae_results/covertype_aae_sparx_{layers}_{no_neurons}.npz",
    input_output_unfaithfulness=results_io,
    structural_unfaithfulness=results_struct,
)
print("Results saved.")
print("Done. Exiting.")
