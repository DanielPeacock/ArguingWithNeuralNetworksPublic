# Data preprocessing  # noqa: D100, INP001
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import aae_analysis_logic
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if TYPE_CHECKING:
    import numpy as np

Path("models/iris").mkdir(parents=True, exist_ok=True)
Path("outputs/iris/aae").mkdir(parents=True, exist_ok=True)

seed = 2025
cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris = pd.read_csv("data/iris.data", names=cols)

label_encoder = LabelEncoder()

iris["class"] = label_encoder.fit_transform(iris["class"])

X = iris.drop(columns="class").to_numpy()
y = iris["class"].to_numpy()

input_feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
output_names = list(label_encoder.classes_)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=2024,
)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

parser = argparse.ArgumentParser(
    description="Train and analyse MLPs for iris dataset.",
)
parser.add_argument(
    "--hidden_layer_size",
    type=str,
    required=True,
    help="Hidden layer sizes for the MLP. Example: --hidden_layer_size 10_20_30",
)

args = parser.parse_args()

hidden_layer_size = tuple(map(int, args.hidden_layer_size.split("_")))

model_file = "models/iris/mlp" + "_".join(map(str, hidden_layer_size)) + ".joblib"
accuracy_file = "models/iris/mlp" + "_".join(map(str, hidden_layer_size)) + ".txt"


def train_mlp(size: tuple) -> None:
    """Train a MLP for a hidden layer size."""
    aae_analysis_logic.make_mlp(
        X_train,
        y_train,
        X_test,
        y_test,
        size,
        model_file,
        accuracy_file,
    )


def analyse_mlp(size: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Run analysis for a hidden layer size."""
    return aae_analysis_logic.run_analysis(
        size,
        X_train,
        y_train,
        X_test,
        y_test,
        output_names,
        input_feature_names,
        output_names[0],
        model_file,
    )


train_mlp(hidden_layer_size)
qbaf_objects, aae_objects, sparx_objects = analyse_mlp(hidden_layer_size)

path_gradient = (
    "outputs/iris/aae/mlp" + "_".join(map(str, hidden_layer_size)) + "_gradient.npz"
)
path_removal = (
    "outputs/iris/aae/mlp" + "_".join(map(str, hidden_layer_size)) + "_removal.npz"
)
path_shap = "outputs/iris/aae/mlp" + "_".join(map(str, hidden_layer_size)) + "_shap.npz"


aae_analysis_logic.generate_dicts(
    aae_analysis_logic.ScoreType.GRADIENT,
    aae_objects,
    sparx_objects,
    qbaf_objects,
    path_gradient,
    input_feature_names,
    output_names,
    output_names[0],
)

aae_analysis_logic.generate_dicts(
    aae_analysis_logic.ScoreType.REMOVAL,
    aae_objects,
    sparx_objects,
    qbaf_objects,
    path_removal,
    input_feature_names,
    output_names,
    output_names[0],
)

aae_analysis_logic.generate_dicts(
    aae_analysis_logic.ScoreType.SHAPLEY,
    aae_objects,
    sparx_objects,
    qbaf_objects,
    path_shap,
    input_feature_names,
    output_names,
    output_names[0],
)
