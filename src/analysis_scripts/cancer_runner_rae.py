# Data preprocessing  # noqa: D100, INP001
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rae_analysis_logic
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo

Path("models/cancer").mkdir(parents=True, exist_ok=True)
Path("outputs/cancer/rae").mkdir(parents=True, exist_ok=True)

seed = 2025

breast_cancer = fetch_ucirepo(id=17)

X_cancer = breast_cancer.data.features
y_cancer = breast_cancer.data.targets

encoder_cancer = LabelEncoder()
y_cancer = y_cancer.apply(encoder_cancer.fit_transform)

X_cancer = X_cancer.to_numpy()
y_cancer = y_cancer.to_numpy()

X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer,
    y_cancer,
    test_size=0.2,
    random_state=seed,
)

h = 1e-8
scaler_cancer = MinMaxScaler()
scaler_cancer.fit(X_train_cancer)
X_train_cancer = scaler_cancer.transform(X_train_cancer)
X_test_cancer = np.clip(scaler_cancer.transform(X_test_cancer), h, 1 - h)

y_train_cancer = y_train_cancer.reshape(-1)
y_test_cancer = y_test_cancer.reshape(-1)

input_names_cancer = breast_cancer.data.features.columns.tolist()
output_names_cancer = breast_cancer.data.targets.columns.tolist()
topic_arg_cancer = "Diagnosis"


parser = argparse.ArgumentParser(
    description="Train and analyse MLPs for cancer dataset.",
)
parser.add_argument(
    "--hidden_layer_size",
    type=str,
    required=True,
    help="Hidden layer sizes for the MLP. Example: --hidden_layer_size 10_20_30",
)

args = parser.parse_args()

hidden_layer_size = tuple(map(int, args.hidden_layer_size.split("_")))

model_file = "models/cancer/mlp" + "_".join(map(str, hidden_layer_size)) + ".joblib"
accuracy_file = "models/cancer/mlp" + "_".join(map(str, hidden_layer_size)) + ".txt"


def train_mlp(size: tuple) -> None:
    """Train a MLP for a hidden layer size."""
    rae_analysis_logic.make_mlp(
        X_train_cancer,
        y_train_cancer,
        X_test_cancer,
        y_test_cancer,
        size,
        model_file,
        accuracy_file,
    )


def analyse_mlp(size: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Run analysis for a hidden layer size."""
    return rae_analysis_logic.run_analysis(
        size,
        X_train_cancer,
        y_train_cancer,
        X_test_cancer,
        y_test_cancer,
        output_names_cancer,
        input_names_cancer,
        topic_arg_cancer,
        model_file,
    )


train_mlp(hidden_layer_size)
qbaf_objects, rae_objects, sparx_objects = analyse_mlp(hidden_layer_size)

Path("outputs/cancer/rae").mkdir(parents=True, exist_ok=True)

path_removal = (
    "outputs/cancer/rae/mlp" + "_".join(map(str, hidden_layer_size)) + "_removal.npz"
)
path_shap = (
    "outputs/cancer/rae/mlp" + "_".join(map(str, hidden_layer_size)) + "_shap.npz"
)

rae_analysis_logic.generate_dicts(
    rae_analysis_logic.ScoreType.REMOVAL,
    rae_objects,
    sparx_objects,
    qbaf_objects,
    path_removal,
    input_names_cancer,
    output_names_cancer,
)

rae_analysis_logic.generate_dicts(
    rae_analysis_logic.ScoreType.SHAPLEY,
    rae_objects,
    sparx_objects,
    qbaf_objects,
    path_shap,
    input_names_cancer,
    output_names_cancer,
)
