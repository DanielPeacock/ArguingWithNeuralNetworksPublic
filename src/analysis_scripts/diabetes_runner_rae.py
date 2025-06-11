# Data preprocessing  # noqa: D100, INP001
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import rae_analysis_logic
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if TYPE_CHECKING:
    import numpy as np

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


parser = argparse.ArgumentParser(
    description="Train and analyse MLPs for diabetes dataset.",
)
parser.add_argument(
    "--hidden_layer_size",
    type=str,
    required=True,
    help="Hidden layer sizes for the MLP. Example: --hidden_layer_size 10_20_30",
)

args = parser.parse_args()

hidden_layer_size = tuple(map(int, args.hidden_layer_size.split("_")))

model_file = "models/diabetes/mlp" + "_".join(map(str, hidden_layer_size)) + ".joblib"
accuracy_file = "models/diabetes/mlp" + "_".join(map(str, hidden_layer_size)) + ".txt"


def train_mlp(size: tuple) -> None:
    """Train a MLP for a hidden layer size."""
    rae_analysis_logic.make_mlp(
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
    return rae_analysis_logic.run_analysis(
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
qbaf_objects, rae_objects, sparx_objects = analyse_mlp(hidden_layer_size)

Path("outputs/diabetes/rae").mkdir(parents=True, exist_ok=True)

path_removal = (
    "outputs/diabetes/rae/mlp" + "_".join(map(str, hidden_layer_size)) + "_removal.npz"
)
path_shap = (
    "outputs/diabetes/rae/mlp" + "_".join(map(str, hidden_layer_size)) + "_shap.npz"
)

rae_analysis_logic.generate_dicts(
    rae_analysis_logic.ScoreType.REMOVAL,
    rae_objects,
    sparx_objects,
    qbaf_objects,
    path_removal,
    input_feature_names,
    output_names,
)

rae_analysis_logic.generate_dicts(
    rae_analysis_logic.ScoreType.SHAPLEY,
    rae_objects,
    sparx_objects,
    qbaf_objects,
    path_shap,
    input_feature_names,
    output_names,
)
