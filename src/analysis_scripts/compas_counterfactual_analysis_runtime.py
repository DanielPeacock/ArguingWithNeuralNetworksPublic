from __future__ import annotations  # noqa: D100, INP001

import argparse
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.append("src/")

import numpy as np
from sklearn.calibration import LabelEncoder
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


# Loading dataset
def load_compas(df):  # noqa: ANN001, ANN201, C901, D103, PLR0915
    # Not relevant
    del df["name"]
    del df["first"]
    del df["last"]
    df["age"] = df["age_cat"]
    del df["age_cat"]  # Alreafy in age
    del df["dob"]  # Already in age
    del df["vr_case_number"]
    del df["r_case_number"]
    del df["c_case_number"]
    del df["days_b_screening_arrest"]

    # Potentially useless
    del df["c_offense_date"]
    del df["c_jail_in"]
    del df["c_jail_out"]
    del df["event"]
    del df["start"]
    del df["end"]

    # Very partial and potentially useless
    del df["r_days_from_arrest"]
    del df["r_jail_in"]
    del df["r_jail_out"]
    del df["r_offense_date"]

    # There is another better cleaned column (and/or less empty)
    del df["r_charge_degree"]
    del df["vr_charge_degree"]
    del df["r_charge_desc"]

    # Almost empty
    del df["vr_offense_date"]
    del df["vr_charge_desc"]
    del df["c_arrest_date"]

    # Empty
    del df["violent_recid"]

    # Duplicates
    del df["priors_count.1"]

    # Only one unique value
    del df["v_type_of_assessment"]
    del df["type_of_assessment"]

    # Prediction of COMPAS
    del df["v_decile_score"]
    del df["score_text"]
    del df["screening_date"]
    del df["decile_score.1"]
    del df["v_screening_date"]
    del df["v_score_text"]
    del df["compas_screening_date"]
    del df["c_days_from_compas"]
    del df["decile_score"]

    # Custody
    df = df.dropna()  # noqa: PD901
    df["custody"] = (
        (
            df["out_custody"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))  # noqa: DTZ007
            - df["in_custody"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))  # noqa: DTZ007
        )
        .apply(lambda x: x.total_seconds() / 3600 / 24)
        .astype(int)
    )
    del df["out_custody"]
    del df["in_custody"]

    def summarise_charge(x):  # noqa: ANN001, ANN202, C901, PLR0912
        drugs = [
            "clonaz",
            "heroin",
            "cocaine",
            "cannabi",
            "drug",
            "pyrrolidin",
            "Methyl",
            "MDMA",
            "Ethylone",
            "Alprazolam",
            "Oxycodone",
            "Methadone",
            "Methamph",
            "Bupren",
            "Lorazepam",
            "controlled",
            "Amphtamine",
            "contro",
            "cont sub",
            "rapher",
            "fluoro",
            "ydromor",
            "methox",
            "iazepa",
            "XLR11",
            "steroid",
            "morphin",
            "contr sub",
            "enzylpiper",
            "butanediol",
            "phentermine",
            "Fentanyl",
            "Butylone",
            "Hydrocodone",
            "LSD",
            "Amobarbital",
            "Amphetamine",
            "Codeine",
            "Carisoprodol",
        ]
        drugs_selling = ["sel", "del", "traf", "manuf"]
        if sum([d.lower() in x.lower() for d in drugs]) > 0:
            if sum([h in x.lower() for h in drugs_selling]) > 0:
                x = "Drug Traffic"
            else:
                x = "Drug Possess"
        elif "murd" in x.lower() or "manslaughter" in x.lower():
            x = "Murder"
        elif (
            "sex" in x.lower()
            or "porn" in x.lower()
            or "voy" in x.lower()
            or "molest" in x.lower()
            or "exhib" in x.lower()
        ):
            x = "Sex Crime"
        elif "assault" in x.lower() or "carjacking" in x.lower():
            x = "Assault"
        elif (
            "child" in x.lower()
            or "domestic" in x.lower()
            or "negle" in x.lower()
            or "abuse" in x.lower()
        ):
            x = "Family Crime"
        elif "batt" in x.lower():
            x = "Battery"
        elif (
            "burg" in x.lower()
            or "theft" in x.lower()
            or "robb" in x.lower()
            or "stol" in x.lower()
        ):
            x = "Theft"
        elif (
            "fraud" in x.lower()
            or "forg" in x.lower()
            or "laund" in x.lower()
            or "countrfeit" in x.lower()
            or "counter" in x.lower()
            or "credit" in x.lower()
        ):
            x = "Fraud"
        elif "prost" in x.lower():
            x = "Prostitution"
        elif "trespa" in x.lower() or "tresspa" in x.lower():
            x = "Trespass"
        elif "tamper" in x.lower() or "fabricat" in x.lower():
            x = "Tampering"
        elif (
            "firearm" in x.lower()
            or "wep" in x.lower()
            or "wea" in x.lower()
            or "missil" in x.lower()
            or "shoot" in x.lower()
        ):
            x = "Firearm"
        elif "alking" in x.lower():
            x = "Stalking"
        elif "dama" in x.lower():
            x = "Damage"
        elif (
            "driv" in x.lower()
            or "road" in x.lower()
            or "speed" in x.lower()
            or "dui" in x.lower()
            or "d.u.i." in x.lower()
        ):
            x = "Driving"

        else:
            x = "Other"

        return x

    df["charge_desc"] = df["c_charge_desc"].apply(summarise_charge)
    del df["c_charge_desc"]

    CUSTODY_RANGES = {  # noqa: N806
        (0, 1): "0 days",
        #         (1,2): '1 day',
        #         (2,5): '2-4 days',
        #         (5,10): '5-9 days',
        (1, 10): "1-9 days",
        #         (10,30): '10-29 days',
        #         (30,90): '1-3 months',
        #         (90,365): '3-12 months',
        (10, 30): "10-29 days",
        (30, 365): "1-12 months",
        #         (365,365*2): '1 year',
        #         (365*2,365*3): '2 years',
        (365 * 1, 365 * 3): "1-2 years",
        (365 * 3, 365 * 5): "3-4 years",
        #         (365*5,365*10): '5-9 years',
        (365 * 5, df["custody"].max() + 1): "5 years or more",
        #         (365*10, df['custody'].max()+1): '10 years or more'
    }

    PRIORS_RANGES = {  # noqa: N806
        (0, 1): "0 priors",
        (1, 2): "1 priors",
        #         (2,3): '2 priors',
        #         (3,5): '3-4 priors',
        (2, 5): "2-4 priors",
        (5, 10): "5-9 priors",
        (10, df["priors_count"].max() + 1): "10 priors or more",
    }
    JUV_OTHER_RANGES = {  # noqa: N806
        (0, 1): "0 juv others",
        (1, 2): "1 juv others",
        #         (2,3): '2 juv others',
        #         (3,5): '3-4 juv others',
        (2, 5): "2-4 juv others",
        (5, df["juv_other_count"].max() + 1): "5 or more juv others",
    }
    JUV_FEL_RANGES = {  # noqa: N806
        (0, 1): "0 juv fel",
        (1, 2): "1 juv fel",
        #         (2,3): '2 juv fel',
        #         (3,5): '3-4 juv fel',
        (2, 5): "2-4 juv fel",
        (5, df["juv_fel_count"].max() + 1): "5 or more juv fel",
    }
    JUV_MISD_RANGES = {  # noqa: N806
        (0, 1): "0 juv misd",
        (1, 2): "1 juv misd",
        #         (2,3): '2 juv misd',
        #         (3,5): '3-4 juv misd',
        (2, 5): "2-4 juv misd",
        (5, df["juv_misd_count"].max() + 1): "5 or more juv misd",
    }

    def get_range(x, RANGES):  # noqa: ANN001, ANN202, N803
        for (a, b), label in RANGES.items():
            if x >= a and x < b:
                return label
        return None

    df["custody"] = df["custody"].apply(lambda x: get_range(x, CUSTODY_RANGES))
    df["priors_count"] = df["priors_count"].apply(lambda x: get_range(x, PRIORS_RANGES))
    df["juv_other_count"] = df["juv_other_count"].apply(
        lambda x: get_range(x, JUV_OTHER_RANGES),
    )
    df["juv_fel_count"] = df["juv_fel_count"].apply(
        lambda x: get_range(x, JUV_FEL_RANGES),
    )
    df["juv_misd_count"] = df["juv_misd_count"].apply(
        lambda x: get_range(x, JUV_MISD_RANGES),
    )

    df["is_recid"] = df["is_violent_recid"].apply(lambda x: "Yes" if x == 1 else "No")
    df["is_violent_recid"] = df["is_violent_recid"].apply(
        lambda x: "Yes" if x == 1 else "No",
    )
    df["two_year_recid"] = df["two_year_recid"].apply(
        lambda x: "Yes" if x == 1 else "No",
    )
    df["charge_degree"] = df["c_charge_degree"].apply(
        lambda x: "Felony" if x == "F" else "Misdemeanor",
    )
    del df["c_charge_degree"]

    return df


seed = 2025
compas = pd.read_csv("data/compas-scores-two-years.csv")
compas = load_compas(compas)

X_compas = compas.drop(columns=["two_year_recid"])
y_compas = compas[["two_year_recid"]]

encoder_compas = LabelEncoder()
X_compas = X_compas.apply(encoder_compas.fit_transform)
y_compas = y_compas.apply(encoder_compas.fit_transform)

X_compas = X_compas.to_numpy()
y_compas = y_compas.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X_compas,
    y_compas,
    test_size=0.2,
    random_state=seed,
)

scaler_compas = MinMaxScaler()
h = 1e-8
scaler_compas.fit(X_train)
X_train_compas = scaler_compas.transform(X_train)
X_test_compas = np.clip(scaler_compas.transform(X_test), h, 1 - h)

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

input_feature_names = compas.drop(columns=["two_year_recid"]).columns.tolist()
output_names = ["two_year_recid"]

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
    min_accuracy = 0.65

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
    f"outputs/compas/ce/runtime_mlp_{size_str}.npz",
    timings_original=timings_original,
    timings_sparse=timings_sparse,
)
