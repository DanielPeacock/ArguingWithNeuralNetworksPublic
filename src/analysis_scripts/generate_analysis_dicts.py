# noqa: D100, INP001
from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

sys.path.append("src/")

import Uncertainpy.src.uncertainpy.gradual as grad

if TYPE_CHECKING:
    from mlp_to_qbaf_converter.argument_attribution_explanation import AAE


class ScoreType(Enum):
    """The type of score to get from an AAE."""

    GRADIENT = 1
    REMOVAL = 2
    SHAPLEY = 3


def get_scores(aee: AAE, score_type: ScoreType) -> dict[str, float]:
    """Get the scores from an AAE object based on the score type.

    :param aee: The AAE object
    :param score_type: The type of score to get

    Returns:
        The scores.

    """
    match score_type:
        case ScoreType.GRADIENT:
            return aee.get_gradients()
        case ScoreType.REMOVAL:
            return aee.get_removal_scores()
        case ScoreType.SHAPLEY:
            return aee.get_shapley_values()
        case _:
            msg = "Invalid score type"
            raise ValueError(msg)


def generate_dicts(  # noqa: C901, PLR0912, PLR0915
    score_type: ScoreType,
) -> tuple[
    dict[Path, np.ndarray],
    dict[Path, np.ndarray],
    dict[Path, np.ndarray],
    dict[Path, np.ndarray],
]:
    """Generate dicts: max argument, top n scores and averaged scores and impact diff.

    :param score_type: The type of score to get

    Returns:
        The dictionaries.

    """
    max_argument_dict = {}
    top_n_scores_dict = {}
    averaged_scores_dict = {}
    removal_impact_dict = {}

    num_loads = len(list(Path("outputs/diabetes/").glob("mlp*")))
    i = 1
    input_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    topic_arg = "Diabetes?"

    for mlp in Path("outputs/diabetes/").glob("mlp*"):
        print(f"Processing {i}/{num_loads}: {mlp.name}")

        aaes = np.load(f"{mlp}/aae_objects.npz", allow_pickle=True)["arr_0"]
        sps = np.load(f"{mlp}/sparx_objects.npz", allow_pickle=True)["arr_0"]
        qbafs = np.load(f"{mlp}/qbaf_objects.npz", allow_pickle=True)["arr_0"]

        print("Loaded data")

        # Parameters for max argument
        max_args = np.zeros((aaes.shape[0], aaes.shape[1] - 1))

        # Parameters for top n scores
        num_args = len(qbafs[0, 0].arguments) - 1
        # In the most sparse case, the number of arguments is 10% of the original
        num_args_sparse = int(num_args * 0.1)
        # Either compute stats for this or the top 10 scores
        num_to_check = min(num_args_sparse, 10)
        top_scores = np.empty_like(aaes, dtype=np.ndarray)

        # Parameters for averaged scores
        rankings = []

        # Parameters for removal impacts
        impacts = np.zeros((aaes.shape[0], aaes.shape[1] - 1))

        for example in range(len(aaes)):
            # Max argument processing
            full_aae = aaes[example][0]
            full_scores = get_scores(full_aae, score_type)
            max_score_arg_full = max(full_scores, key=lambda arg: abs(full_scores[arg]))

            if max_score_arg_full in input_names:
                max_score_full_layer = "input"
                max_score_full_neuron = max_score_arg_full
            else:
                max_score_full_split = max_score_arg_full.split(" ")
                max_score_full_layer = max_score_full_split[1]
                max_score_full_neuron = max_score_full_split[3]

            for sparse_percent in range(1, len(aaes[example])):
                scores = get_scores(aaes[example, sparse_percent], score_type)
                max_score_arg_sparse = max(scores, key=lambda arg: abs(scores[arg]))

                if max_score_arg_sparse in input_names:
                    if max_score_full_layer != "input":
                        max_args[example, sparse_percent - 1] = False
                    else:
                        max_args[example, sparse_percent - 1] = (
                            max_score_full_neuron == max_score_arg_sparse
                        )
                else:
                    max_score_sparse_split = max_score_arg_sparse.split(" ")
                    max_score_sparse_layer = max_score_sparse_split[1]
                    max_score_sparse_neuron = max_score_sparse_split[3]

                    if max_score_full_layer != max_score_sparse_layer:
                        max_args[example, sparse_percent - 1] = False
                    else:
                        sp = sps[example][sparse_percent - 1]
                        containing = sp.get_containing_neurons(
                            int(max_score_sparse_layer),
                            int(max_score_sparse_neuron),
                        )
                        max_args[example, sparse_percent - 1] = (
                            int(max_score_full_neuron) in containing
                        )

            # Top n scores processing
            for sparse_percent in range(len(qbafs[example])):
                # Get scores
                scores = get_scores(aaes[example, sparse_percent], score_type).values()
                # Sort them
                scores = np.array(list(scores))
                scores = np.abs(scores)
                scores = np.sort(scores)[::-1]
                # Get the top scores
                scores = scores[:num_to_check]

                top_scores[example, sparse_percent] = scores

            # Averaged scores processing
            original_aae = aaes[example, 0]
            original_aae_scores = get_scores(original_aae, score_type)
            scores = np.empty((len(sps[example]), 2), dtype=np.ndarray)
            for sparse_percent in range(1, len(aaes[example])):
                sp = sps[example, sparse_percent - 1]
                qbaf = qbafs[example, sparse_percent]
                sparse_aae = aaes[example, sparse_percent]
                sparse_aae_scores = get_scores(sparse_aae, score_type)
                averaged_original_scores = []
                for arg in qbaf.arguments:
                    if arg == topic_arg:
                        continue
                    if arg in input_names:
                        averaged_original_scores.append(
                            (arg, abs(original_aae_scores[arg])),
                        )
                    else:
                        split_name = arg.split(" ")
                        layer = int(split_name[1])
                        neuron = int(split_name[3])
                        clusters = sp.get_containing_neurons(layer, neuron)
                        average_score = 0
                        for n in clusters:
                            original_name = f"Layer {layer} Neuron {n}"
                            average_score += abs(original_aae_scores[original_name])
                        average_score /= len(clusters)
                        averaged_original_scores.append((arg, average_score))
                # Sort the scores and store them and the names of the clustered
                # arguments
                # The first contains the averaged original scores (averaging the scores
                # of the original AAE based on the clusters)
                # The second contains the scores of the sparse AAE
                scores[sparse_percent - 1, 0] = np.array(averaged_original_scores)
                sparse_items = list(sparse_aae_scores.items())
                sparse_items = [(name, abs(val)) for name, val in sparse_items]
                scores[sparse_percent - 1, 1] = np.array(sparse_items)
            rankings.append(scores)

            original_aae = aaes[example, 0]
            original_aae_scores = get_scores(original_aae, score_type)
            max_arg_original_name = max(
                original_aae_scores,
                key=lambda arg: abs(original_aae_scores[arg]),
            )
            original_qbaf = qbafs[example, 0]
            qbaf_original_remove_highest = original_qbaf.remove_arguments(
                [original_qbaf.arguments[max_arg_original_name]],
            )
            grad.algorithms.computeStrengthValues(
                qbaf_original_remove_highest,
                grad.SumAggregation(),
                grad.MLPBasedInfluence(),
            )

            output_strength_original_removed = qbaf_original_remove_highest.arguments[
                topic_arg
            ].strength

            for sparse_percent in range(1, len(qbafs[example])):
                qbaf = qbafs[example, sparse_percent]

                aae = aaes[example, sparse_percent]
                scores = get_scores(aae, score_type)
                highest_arg_name = max(scores, key=lambda arg: abs(scores[arg]))
                qbaf_remove_highest = qbaf.remove_arguments(
                    [qbaf.arguments[highest_arg_name]],
                )
                grad.algorithms.computeStrengthValues(
                    qbaf_remove_highest,
                    grad.SumAggregation(),
                    grad.MLPBasedInfluence(),
                )
                qbaf_removed_strength = qbaf_remove_highest.arguments[
                    topic_arg
                ].strength

                impacts[example, sparse_percent - 1] = abs(
                    output_strength_original_removed - qbaf_removed_strength,
                )

        del aaes
        del sps
        del qbafs

        max_argument_dict[mlp] = max_args.mean(axis=0) * 100
        top_n_scores_dict[mlp] = top_scores
        averaged_scores_dict[mlp] = rankings
        removal_impact_dict[mlp] = impacts

        i += 1

    return (
        max_argument_dict,
        top_n_scores_dict,
        averaged_scores_dict,
        removal_impact_dict,
    )


if __name__ == "__main__":
    ArgParser = argparse.ArgumentParser()
    ArgParser.add_argument(
        "--score_type",
        type=str,
        choices=["gradient", "removal", "shapley"],
        default="gradient",
    )
    score_type = ArgParser.parse_args().score_type
    if score_type == "gradient":
        score_type = ScoreType.GRADIENT
        save_name = "gradient_result_dicts"
    elif score_type == "removal":
        score_type = ScoreType.REMOVAL
        save_name = "removal_result_dicts"
    else:
        score_type = ScoreType.SHAPLEY
        save_name = "shapley_result_dicts"

    max_argument_dict, top_n_scores_dict, averaged_scores_dict, removal_impact_dict = (
        generate_dicts(
            score_type,
        )
    )
    path = "outputs/diabetes/"

    joblib.dump(max_argument_dict, f"{path}{save_name}_max_arguments.pkl", compress=3)
    joblib.dump(top_n_scores_dict, f"{path}{save_name}_top_n_scores.pkl", compress=3)
    joblib.dump(
        averaged_scores_dict,
        f"{path}{save_name}_averaged_scores.pkl",
        compress=3,
    )

    joblib.dump(
        removal_impact_dict,
        f"{path}{save_name}_removal_impact.pkl",
        compress=3,
    )
