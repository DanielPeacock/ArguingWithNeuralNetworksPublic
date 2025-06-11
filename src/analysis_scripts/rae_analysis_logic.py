"""RAE analysis logic."""  # noqa: INP001

from __future__ import annotations

import sys

sys.path.append("src/")

from enum import Enum
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import kendalltau, pearsonr, wasserstein_distance
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import Uncertainpy.src.uncertainpy.gradual as grad
from mlp_to_qbaf_converter.mlp_to_qbaf import MLPToQBAF
from mlp_to_qbaf_converter.relation_attribution_explanation import RAE
from sparx.sparx import LocalSpArX


def make_mlp(  # noqa: PLR0913
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    X_test: np.ndarray,  # noqa: N803
    y_test: np.ndarray,
    hidden_layer_sizes: tuple,
    model_file: str,
    accuracy_file: str,
    seed: int = 2025,
    min_accuracy: float = 0.7,
) -> None:
    """Train a MLP for a hidden layer size.

    Train a MLP with the given hidden layer sizes and save the model.
    Save accuracies in a text file and print the classification report.
    If the model already exists, no training is done. Loads the model
    and prints the classification report.

    Args:
        X_train: The training data.
        y_train: The training labels.
        X_test: The test data.
        y_test: The test labels.
        hidden_layer_sizes: The hidden layer sizes of the MLP.
        model_file: The file to save the model.
        accuracy_file: The file to save the accuracies.
        seed: The random seed
        min_accuracy: The minimum accuracy to accept the model.

    """
    if not Path(model_file).exists():
        print("Training model...")
        # Initialize MLPClassifier with balanced classes
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        solvers = ["adam", "sgd", "lbfgs"]

        best_mlp = None
        best_score = 0
        params = {"alpha": alphas, "solver": solvers}
        grid_search = GridSearchCV(
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="logistic",
                random_state=seed,
                max_iter=5000,
                early_stopping=True,
            ),
            params,
            cv=3,
            n_jobs=-1,
            verbose=2,
        )

        grid_search.fit(X_train, y_train)
        best_mlp = grid_search.best_estimator_
        best_score = grid_search.best_score_

        if best_score < min_accuracy:
            msg = "Model accuracy is too low"
            raise ValueError(msg)

        # Save model
        joblib.dump(best_mlp, model_file)
        mlp = best_mlp

    else:
        mlp = joblib.load(model_file)

    with Path(accuracy_file).open("w") as acc:
        acc.write(f"Train accuracy: {mlp.score(X_train, y_train)}\n")
        acc.write(f"Test accuracy: {mlp.score(X_test, y_test)}\n")
        acc.write(classification_report(y_test, mlp.predict(X_test)))


def run_computations(  # noqa: PLR0913
    neurons_per_layer: list[int],
    mlp: MLPClassifier,
    input_feature_names: list[str],
    output_names: list[str],
    example: np.ndarray,
    example_row: np.ndarray,
    topic_arg: str,
    train_set: np.ndarray,
    X_test: np.ndarray,  # noqa: N803
    shrink_percent: int,
) -> tuple[grad.BAG, RAE] | tuple[grad.BAG, RAE, LocalSpArX]:
    """Run computations for a given shrink percentage.

    Args:
        neurons_per_layer: The number of neurons in each layer.
        mlp: The MLP model.
        input_feature_names: The input feature names.
        output_names: The output names.
        example: The example.
        example_row: The example row.
        topic_arg: The topic argument.
        train_set: The training set.
        X_test: The test set.
        shrink_percent: The shrink percentage.

    Returns:
        A tuple of QBAF, RAE and SpArX objects. If shrink_percent is 0, only QBAF\
            and RAE objects are returned.

    """
    if shrink_percent == 0:
        qbaf = MLPToQBAF(
            neurons_per_layer,
            mlp.coefs_,
            mlp.intercepts_,
            "logistic",
            input_feature_names,
            output_names,
            example,
        ).get_qbaf()

        rae = RAE(
            qbaf,
            grad.SumAggregation(),
            grad.MLPBasedInfluence(),
            topic_arg,
            shap_seed=2025,
            shap_samples=1000,
            verbose=False,
            do_shap=True,
            do_removal=True,
        )

        print(f"Computations for {shrink_percent}% sparsification done.", flush=True)
        return qbaf, rae

    sp = LocalSpArX(
        mlp.coefs_,
        mlp.intercepts_,
        "logistic",
        shrink_percent,
        example_row,
        train_set,
        np.sqrt(X_test.shape[1]) * 0.75,
    )
    sp_weights, sp_biases = sp.get_sparsified_mlp()

    qbaf = MLPToQBAF(
        sp.get_sparsified_shape(),
        sp_weights,
        sp_biases,
        "logistic",
        input_feature_names,
        output_names,
        example,
    ).get_qbaf()

    rae = RAE(
        qbaf,
        grad.SumAggregation(),
        grad.MLPBasedInfluence(),
        topic_arg,
        shap_seed=2025,
        shap_samples=1000,
        verbose=False,
        do_shap=True,
        do_removal=True,
    )

    print(f"Computations for {shrink_percent}% sparsification done.", flush=True)

    return qbaf, rae, sp


def run_analysis(  # noqa: PLR0913
    hidden_layer_sizes: tuple,
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    X_test: np.ndarray,  # noqa: N803
    y_test: np.ndarray,
    output_names: list[str],
    input_feature_names: list[str],
    topic_arg: str,
    model_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run analysis for a given hidden layer size.

    Go through all examples in the test set and for each example:
    1. Create a QBAF for the full model.
    2. Create an RAE for the full model.
    3. Create a QBAF for the sparsified model.
    4. Create an RAE for the sparsified model.
    7. Repeat for all shrink percentages.

    Args:
        hidden_layer_sizes: The hidden layer sizes of the MLP.
        X_train: The training data.
        y_train: The training labels.
        X_test: The test data.
        y_test: The test labels.
        output_names: The output names.
        input_feature_names: The input feature names.
        topic_arg: The topic argument.
        model_file: The file to save the model.

    Returns:
        Arrays of QBAFs created, RAEs created and SpArX objects created.

    """
    mlp = joblib.load(model_file)

    X_test = np.clip(X_test, 0, 1)  # noqa: N806

    neurons_per_layer = [
        mlp.n_features_in_,
        *list(mlp.hidden_layer_sizes),
        mlp.n_outputs_,
    ]

    train_set = np.column_stack((X_train, y_train))
    shrink_percentages = [0, 20, 40, 60, 80]

    sparx_objects = np.empty(
        (len(X_test), len(shrink_percentages) - 1),
        dtype=object,
    )
    rae_objects = np.empty(
        (len(X_test), len(shrink_percentages)),
        dtype=object,
    )
    qbaf_objects = np.empty(
        (len(X_test), len(shrink_percentages)),
        dtype=object,
    )

    for example_row_num in range(len(X_test)):
        print(f"Example {example_row_num + 1} / {len(X_test)}")
        example = X_test[example_row_num]
        example_row = np.append(example, y_test[example_row_num])

        results = joblib.Parallel(n_jobs=len(shrink_percentages))(
            joblib.delayed(run_computations)(
                neurons_per_layer,
                mlp,
                input_feature_names,
                output_names,
                example,
                example_row,
                topic_arg,
                train_set,
                X_test,
                shrink_percent,
            )
            for shrink_percent in shrink_percentages
        )

        for i, (shrink_percent, result) in enumerate(zip(shrink_percentages, results)):
            if shrink_percent == 0:
                qbaf_objects[example_row_num, i], rae_objects[example_row_num, i] = (
                    result
                )
            else:
                (
                    qbaf_objects[example_row_num, i],
                    rae_objects[example_row_num, i],
                    sparx_objects[example_row_num, i - 1],
                ) = result

        if example_row_num % 10 == 0:
            print(
                f"{round(example_row_num / len(X_test) * 100, 3)}% for \
hidden layer sizes: {hidden_layer_sizes} done.",
                flush=True,
            )

    print(f"Computation for hidden layer sizes: {hidden_layer_sizes} done.", flush=True)
    return qbaf_objects, rae_objects, sparx_objects


class ScoreType(Enum):
    """The type of score to get from an RAE."""

    SHAPLEY = 1
    REMOVAL = 2


def get_scores(rae: RAE, score_type: ScoreType) -> dict[str, float]:
    """Get the scores from an RAE object based on the score type.

    :param aee: The RAE object
    :param score_type: The type of score to get

    Returns:
        The scores.

    """
    match score_type:
        case ScoreType.REMOVAL:
            return rae.get_removal_scores()
        case ScoreType.SHAPLEY:
            return rae.get_shap_scores()
        case _:
            msg = "Invalid score type"
            raise ValueError(msg)


def generate_dicts(  # noqa: C901, PLR0912, PLR0913, PLR0915
    score_type: ScoreType,
    raes: np.ndarray[RAE],
    sps: np.ndarray[LocalSpArX],
    qbafs: np.ndarray[grad.BAG],
    save_path: str,
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Generate averaged scores and rankings for each example.

    The scores are averaged over matching attacks/ supporters in the original.

    :param score_type: The type of score to get from an RAE.
    :param raes: The RAE objects.
    :param sps: The SpArX objects.
    :param qbafs: The QBAF objects.
    :param save_path: The path to save the rankings.
    :param input_names: The input names.
    :param output_names: The output names.
    """
    rankings = []
    rankings_variances = []
    rankings_weighted = []
    non_averaged_rankings = []
    wasserstein_distances = []
    kendall_taus = []
    pearson_correlations = []

    for example in range(len(raes)):
        original_rae = raes[example, 0]
        original_qbaf = qbafs[example, 0]
        original_scores = get_scores(original_rae, score_type)
        scores = np.empty((len(sps[example]), 2), dtype=np.ndarray)
        scores_variances = np.empty((len(sps[example]), 1), dtype=np.ndarray)
        scores_weighted = np.empty((len(sps[example]), 2), dtype=np.ndarray)
        wasserstein_tmp = np.empty((len(sps[example]), 1), dtype=float)
        kendall_taus_tmp = np.empty((len(sps[example]), 1), dtype=float)
        pearson_correlations_tmp = np.empty((len(sps[example]), 1), dtype=float)

        for sparse_percent in range(1, len(raes[example])):
            sp = sps[example, sparse_percent - 1]
            sparse_qbaf = qbafs[example, sparse_percent]
            sparse_rae = raes[example, sparse_percent]
            sparse_scores = get_scores(sparse_rae, score_type)
            averaged_scores = {}

            for relation in sparse_scores:
                # Tuple of summed score and number of relations averaged
                if "Attack" in relation:
                    r = relation.replace("Attack(", "")
                else:
                    r = relation.replace("Support(", "")

                r = r.replace(")", "")
                averaged_scores[r] = []

            for arg in sparse_qbaf.arguments:
                if arg in input_names:
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
                    if layer == "output":
                        original_name = n
                    else:
                        original_name = f"Layer {layer} Neuron {n}"

                    original_arg = original_qbaf.arguments[original_name]
                    original_attackers = original_arg.attackers
                    original_supporters = original_arg.supporters

                    for attacker in original_attackers:
                        attacker_name = attacker.name
                        if attacker_name in input_names:
                            sparse_cluster_name = attacker_name
                        else:
                            attacker_name_split = attacker_name.split(" ")
                            attacker_layer = int(attacker_name_split[1])
                            attacker_neuron = int(attacker_name_split[3])
                            sparse_cluster = (
                                sp.cluster_labels[attacker_layer - 1][
                                    attacker_neuron - 1
                                ]
                                + 1
                            )
                            sparse_cluster_name = (
                                f"Layer {attacker_layer} Neuron {sparse_cluster}"
                            )

                        original_attack_string = (
                            f"Attack({attacker_name} to {original_name})"
                        )
                        original_score = original_scores[original_attack_string]
                        sparse_attack_string = f"{sparse_cluster_name} to {arg}"
                        averaged_scores[sparse_attack_string].append(original_score)

                    for supporter in original_supporters:
                        supporter_name = supporter.name
                        if supporter_name in input_names:
                            sparse_cluster_name = supporter_name
                        else:
                            supporter_name_split = supporter_name.split(" ")
                            supporter_layer = int(supporter_name_split[1])
                            supporter_neuron = int(supporter_name_split[3])
                            sparse_cluster = (
                                sp.cluster_labels[supporter_layer - 1][
                                    supporter_neuron - 1
                                ]
                                + 1
                            )
                            sparse_cluster_name = (
                                f"Layer {supporter_layer} Neuron {sparse_cluster}"
                            )

                        original_support_string = (
                            f"Support({supporter_name} to {original_name})"
                        )
                        original_score = original_scores[original_support_string]
                        sparse_support_string = f"{sparse_cluster_name} to {arg}"
                        averaged_scores[sparse_support_string].append(original_score)

            averaged_scores_variance = {}

            for relation, scores_to_average in averaged_scores.items():
                averaged_scores[relation] = np.mean(scores_to_average)
                averaged_scores_variance[relation] = np.var(scores_to_average)

            averaged_scores = list(averaged_scores.items())
            scores[sparse_percent - 1, 0] = np.array(averaged_scores)

            scores_variances[sparse_percent - 1, 0] = np.array(
                list(averaged_scores_variance.items()),
            )

            sparse_items = list(sparse_scores.items())
            scores[sparse_percent - 1, 1] = np.array(sparse_items)

        for sparse_percent in range(1, len(raes[example])):
            sp = sps[example, sparse_percent - 1]
            sparse_qbaf = qbafs[example, sparse_percent]
            sparse_rae = raes[example, sparse_percent]
            sparse_scores = get_scores(sparse_rae, score_type)
            averaged_scores_weighted = {}

            for relation in sparse_scores:
                # Tuple of summed score and number of relations averaged
                if "Attack" in relation:
                    r = relation.replace("Attack(", "")
                else:
                    r = relation.replace("Support(", "")

                r = r.replace(")", "")
                averaged_scores_weighted[r] = (0, 0)

            for arg in sparse_qbaf.arguments:
                if arg in input_names:
                    continue

                if arg in output_names:
                    clusters = [arg]
                    layer = "output"
                    neuron = arg
                    arg_cluster_size = 1
                else:
                    split_name = arg.split(" ")
                    layer = int(split_name[1])
                    neuron = int(split_name[3])

                    clusters = sp.get_containing_neurons(layer, neuron)
                    arg_cluster_size = len(clusters)

                for n in clusters:
                    if layer == "output":
                        original_name = n
                    else:
                        original_name = f"Layer {layer} Neuron {n}"

                    original_arg = original_qbaf.arguments[original_name]
                    original_attackers = original_arg.attackers
                    original_supporters = original_arg.supporters

                    for attacker in original_attackers:
                        attacker_name = attacker.name
                        if attacker_name in input_names:
                            sparse_cluster_name = attacker_name
                        else:
                            attacker_name_split = attacker_name.split(" ")
                            attacker_layer = int(attacker_name_split[1])
                            attacker_neuron = int(attacker_name_split[3])
                            sparse_cluster = (
                                sp.cluster_labels[attacker_layer - 1][
                                    attacker_neuron - 1
                                ]
                                + 1
                            )
                            sparse_cluster_name = (
                                f"Layer {attacker_layer} Neuron {sparse_cluster}"
                            )

                        original_attack_string = (
                            f"Attack({attacker_name} to {original_name})"
                        )
                        original_score = original_scores[original_attack_string]
                        sparse_attack_string = f"{sparse_cluster_name} to {arg}"
                        averaged_scores_weighted[sparse_attack_string] = (
                            averaged_scores_weighted[sparse_attack_string][0]
                            + (
                                original_qbaf.arguments[original_name].strength
                                * original_score
                            ),
                            averaged_scores_weighted[sparse_attack_string][1]
                            + (
                                arg_cluster_size
                                * sparse_qbaf.arguments[sparse_cluster_name].strength
                            ),
                        )

                    for supporter in original_supporters:
                        supporter_name = supporter.name
                        if supporter_name in input_names:
                            sparse_cluster_name = supporter_name
                        else:
                            supporter_name_split = supporter_name.split(" ")
                            supporter_layer = int(supporter_name_split[1])
                            supporter_neuron = int(supporter_name_split[3])
                            sparse_cluster = (
                                sp.cluster_labels[supporter_layer - 1][
                                    supporter_neuron - 1
                                ]
                                + 1
                            )
                            sparse_cluster_name = (
                                f"Layer {supporter_layer} Neuron {sparse_cluster}"
                            )

                        original_support_string = (
                            f"Support({supporter_name} to {original_name})"
                        )
                        original_score = original_scores[original_support_string]
                        sparse_support_string = f"{sparse_cluster_name} to {arg}"
                        averaged_scores_weighted[sparse_support_string] = (
                            averaged_scores_weighted[sparse_support_string][0]
                            + (
                                original_qbaf.arguments[original_name].strength
                                * original_score
                            ),
                            averaged_scores_weighted[sparse_support_string][1]
                            + (
                                arg_cluster_size
                                * sparse_qbaf.arguments[sparse_cluster_name].strength
                            ),
                        )
            for relation in averaged_scores_weighted:
                averaged_scores_weighted[relation] = (
                    # Avoid division by zero by adding a small number
                    averaged_scores_weighted[relation][0]
                    / (1e-8 + averaged_scores_weighted[relation][1])
                )

            averaged_scores_weighted = list(averaged_scores_weighted.items())
            scores_weighted[sparse_percent - 1, 0] = np.array(averaged_scores_weighted)

            sparse_items = list(sparse_scores.items())
            scores_weighted[sparse_percent - 1, 1] = np.array(sparse_items)

        # Processing for getting top merged weights scores (expanding) + comparing
        original_scores_converted = {}

        for relation in original_scores:
            if "Attack" in relation:
                r = relation.replace("Attack(", "")
            else:
                r = relation.replace("Support(", "")

            r = r.replace(")", "")
            original_scores_converted[r] = original_scores[relation]

        scores_sparse_to_original = np.zeros(len(sps[example]))

        for sparse_percent in range(1, len(raes[example])):
            sp = sps[example, sparse_percent - 1]
            sparse_qbaf = qbafs[example, sparse_percent]
            sparse_rae = raes[example, sparse_percent]
            sparse_scores = list(get_scores(sparse_rae, score_type).items())
            sparse_scores = sorted(sparse_scores, key=lambda x: abs(x[1]), reverse=True)
            averaged_scores = {}

            relation_max = sparse_scores[0][0]

            if "Attack" in relation_max:
                r = relation_max.replace("Attack(", "")
            else:
                r = relation_max.replace("Support(", "")

            r = r.replace(")", "")

            start_cluster = r.split(" to ")[0]

            if start_cluster in input_names or start_cluster in output_names:
                containing_args_start = [start_cluster]
            else:
                layer_start = int(start_cluster.split(" ")[1])
                neuron_start = int(start_cluster.split(" ")[3])
                containing_args_start = sp.get_containing_neurons(
                    layer_start,
                    neuron_start,
                )
                containing_args_start = [
                    f"Layer {layer_start} Neuron {n}" for n in containing_args_start
                ]

            end_cluster = r.split(" to ")[1]

            if end_cluster in input_names or end_cluster in output_names:
                containing_args_end = [end_cluster]
            else:
                layer_end = int(end_cluster.split(" ")[1])
                neuron_end = int(end_cluster.split(" ")[3])
                containing_args_end = sp.get_containing_neurons(
                    layer_end,
                    neuron_end,
                )
                containing_args_end = [
                    f"Layer {layer_end} Neuron {n}" for n in containing_args_end
                ]

            top_n = len(containing_args_start) * len(containing_args_end)

            original_relations = sorted(
                original_scores_converted.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_n]
            original_relations = [r[0] for r in original_relations]

            for start_arg in containing_args_start:
                for end_arg in containing_args_end:
                    r_contained = f"{start_arg} to {end_arg}"
                    if r_contained in original_relations:
                        scores_sparse_to_original[sparse_percent - 1] += 1

            scores_sparse_to_original[sparse_percent - 1] /= top_n
            scores_sparse_to_original[sparse_percent - 1] *= 100

        for sparse_percent in range(1, len(raes[example])):
            original_rae = raes[example, 0]
            sparse_rae = raes[example, sparse_percent]
            scores_original = list(get_scores(original_rae, score_type).values())
            scores_sparse = list(get_scores(sparse_rae, score_type).values())
            wasserstein_tmp[sparse_percent - 1] = wasserstein_distance(
                scores_original,
                scores_sparse,
            )

        for sparse_percent in range(1, len(raes[example])):
            original_rae = raes[example, 0]
            sparse_rae = raes[example, sparse_percent]
            sparse_qbaf = qbafs[example, sparse_percent]
            sp = sps[example, sparse_percent - 1]
            count_no_merged = {}

            scores_original = get_scores(original_rae, score_type)
            scores_original_without_types = {}
            for relation in scores_original:
                if "Attack" in relation:
                    r = relation.replace("Attack(", "")
                else:
                    r = relation.replace("Support(", "")

                r = r.replace(")", "")
                scores_original_without_types[r] = scores_original[relation]
            scores_sparse = get_scores(sparse_rae, score_type)
            scores_sparse_without_types = {}
            for relation in scores_sparse:
                if "Attack" in relation:
                    r = relation.replace("Attack(", "")
                else:
                    r = relation.replace("Support(", "")

                r = r.replace(")", "")
                scores_sparse_without_types[r] = scores_sparse[relation]
                count_no_merged[r] = 0

            for sparse_arg in sparse_qbaf.arguments:
                if sparse_arg in input_names:
                    continue

                if sparse_arg in output_names:
                    clusters = [sparse_arg]
                    layer = "output"
                    neuron = sparse_arg
                else:
                    split_name = sparse_arg.split(" ")
                    layer = int(split_name[1])
                    neuron = int(split_name[3])

                    clusters = sp.get_containing_neurons(layer, neuron)

                for n in clusters:
                    if layer == "output":
                        original_name = n
                    else:
                        original_name = f"Layer {layer} Neuron {n}"

                    original_arg = original_qbaf.arguments[original_name]
                    original_attackers = original_arg.attackers
                    original_supporters = original_arg.supporters

                    for attacker in original_attackers:
                        attacker_name = attacker.name
                        if attacker_name in input_names:
                            sparse_cluster_name = attacker_name
                        else:
                            attacker_name_split = attacker_name.split(" ")
                            attacker_layer = int(attacker_name_split[1])
                            attacker_neuron = int(attacker_name_split[3])
                            sparse_cluster = (
                                sp.cluster_labels[attacker_layer - 1][
                                    attacker_neuron - 1
                                ]
                                + 1
                            )
                            sparse_cluster_name = (
                                f"Layer {attacker_layer} Neuron {sparse_cluster}"
                            )

                        sparse_string = f"{sparse_cluster_name} to {sparse_arg}"

                        count_no_merged[sparse_string] += 1

                    for supporter in original_supporters:
                        supporter_name = supporter.name
                        if supporter_name in input_names:
                            sparse_cluster_name = supporter_name
                        else:
                            supporter_name_split = supporter_name.split(" ")
                            supporter_layer = int(supporter_name_split[1])
                            supporter_neuron = int(supporter_name_split[3])
                            sparse_cluster = (
                                sp.cluster_labels[supporter_layer - 1][
                                    supporter_neuron - 1
                                ]
                                + 1
                            )
                            sparse_cluster_name = (
                                f"Layer {supporter_layer} Neuron {sparse_cluster}"
                            )

                        sparse_string = f"{sparse_cluster_name} to {sparse_arg}"

                        count_no_merged[sparse_string] += 1

            original_scores = list(scores_original_without_types.values())
            sparse_scores = []

            for relation, number_to_add in count_no_merged.items():
                sparse_scores += [
                    scores_sparse_without_types[relation],
                ] * number_to_add

            sparse_scores = np.array(sparse_scores)
            original_scores = np.array(original_scores)

            sparse_scores = np.sort(sparse_scores)
            original_scores = np.sort(original_scores)

            kendall_taus_tmp[sparse_percent - 1] = kendalltau(
                sparse_scores,
                original_scores,
            )[0]
            pearson_correlations_tmp[sparse_percent - 1] = pearsonr(
                sparse_scores,
                original_scores,
            )[0]

        rankings.append(scores)
        rankings_variances.append(scores_variances)
        rankings_weighted.append(scores_weighted)
        non_averaged_rankings.append(scores_sparse_to_original)
        wasserstein_distances.append(wasserstein_tmp)
        kendall_taus.append(kendall_taus_tmp)
        pearson_correlations.append(pearson_correlations_tmp)

    np.savez_compressed(
        save_path,
        rankings=rankings,
        rankings_variances=rankings_variances,
        rankings_weighted=rankings_weighted,
        non_averaged_rankings=non_averaged_rankings,
        wasserstein_distances=wasserstein_distances,
        kendall_taus=kendall_taus,
        pearson_correlations=pearson_correlations,
    )
