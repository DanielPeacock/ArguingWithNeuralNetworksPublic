# noqa: D100, INP001
from __future__ import annotations

import sys

sys.path.append("src/")

import sys
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import Uncertainpy.src.uncertainpy.gradual as grad
from mlp_to_qbaf_converter.argument_attribution_explanation import AAE
from mlp_to_qbaf_converter.mlp_to_qbaf import MLPToQBAF
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
    h: float,
    train_set: np.ndarray,
    X_test: np.ndarray,  # noqa: N803
    shrink_percent: int,
) -> tuple[grad.BAG, AAE] | tuple[grad.BAG, AAE, LocalSpArX]:
    """Run computations for a given shrink percentage.

    Args:
        neurons_per_layer: The number of neurons in each layer.
        mlp: The MLP model.
        input_feature_names: The input feature names.
        output_names: The output names.
        example: The example.
        example_row: The example row.
        topic_arg: The topic argument.
        h: The gradient pertubation value.
        train_set: The training set.
        X_test: The test set.
        shrink_percent: The shrink percentage.

    Returns:
        A tuple of QBAF, AAE and SpArX objects. If shrink_percent is 0, only QBAF\
            and AAE objects are returned.

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

        aae = AAE(
            qbaf,
            grad.SumAggregation(),
            grad.MLPBasedInfluence(),
            topic_arg,
            h,
            do_shap=True,
            do_removal=True,
            do_gradient=True,
            verbose=False,
            shap_samples=1000,
        )
        print(f"Computations for {shrink_percent}% sparsification done.", flush=True)
        return qbaf, aae

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

    aae = AAE(
        qbaf,
        grad.SumAggregation(),
        grad.MLPBasedInfluence(),
        topic_arg,
        h,
        do_shap=True,
        do_gradient=True,
        do_removal=True,
        verbose=False,
        shap_samples=1000,
    )

    print(f"Compuations for {shrink_percent}% sparsification done.", flush=True)

    return qbaf, aae, sp


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
    2. Create an AAE for the full model.
    3. Create a QBAF for the sparsified model.
    4. Create an AAE for the sparsified model.
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
        Arrays of QBAFs created, AAEs created and SpArX objects created.

    """
    mlp = joblib.load(model_file)
    # For AAEs, avoid log errors
    h = 1e-12
    X_test = np.clip(X_test, h, 1 - h)  # noqa: N806

    neurons_per_layer = [
        mlp.n_features_in_,
        *list(mlp.hidden_layer_sizes),
        mlp.n_outputs_,
    ]

    train_set = np.column_stack((X_train, y_train))
    shrink_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    sparx_objects = np.empty(
        (len(X_test), len(shrink_percentages) - 1),
        dtype=object,
    )
    aae_objects = np.empty(
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

        results = Parallel(n_jobs=len(shrink_percentages))(
            delayed(run_computations)(
                neurons_per_layer,
                mlp,
                input_feature_names,
                output_names,
                example,
                example_row,
                topic_arg,
                h,
                train_set,
                X_test,
                shrink_percent,
            )
            for shrink_percent in shrink_percentages
        )

        for i, (shrink_percent, result) in enumerate(zip(shrink_percentages, results)):
            if shrink_percent == 0:
                qbaf_objects[example_row_num, i], aae_objects[example_row_num, i] = (
                    result
                )
            else:
                (
                    qbaf_objects[example_row_num, i],
                    aae_objects[example_row_num, i],
                    sparx_objects[example_row_num, i - 1],
                ) = result

        if example_row_num % 10 == 0:
            print(
                f"{round(example_row_num / len(X_test) * 100, 3)}% for \
hidden layer sizes: {hidden_layer_sizes} done.",
                flush=True,
            )

    print(f"Computation for hidden layer sizes: {hidden_layer_sizes} done.", flush=True)
    return qbaf_objects, aae_objects, sparx_objects


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
            return aee.get_shap_scores()
        case _:
            msg = "Invalid score type"
            raise ValueError(msg)


def generate_dicts(  # noqa: C901, PLR0912, PLR0913, PLR0915
    score_type: ScoreType,
    aaes: np.ndarray,
    sps: np.ndarray,
    qbafs: np.ndarray,
    save_path: str,
    input_names: list[str],
    output_names: list[str],
    topic_arg: str,
) -> None:
    """Generate dicts: max argument, top n scores and averaged scores and impact diff.

    :param score_type: The type of score to get
    :param aaes: The AAE objects
    :param sps: The SpArX objects
    :param qbafs: The QBAF objects
    :param save_path: The path to save the results
    :param input_names: The names of the input arguments
    :param output_names: The names of the output arguments
    :param topic_arg: The name of the topic argument

    """
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
                if arg in input_names or arg in output_names:
                    averaged_original_scores.append(
                        (arg, original_aae_scores[arg]),
                    )
                else:
                    split_name = arg.split(" ")
                    layer = int(split_name[1])
                    neuron = int(split_name[3])
                    clusters = sp.get_containing_neurons(layer, neuron)
                    average_score = 0
                    for n in clusters:
                        original_name = f"Layer {layer} Neuron {n}"
                        average_score += original_aae_scores[original_name]
                    average_score /= len(clusters)
                    averaged_original_scores.append((arg, average_score))
            # Sort the scores and store them and the names of the clustered
            # arguments
            # The first contains the averaged original scores (averaging the scores
            # of the original AAE based on the clusters)
            # The second contains the scores of the sparse AAE
            scores[sparse_percent - 1, 0] = np.array(averaged_original_scores)
            sparse_items = list(sparse_aae_scores.items())
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
            qbaf_removed_strength = qbaf_remove_highest.arguments[topic_arg].strength

            impacts[example, sparse_percent - 1] = abs(
                output_strength_original_removed - qbaf_removed_strength,
            )

    max_args = max_args.mean(axis=0) * 100

    np.savez_compressed(
        save_path,
        max_arguments=max_args,
        top_scores=top_scores,
        rankings=rankings,
        impacts=impacts,
    )
