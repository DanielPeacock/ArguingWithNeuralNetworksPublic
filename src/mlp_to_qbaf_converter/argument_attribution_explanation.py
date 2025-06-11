"""Class and methods to represent Argument Attribution Explanation."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
from tqdm import tqdm

import Uncertainpy.src.uncertainpy.gradual as grad

# Changes:


# - Create superclass Explanations
# - Move AAE class to a subclass of Explanations
# - Only allow one topic argument name \
#  i.e new class will need to be created to get explanations \
#  for different topic args
class AAE:
    """Represent Argument Attribution Explanations."""

    def __init__(  # noqa: PLR0913
        self,
        qbaf: grad.BAG,
        agg_func,  # noqa: ANN001
        inf_func,  # noqa: ANN001
        topic_arg_name: str,
        h: float,
        shap_samples: int = 1000,
        shap_seed: int | None = None,
        verbose: bool = True,  # noqa: FBT001, FBT002
        do_shap: bool = True,  # noqa: FBT001, FBT002
        do_removal: bool = True,  # noqa: FBT001, FBT002
        do_gradient: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initalise qbaf and generate explanation statistics.

        :param qbaf: The QBAF to generate explanations for (assumed acyclic).
        :param agg_func: The aggregation function to be used in the QBAF.
        :param inf_func: The influence function to be used in the QBAF.
        :param topic_arg_name: The name of the topic argument.
        :param h: The pertubation to be used when calculating the gradient.
        :param shap_samples: The number of samples to use when calculating SHAP scores.
        :param shap_seed: The seed to use when calculating SHAP scores.
        :param verbose: Whether to print progress bars.
        :param do_shap: Whether to calculate SHAP scores.
        :param do_removal: Whether to calculate removal scores.
        :param do_gradient: Whether to calculate gradient scores.
        """
        self.qbaf = qbaf
        self.agg_func = agg_func
        self.inf_func = inf_func
        self.topic_arg_name = topic_arg_name
        self.h = h
        self.shap_samples = shap_samples
        self.verbose = verbose

        if shap_seed is not None:
            random.seed(shap_seed)

        self.gradients = self.compute_topic_arg_gradients() if do_gradient else {}
        self.shap_scores = self.compute_shap_all() if do_shap else {}
        self.removal_scores = self.compute_removal_all() if do_removal else {}

    def compute_topic_arg_gradients(self) -> dict[str, float]:
        """Compute all the gradients for the topic arg.

        Returns:
            Dictionary of each argument name, with values the gradient score.

        """
        gradients = {}

        topic_arg = self.qbaf.arguments[self.topic_arg_name]
        arguments = list(self.qbaf.arguments.values())
        arguments.remove(topic_arg)

        for arg in tqdm(
            arguments,
            desc="Computing gradients",
            disable=not self.verbose,
        ):
            gradients[arg.name] = self.compute_args_gradient(arg, topic_arg)

        return gradients

    def compute_args_gradient(self, arg1: grad.Argument, arg2: grad.Argument) -> float:
        """Get the gradient between two arguments.

        :param arg1: Argument 1.
        :param arg2: Argument 2.

        Returns:
            The gradient between the arguments.

        """
        original_strength = arg2.strength
        initial_weight = arg1.get_initial_weight()

        arg1.reset_initial_weight(initial_weight + self.h)
        grad.algorithms.computeStrengthValues(self.qbaf, self.agg_func, self.inf_func)
        counterfactual_strength = arg2.strength
        gradient = (counterfactual_strength - original_strength) / self.h

        arg1.reset_initial_weight(initial_weight)
        grad.algorithms.computeStrengthValues(self.qbaf, self.agg_func, self.inf_func)

        return gradient

    def compute_shap_all(self) -> dict[str, float]:
        """Get the shap scores for the topic argument.

        Returns:
            A dictionary with keys of each argument. Values are the shap score
            for the corresponding argument.

        """
        shap_scores = {}
        topic_arg = self.qbaf.arguments[self.topic_arg_name]

        arguments = list(self.qbaf.arguments.values())
        arguments.remove(topic_arg)

        for arg in tqdm(
            arguments,
            desc="Computing SHAP scores",
            disable=not self.verbose,
        ):
            shap_scores[arg.name] = self.compute_arg_shap(topic_arg, arg)

        return shap_scores

    def compute_arg_shap(self, topic_arg: grad.Argument, arg: grad.Argument) -> float:
        """Get the attribution for an Argument arg and a topic argument.

        :param topic_arg: the topic argument.
        :param arg: the current argument to compute attribution against.

        Returns:
            The attribution score for this argument.

        """
        arguments = list(self.qbaf.arguments.values())
        arguments.remove(topic_arg)
        arguments.remove(arg)

        diff_sum = 0

        for _ in range(self.shap_samples):
            subset_length = random.randrange(0, len(arguments))  # noqa: S311
            subset = random.sample(arguments, subset_length)
            args_to_remove = set(arguments) - set(subset)

            subset_qbaf_with_arg = self.qbaf.remove_arguments(args_to_remove)

            grad.algorithms.computeStrengthValues(
                subset_qbaf_with_arg,
                self.agg_func,
                self.inf_func,
            )
            strength_before_removal = subset_qbaf_with_arg.arguments[
                topic_arg.name
            ].strength

            subset_qbaf_without_arg = subset_qbaf_with_arg.remove_arguments([arg])
            grad.algorithms.computeStrengthValues(
                subset_qbaf_without_arg,
                self.agg_func,
                self.inf_func,
            )
            strength_after_removal = subset_qbaf_without_arg.arguments[
                topic_arg.name
            ].strength

            diff = strength_before_removal - strength_after_removal
            diff_sum += diff

        return diff_sum / self.shap_samples

    def compute_removal_all(self) -> dict[str, float]:
        """Compute the change in strength if non-topic arguments are removed.

        Returns:
            Dictionary with keys as argument name and removal score as value.

        """
        removal_scores = {}

        topic_arg = self.qbaf.arguments[self.topic_arg_name]
        arguments = list(self.qbaf.arguments.values())
        arguments.remove(topic_arg)

        for arg in tqdm(
            arguments,
            desc="Computing removal scores",
            disable=not self.verbose,
        ):
            removal_scores[arg.name] = self.compute_removal_score_arg(arg)

        return removal_scores

    def compute_removal_score_arg(
        self,
        remove_arg: grad.Argument,
    ) -> float:
        """Compute the change in strength of topic argument when an argument is removed.

        :param topic_arg_name: The name of the topic argument to be checked.
        :param remove_arg: The argument to be removed.

        Returns:
            The difference in topic argument strength.

        """
        initial_strength = self.qbaf.arguments[self.topic_arg_name].strength
        removed_qbaf = self.qbaf.remove_arguments([remove_arg])

        grad.algorithms.computeStrengthValues(
            removed_qbaf,
            self.agg_func,
            self.inf_func,
        )

        removal_strength = removed_qbaf.arguments[self.topic_arg_name].strength

        return initial_strength - removal_strength

    def get_gradients(self) -> dict[str, float]:
        """Get the gradient explanations.

        Returns:
            Dictionary of each topic argument, each with a list of tuples
            of the other argument and gradient.

        """
        return self.gradients

    def get_shap_scores(self) -> dict[str, float]:
        """Get the shap score explanations.

        Returns:
            A dictionary with keys of each topic argument. Values are
            a list with a tuple of the other argument name and the score.

        """
        return self.shap_scores

    def get_removal_scores(self) -> dict[str, float]:
        """Get the removal scores.

        Returns:
            Dictionary with keys as topic argument names, each with a\
            list of tuples containing the argument removed and the change in\
            strength.

        """
        return self.removal_scores

    def create_gradient_plot(
        self,
        figsize: tuple[int, int] = (8, 4),
        n_highest: int | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a plot of the gradients.

        :param figsize: The size of the plot.
        :param n_highest: The number of highest gradients to plot.

        Returns:
            The plot axes.

        """
        fig, ax = plt.subplots(figsize=figsize)
        if not n_highest:
            n_highest = len(self.gradients)
        highest_gradients = dict(
            sorted(
                self.gradients.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:n_highest],
        )

        ax.barh(list(highest_gradients), list(highest_gradients.values()), height=0.5)
        ax.set_xlabel("Gradient")
        ax.set_ylabel("Argument")
        ax.set_title("AAE Gradient Scores")

        return fig, ax

    def create_shap_plot(
        self,
        figsize: tuple[int, int] = (8, 4),
        n_highest: int | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a plot of the SHAP scores.

        :param figsize: The size of the plot.
        :param n_highest: The number of highest SHAP scores to plot.

        Returns:
            The plot axes.

        """
        fig, ax = plt.subplots(figsize=figsize)
        if not n_highest:
            n_highest = len(self.shap_scores)
        highest_shap_scores = dict(
            sorted(
                self.shap_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:n_highest],
        )
        ax.barh(
            list(highest_shap_scores),
            list(highest_shap_scores.values()),
            height=0.5,
        )
        ax.set_xlabel("SHAP Score")
        ax.set_ylabel("Argument")
        ax.set_title("AAE SHAP Scores")

        return fig, ax

    def create_removal_plot(
        self,
        figsize: tuple[int, int] = (8, 4),
        n_highest: int | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a plot of the removal scores.

        :param figsize: The size of the plot.
        :param n_highest: The number of highest removal scores to plot.

        Returns:
            The plot axes.

        """
        fig, ax = plt.subplots(figsize=figsize)

        if not n_highest:
            n_highest = len(self.removal_scores)

        highest_removal_scores = dict(
            sorted(
                self.removal_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:n_highest],
        )

        ax.barh(
            list(highest_removal_scores),
            list(highest_removal_scores.values()),
            height=0.5,
        )
        ax.set_xlabel("Removal Score")
        ax.set_ylabel("Argument")
        ax.set_title("AAE Removal Scores")

        return fig, ax
