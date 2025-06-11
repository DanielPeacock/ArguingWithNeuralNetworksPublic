"""Class and methods to represent Relation Attribution Explanation."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
from tqdm import tqdm

import Uncertainpy.src.uncertainpy.gradual as grad


class RAE:
    """Represent Relation Attribution Explanations."""

    def __init__(  # noqa: PLR0913
        self,
        qbaf: grad.BAG,
        agg_func,  # noqa: ANN001
        inf_func,  # noqa: ANN001
        topic_arg_name: str,
        shap_samples: int = 1000,
        shap_seed: int | None = None,
        do_shap: bool = True,  # noqa: FBT001, FBT002
        do_removal: bool = True,  # noqa: FBT001, FBT002
        verbose: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialise qbaf and generate relation explanation statistics.

        :param qbaf: The QBAF to generate explanations for (assumed acyclic).
        :param agg_func: The aggregation function to be used in the QBAF.
        :param inf_func: The influence function to be used in the QBAF.
        :param topic_arg_name: The name of the topic argument.
        :param shap_samples: The number of samples to use when calculating SHAP scores.
        :param shap_seed: The seed to use when calculating SHAP scores.
        :param do_shap: Whether to compute SHAP scores.
        :param do_removal: Whether to compute removal scores.
        :param verbose: Whether to print progress bars.
        """
        self.qbaf = qbaf
        self.relations = self.qbaf.attacks + self.qbaf.supports
        self.agg_func = agg_func
        self.inf_func = inf_func
        self.topic_arg_name = topic_arg_name
        self.shap_samples = shap_samples
        self.verbose = verbose

        if shap_seed is not None:
            random.seed(shap_seed)

        if do_shap:
            self.shap_scores = self.compute_shap_scores()
        else:
            self.shap_scores = {}

        if do_removal:
            self.removal_scores = self.compute_removal_scores()
        else:
            self.removal_scores = {}

    def compute_shap_scores(self) -> dict[str, float]:
        """Compute SHAP score for a given relation argument.

        Args:
            topic_arg: The topic argument to compute the SHAP score for.

        Returns:
            Dictionary of each relation and its SHAP score.

        """
        scores = {}

        for relation in tqdm(
            self.relations,
            desc="Computing SHAP scores",
            disable=not self.verbose,
        ):
            other_relations = self.relations.copy()
            other_relations.remove(relation)

            diff_sum = 0

            for _ in range(self.shap_samples):
                qbaf_new = self.qbaf
                subset_length = random.randrange(0, len(other_relations) - 1)  # noqa: S311
                to_remove = random.sample(other_relations, subset_length)

                for r in to_remove:
                    if isinstance(r, grad.Attack):
                        qbaf_new = qbaf_new.remove_attack(r)
                    else:
                        qbaf_new = qbaf_new.remove_support(r)

                grad.algorithms.computeStrengthValues(
                    qbaf_new,
                    self.agg_func,
                    self.inf_func,
                )
                strength_before_drop = qbaf_new.arguments[self.topic_arg_name].strength

                if isinstance(relation, grad.Attack):
                    qbaf_new = qbaf_new.remove_attack(relation)
                else:
                    qbaf_new = qbaf_new.remove_support(relation)

                grad.algorithms.computeStrengthValues(
                    qbaf_new,
                    self.agg_func,
                    self.inf_func,
                )
                strength_after_drop = qbaf_new.arguments[self.topic_arg_name].strength

                diff = strength_before_drop - strength_after_drop
                diff_sum += diff

            # Create an easier to read string for the relation

            if isinstance(relation, grad.Attack):
                relation_string = (
                    f"Attack({relation.attacker.name} to {relation.attacked.name})"
                )
            else:
                relation_string = (
                    f"Support({relation.supporter.name} to {relation.supported.name})"
                )

            scores[relation_string] = diff_sum / self.shap_samples

        return scores

    def compute_removal_scores(self) -> dict[str, float]:
        """Compute removal scores for a given topic argument.

        Returns:
            A dictionary of arguments and their removal scores.

        """
        grad.algorithms.computeStrengthValues(self.qbaf, self.agg_func, self.inf_func)
        strength_before_removal = self.qbaf.arguments[self.topic_arg_name].strength

        scores = {}

        for relation in tqdm(
            self.relations,
            desc="Computing removal scores",
            disable=not self.verbose,
        ):
            if isinstance(relation, grad.Attack):
                qbaf_new = self.qbaf.remove_attack(relation)
                relation_string = (
                    f"Attack({relation.attacker.name} to {relation.attacked.name})"
                )
            else:
                qbaf_new = self.qbaf.remove_support(relation)
                relation_string = (
                    f"Support({relation.supporter.name} to {relation.supported.name})"
                )

            grad.algorithms.computeStrengthValues(
                qbaf_new,
                self.agg_func,
                self.inf_func,
            )
            strength_after_removal = qbaf_new.arguments[self.topic_arg_name].strength

            scores[relation_string] = strength_before_removal - strength_after_removal

        return scores

    def get_shap_scores(self) -> dict[str, float]:
        """Get the SHAP scores for the relation arguments."""
        return self.shap_scores

    def get_removal_scores(self) -> dict[str, float]:
        """Get the removal scores for the relation arguments."""
        return self.removal_scores

    def create_shap_plot(
        self,
        figsize: tuple[int, int] = (8, 4),
        n_highest: int | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a plot of the SHAP scores.

        :param figsize: The size of the plot.
        :param n_highest: The number of highest SHAP scores to display.

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
        ax.set_ylabel("Relation")
        ax.set_title("RAE SHAP Scores")

        return fig, ax

    def create_removal_plot(
        self,
        figsize: tuple[int, int] = (8, 4),
        n_highest: int | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a plot of the removal scores.

        :param figsize: The size of the plot.
        :param n_highest: The number of highest removal scores to display.

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
        ax.set_title("RAE Removal Scores")

        return fig, ax
