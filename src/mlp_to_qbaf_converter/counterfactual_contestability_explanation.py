"""Class to compute edge-weighted counterfactual explanations for a given QBAF."""

from __future__ import annotations

import copy
import random

import Uncertainpy.src.uncertainpy.gradual as grad
from mlp_to_qbaf_converter.errors import CounterfactualExplanationNotFoundError


class CounterfactualContestabilityExplanation:
    """Compute edge-weighted counterfactual explanations for a given QBAF."""

    def __init__(  # noqa: PLR0913
        self,
        qbaf: grad.BAG,
        agg_func,  # noqa: ANN001
        inf_func,  # noqa: ANN001
        topic_arg_name: str,
        random_seed: int | None = None,
        max_attempts: int = 100,
        epsilon: float = 1e-5,
        h: float = 0.1,
        M: int = 1000,  # noqa: N803
        delta: float = 1e-2,
        initial_cf_weights: dict[tuple[str, str], float] | None = None,
    ) -> None:
        """Initalise qbaf and generate explanation statistics.

        :param qbaf: The QBAF to generate explanations for (assumed acyclic).
        :param agg_func: The aggregation function to be used in the QBAF.
        :param inf_func: The influence function to be used in the QBAF.
        :param topic_arg_name: The name of the topic argument.
        :param random_seed: The random seed to be used for the QBAF.
        :param max_attempts: The maximum number of attempts to find a valid\
            counterfactual explanation.
        :param epsilon: The epsilon value for the gradient RAE.
        :param h: The step size for the search.
        :param M: The number of iterations for the search.
        :param delta: The delta neighbourhood for the search.
        """
        self.original_qbaf = qbaf
        self.qbaf = copy.deepcopy(qbaf)

        self.agg_func = agg_func
        self.inf_func = inf_func
        self.topic_arg_name = topic_arg_name
        self.epsilon = epsilon
        self.h = h
        self.M = M
        self.delta = delta

        if random_seed is not None:
            self.rng = random.Random(random_seed)  # noqa: S311
        else:
            self.rng = random.Random()  # noqa: S311

        self.ce_dict = self._compute_counterfactual_explanation(
            topic_arg_name,
            max_attempts,
            initial_cf_weights,
        )

    def _assign_random_edge_weights(self, bag: grad.BAG) -> grad.BAG:
        """Assign random edge weights to the QBAF.

        :param bag: The QBAF to assign random edge weights to.

        Returns:
            The QBAF with random edge weights.

        """
        for arg in bag.arguments.values():
            for supporter in arg.supporters:
                random_float = round(self.rng.uniform(0.0, 1.0), 2)
                arg.supporters[supporter] = random_float
            for attacker in arg.attackers:
                random_float = round(self.rng.uniform(0.0, 1.0), 2)
                arg.attackers[attacker] = random_float

        return bag

    def _assign_max_edge_weights(self, bag: grad.BAG) -> grad.BAG:
        """Assign edge weights corresponding to the maximum strength."""
        # traverse all edges
        for arg in bag.arguments.values():
            for supporter in arg.supporters:
                random_float = 1
                arg.supporters[supporter] = random_float
            for attacker in arg.attackers:
                random_float = 0
                arg.attackers[attacker] = random_float
        return bag

    def _assign_min_edge_weights(
        self,
        bag: grad.BAG,
        topic_arg: grad.Argument,
    ) -> grad.BAG:
        """Assign edge weights corresponding to the minimum strength."""
        # traverse all edges
        for arg in bag.arguments.values():
            for supporter in arg.supporters:
                random_float = 0 if arg == topic_arg else 1
                arg.supporters[supporter] = random_float
            for attacker in arg.attackers:
                random_float = 1 if arg == topic_arg else 0
                arg.attackers[attacker] = random_float
        return bag

    def _compute_max(self, bag: grad.BAG, topic_arg: grad.Argument) -> float:
        """Compute the maximum value attainable value of the topic argument.

        :param bag: The QBAF to compute the maximum value for.
        :param topic_arg: The topic argument to compute the maximum value for.

        Returns:
            The maximum value of the topic argument.

        """
        bag_copy = copy.deepcopy(bag)
        topic_arg_copy = bag_copy.arguments[topic_arg.name]

        for arg in bag_copy.arguments.values():
            for supporter in arg.supporters:
                arg.supporters[supporter] = 1.0
            for attacker in arg.attackers:
                arg.attackers[attacker] = 0.0

        grad.algorithms.computeStrengthValues(bag_copy, self.agg_func, self.inf_func)

        return topic_arg_copy.strength

    def _compute_min(self, bag: grad.BAG, topic_arg: grad.Argument) -> float:
        """Compute the minimum value attainable value of the topic argument.

        :param bag: The QBAF to compute the minimum value for.
        :param topic_arg: The topic argument to compute the minimum value for.

        Returns:
            The minimum value of the topic argument.

        """
        bag_copy = copy.deepcopy(bag)
        topic_arg_copy = bag_copy.arguments[topic_arg.name]

        for arg in bag_copy.arguments.values():
            for supporter in arg.supporters:
                arg.supporters[supporter] = 0.0 if (arg == topic_arg_copy) else 1.0
            for attacker in arg.attackers:
                arg.attackers[attacker] = 1.0 if (arg == topic_arg_copy) else 0.0

        grad.algorithms.computeStrengthValues(bag_copy, self.agg_func, self.inf_func)

        return topic_arg_copy.strength

    def _gRAE(  # noqa: N802
        self,
        epsilon: float,
        topic_arg: grad.Argument,
        bag: grad.BAG,
    ) -> dict[tuple[str, str], float]:
        """Compute the gradient RAE for a given topic argument.

        :param epsilon: The epsilon value for the gradient RAE.
        :param topic_arg: The topic argument to compute the gradient RAE for.
        :param bag: The QBAF to compute the gradient RAE for.

        Returns:
            The gradient RAE for the given topic argument.

        """
        grae = {}
        initial_strength = topic_arg.strength

        for arg in bag.arguments.values():
            for supporter in arg.supporters:
                arg.supporters[supporter] += epsilon
                grad.algorithms.computeStrengthValues(bag, self.agg_func, self.inf_func)
                peturbed_strength = topic_arg.strength
                grae[(supporter.name, arg.name)] = (
                    peturbed_strength - initial_strength
                ) / epsilon
                arg.supporters[supporter] -= epsilon
            for attacker in arg.attackers:
                arg.attackers[attacker] += epsilon
                grad.algorithms.computeStrengthValues(bag, self.agg_func, self.inf_func)
                peturbed_strength = topic_arg.strength
                grae[(attacker.name, arg.name)] = (
                    peturbed_strength - initial_strength
                ) / epsilon
                arg.attackers[attacker] -= epsilon
        return grae

    def _adjust_learning_rate(
        self,
        h: float,
        current_strength: float,
        prev_strength: float,
    ) -> float:
        """Adjust the learning rate based on the current and previous strength.

        :param h: The current learning rate.
        :param current_strength: The current strength of the topic argument.
        :param prev_strength: The previous strength of the topic argument.

        Returns:
            The adjusted learning rate.

        """
        threshold = 0.03
        factor = 0.8

        oscillation = abs(current_strength - prev_strength)
        return (
            max(0.001, h * factor) if oscillation > threshold else min(0.5, h / factor)
        )

    def _find_desired_strength(  # noqa: PLR0913
        self,
        epsilon: float,
        iter_limit: int,
        h: float,
        delta: float,
        desired_strength: float,
        topic_arg: grad.Argument,
        bag: grad.BAG,
    ) -> tuple[float, bool, dict]:
        """Find the desired strength of the topic argument.

        :param epsilon: The epsilon value for the search.
        :param M: The number of iterations for the search.
        :param h: The step size for the search.
        :param delta: The delta value for the search.
        :param desired_strength: The desired strength of the topic argument.
        :param topic_arg: The topic argument to find the desired strength for.
        :param bag: The QBAF to find the desired strength for.

        Returns:
            A tuple containing the current strength, a boolean indicating\
            whether a valid counterfactual explanation was found, and the edge\
            weights of the counterfactual explanation.

        """
        grad.algorithms.computeStrengthValues(bag, self.agg_func, self.inf_func)
        current_strength = topic_arg.strength

        while abs(desired_strength - current_strength) > delta and iter_limit > 0:
            grae = self._gRAE(
                epsilon,
                topic_arg,
                bag,
            )

            for arg in bag.arguments.values():
                for supporter in arg.supporters:
                    key = (supporter.name, arg.name)
                    if grae[key]:
                        grad_val = grae[key]
                        direction = 1 if current_strength < desired_strength else -1
                        adjusted_grad = direction * grad_val
                        update = arg.supporters[supporter] + h * adjusted_grad
                        arg.supporters[supporter] = max(0, min(1, update))
                for attacker in arg.attackers:
                    key = (attacker.name, arg.name)
                    if grae[key]:
                        grad_val = grae[key]
                        direction = 1 if current_strength < desired_strength else -1
                        adjusted_grad = direction * grad_val
                        update = arg.attackers[attacker] + h * adjusted_grad
                        arg.attackers[attacker] = max(0, min(1, update))
            grad.algorithms.computeStrengthValues(bag, self.agg_func, self.inf_func)
            prev_strength = current_strength
            current_strength = topic_arg.strength

            h = self._adjust_learning_rate(h, current_strength, prev_strength)
            iter_limit -= 1

        ce_dict = {}

        for arg in bag.arguments.values():
            for supporter in arg.supporters:
                ce_dict[(supporter.name, arg.name)] = arg.supporters[supporter]
            for attacker in arg.attackers:
                ce_dict[(attacker.name, arg.name)] = arg.attackers[attacker]

        valid = abs(desired_strength - current_strength) < delta

        return current_strength, valid, ce_dict

    def _assign_cf_weights(
        self,
        bag: grad.BAG,
        cf_weights: dict[tuple[str, str], float],
    ) -> grad.BAG:
        """Assign the given edge weights to the QBAF.

        :param bag: The QBAF to assign edge weights to.
        :param cf_weights: The edge weights to assign.

        Returns:
            The QBAF with the assigned edge weights.

        """
        for relation, weight in cf_weights.items():
            initial = bag.arguments[relation[0]]
            final = bag.arguments[relation[1]]

            if initial in final.attackers:
                final.attackers[initial] = weight
            else:
                final.supporters[initial] = weight
        return bag

    def _compute_counterfactual_explanation(
        self,
        topic_arg_name: str,
        max_attempts: int = 100,
        initial_cf_weights: dict[tuple[str, str], float] | None = None,
    ) -> dict[tuple[str, str], float]:
        """Compute the counterfactual explanation for a given topic argument.

        :param topic_arg_name: The topic argument to compute the counterfactual\
            explanation for.
        :param max_attempts: The maximum number of attempts to find a valid\
            counterfactual explanation.

        Returns:
            A dictionary containing the edge weights of the counterfactual\
            explanation.

        Raises:
            CounterfactualExplanationNotFoundError: If a valid\
            counterfactual explanation cannot be found after the\
            maximum number of attempts.

        """
        bag = self.qbaf

        if initial_cf_weights is not None:
            bag = self._assign_cf_weights(bag, initial_cf_weights)
        else:
            bag = self._assign_random_edge_weights(bag)

        topic_arg = bag.arguments[topic_arg_name]

        maximum = self._compute_max(bag, topic_arg)
        minimum = self._compute_min(bag, topic_arg)
        self.desired_strength = (maximum + minimum) / 2

        attempts = 0
        valid = False

        while attempts < max_attempts and not valid:
            attempts += 1
            _, valid, edge_weight = self._find_desired_strength(
                self.epsilon,
                self.M,
                self.h,
                self.delta,
                self.desired_strength,
                topic_arg,
                bag,
            )

            if valid:
                break

            if attempts % 3 == 1:
                bag = self._assign_random_edge_weights(bag)
            elif attempts % 3 == 2:  # noqa: PLR2004
                bag = self._assign_max_edge_weights(bag)
            else:
                bag = self._assign_min_edge_weights(bag, topic_arg)

        if not valid:
            raise CounterfactualExplanationNotFoundError
        return edge_weight

    def get_ce_explanation(
        self,
    ) -> dict[tuple[str, str], float]:
        """Get the counterfactual explanation.

        Returns:
            The counterfactual explanation.

        """
        return self.ce_dict
