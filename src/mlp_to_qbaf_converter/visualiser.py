"""Visualiser for QBAF and explanations."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import bokeh
import bokeh.plotting
import numpy as np
import panel as pn
from bokeh.models import (
    BoxZoomTool,
    Ellipse,
    HoverTool,
    MultiLine,
    PanTool,
    ResetTool,
    WheelZoomTool,
)
from tqdm import tqdm

from mlp_to_qbaf_converter.argument_attribution_explanation import AAE
from mlp_to_qbaf_converter.mlp_to_qbaf import MLPToQBAF
from mlp_to_qbaf_converter.relation_attribution_explanation import RAE
from mlp_to_qbaf_converter.utils import generate_nx_graph
from sparx.sparx import ClusteringMethod, LocalSpArX
from Uncertainpy.src.uncertainpy import gradual as grad

if TYPE_CHECKING:
    import networkx as nx

pn.extension("bokeh")
pn.config.sizing_mode = "scale_both"


class AAEScoreType(Enum):
    """Enum for AAE score types."""

    GRADIENT = "gradient"
    SHAP = "shap"
    REMOVAL = "removal"


class RAEScoreType(Enum):
    """Enum for RAE score types."""

    SHAP = "shap"
    REMOVAL = "removal"


class Visualiser:
    """Class to visualise the QBAF and explanations.

    This class generates QBAFs for the original neural network and for
    the neural network with different levels of sparsity. It also generates
    AAEs and RAEs for the original neural network and for the neural network
    with different levels of sparsity. These can then be used to visualise
    the explanations for the neural network.
    """

    def __init__(  # noqa: PLR0913
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        activation_function: str,
        example_row: np.ndarray,
        input_names: list[str],
        output_names: list[str],
        topic_argument: str,
        train_set: np.ndarray,
        kernel_size: float,
        sparse_percents: list[float] | None = None,
        clustering_method: str = ClusteringMethod.AAE_GRADIENT,
        do_aaes: bool = True,  # noqa: FBT001, FBT002
        do_raes: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the Visualiser class."""
        self.weights = weights
        self.biases = biases
        self.activation_function = activation_function
        self.h = 1e-10
        self.example_row = example_row
        self.example_row[:-1] = np.clip(
            self.example_row[:-1],
            self.h,
            1 - self.h,
        )
        self.input_names = input_names
        self.output_names = output_names
        self.topic_argument = topic_argument
        self.train_set = train_set
        self.kernel_size = kernel_size
        self.clustering_method = clustering_method

        if sparse_percents is None:
            self.sparse_percents = [20, 40, 60, 80]
        else:
            self.sparse_percents = sparse_percents

        self.qbafs = self.create_qbafs()
        self.nx_graphs = self.generate_nx_graphs()

        self.do_aaes = do_aaes
        self.do_raes = do_raes

        if do_aaes:
            self.aaes = self.create_aaes()
        if do_raes:
            self.raes = self.create_raes()

    def create_qbafs(self) -> list[grad.BAG]:
        """Create QBAFs for the neural network.

        Returns:
        A list of QBAFs for the neural network.

        """
        qbafs = []

        neurons_per_layer = np.array(
            [weight.shape[0] for weight in self.weights] + [self.weights[-1].shape[1]],
        )

        original_qbaf = MLPToQBAF(
            neurons_per_layer,
            self.weights,
            self.biases,
            self.activation_function,
            self.input_names,
            self.output_names,
            self.example_row[:-1],
        ).get_qbaf()

        qbafs.append(original_qbaf)

        for sparse_percent in self.sparse_percents:
            sp = LocalSpArX(
                self.weights,
                self.biases,
                self.activation_function,
                sparse_percent,
                self.example_row,
                self.train_set,
                self.kernel_size,
                self.input_names,
                self.output_names,
                self.topic_argument,
                self.clustering_method,
            )

            shape = sp.get_sparsified_shape()
            s_weights, s_biases = sp.get_sparsified_mlp()

            sparse_qbaf = MLPToQBAF(
                shape,
                s_weights,
                s_biases,
                self.activation_function,
                self.input_names,
                self.output_names,
                self.example_row[:-1],
            ).get_qbaf()
            qbafs.append(sparse_qbaf)

        return qbafs

    def create_aaes(self) -> list[AAE]:
        """Create AAE explanations for the neural network.

        Returns:
        A list of AAE objects for the QBAFs.

        """
        aaes = []
        h = 1e-10
        for qbaf in tqdm(self.qbafs, desc="Creating AAEs"):
            aae = AAE(
                qbaf,
                grad.SumAggregation(),
                grad.MLPBasedInfluence(),
                self.topic_argument,
                h,
            )
            aaes.append(aae)

        return aaes

    def create_raes(self) -> list[RAE]:
        """Create RAE explanations for the neural network.

        Returns:
        A list of RAE objects for the QBAFs.

        """
        raes = []
        for qbaf in tqdm(self.qbafs, desc="Creating RAEs"):
            rae = RAE(
                qbaf,
                grad.SumAggregation(),
                grad.MLPBasedInfluence(),
                self.topic_argument,
            )
            raes.append(rae)

        return raes

    def get_scores_aae(self, aae: AAE, score_type: AAEScoreType) -> dict[str, float]:
        """Get the AAE and RAE scores for the QBAFs.

        :param aae: The AAE object.
        :param score_type: The type of AAE score to get.

        Returns:
        The corresponding AAE scores dictionary.

        Raises:
        ValueError: If the score type is not supported.

        """
        match score_type:
            case AAEScoreType.GRADIENT:
                return aae.get_gradients()
            case AAEScoreType.SHAP:
                return aae.get_shap_scores()
            case AAEScoreType.REMOVAL:
                return aae.get_removal_scores()
            case _:
                msg = f"Unknown AAE score type: {score_type}"
                raise ValueError(msg)

    def get_scores_rae(self, rae: RAE, score_type: RAEScoreType) -> dict[str, float]:
        """Get the RAE scores for the QBAFs.

        :param rae: The RAE object.
        :param score_type: The type of RAE score to get.

        Returns:
        The corresponding RAE scores dictionary.

        Raises:
        ValueError: If the score type is not supported.

        """
        match score_type:
            case RAEScoreType.SHAP:
                return rae.get_shap_scores()
            case RAEScoreType.REMOVAL:
                return rae.get_removal_scores()
            case _:
                msg = f"Unknown RAE score type: {score_type}"
                raise ValueError(msg)

    def generate_nx_graphs(self) -> dict[float, nx.DiGraph]:
        """Generate NetworkX graphs from the QBAFs.

        Returns:
        A dictionary of NetworkX graphs for each QBAF.

        """
        return [generate_nx_graph(qbaf) for qbaf in self.qbafs]

    def render_graph(  # noqa: C901, PLR0912, PLR0915
        self,
        graph: nx.DiGraph,
        aae: AAE | None = None,
        rae: RAE | None = None,
        aae_score_type: AAEScoreType = AAEScoreType.GRADIENT,
        rae_score_type: RAEScoreType = RAEScoreType.SHAP,
    ) -> bokeh.models.Plot:
        """Render the NetworkX graph using Bokeh.

        :param graph: The NetworkX graph to render.
        :param aae: The AAE object to use for the graph. Optional.
        :param rae: The RAE object to use for the graph. Optional.
        :param aae_score_type: The type of AAE score to use for the graph.\
            Only used if aae is provided.
        :param rae_score_type: The type of RAE score to use for the graph.\
            Only used if rae is provided.

        Returns:
        The Bokeh plot object.

        """
        nodes = list(graph.nodes())
        nodes_split = []
        layers = []

        for node in nodes:
            if node in self.input_names or node in self.output_names:
                continue

            split = node.split(" ")
            layer = split[1]
            neuron = split[3]

            nodes_split.append((layer, neuron))

        layers = sorted({layer for layer, _ in nodes_split})

        nodes_per_layer = {}

        for layer in layers:
            nodes_per_layer[layer] = [
                neuron for lr, neuron in nodes_split if lr == layer
            ]

        nodes_per_layer["input"] = self.input_names
        nodes_per_layer["output"] = self.output_names

        # Create a layout for the nodes

        pos = {}
        scaling = 2

        for layer in layers:
            for i, neuron in enumerate(nodes_per_layer[layer]):
                x_pos = int(layer)
                y_pos = (1 / len(nodes_per_layer[layer])) * (i + 1) * scaling
                pos[f"Layer {layer} Neuron {neuron}"] = (x_pos, y_pos)

        for i, neuron in enumerate(self.input_names):
            x_pos = 0
            y_pos = (1 / len(self.input_names)) * (i + 1) * scaling
            pos[neuron] = (x_pos, y_pos)

        for i, neuron in enumerate(self.output_names):
            x_pos = len(layers) + 1
            y_pos = (1 / len(self.output_names)) * (i + 1) * scaling
            pos[neuron] = (x_pos, y_pos)

        node_labels = []
        node_labels.append(("Name", "@name"))

        if aae:
            scores = self.get_scores_aae(aae, aae_score_type)
            for node in graph.nodes():
                if node == self.topic_argument:
                    graph.nodes[node]["aae_score"] = "Topic Argument"
                else:
                    graph.nodes[node]["aae_score"] = scores[node]

            node_labels.append(("AAE Score", "@aae_score"))

        if rae:
            scores = self.get_scores_rae(rae, rae_score_type)
            for relation in scores:
                if "Attack" in relation:
                    relation_tmp = relation.replace("Attack(", "").replace(")", "")
                else:
                    relation_tmp = relation.replace("Support(", "").replace(")", "")

                relation_tmp = relation_tmp.split(" to ")
                inital = relation_tmp[0]
                final = relation_tmp[1]
                graph.edges()[(inital, final)]["rae_score"] = scores[relation]

        min_width = 0.1

        for u, v in graph.edges():
            weight = graph.edges[u, v].get("edge_weight", min_width)
            graph.edges[u, v]["line_width"] = max(weight, min_width)

        bokeh_graph = bokeh.plotting.from_networkx(graph, pos)

        bokeh_graph.edge_renderer.glyph = MultiLine(
            line_color="edge_color",
            line_width="line_width",
            line_dash="style",
        )
        bokeh_graph.node_renderer.glyph = Ellipse(
            height=0.3,
            width=0.3,
            fill_color="fill_color",
        )

        hover_nodes = HoverTool(
            tooltips=node_labels,
            renderers=[bokeh_graph.node_renderer],
        )

        hover_edges = HoverTool(
            tooltips=[("RAE Score", "@rae_score"), ("Weight", "@edge_weight")],
            renderers=[bokeh_graph.edge_renderer],
            line_policy="interp",
        )

        plot = bokeh.models.Plot(
            sizing_mode="stretch_both",
        )
        plot.add_tools(
            hover_nodes,
            hover_edges,
            BoxZoomTool(),
            ResetTool(),
            PanTool(),
            WheelZoomTool(),
        )

        plot.renderers.append(bokeh_graph)

        return plot

    def show(self) -> None:
        """Show the visualiser.

        Creates a Panel app to visualise the QBAFs and explanations.
        It allows the user to select different QBAFs and explanations to visualise.
        """
        selectors = pn.Row(
            pn.widgets.Select(
                name="Select Sparsifcation Level",
                options=[0, *self.sparse_percents],
                value=0,
            ),
        )
        if self.do_aaes:
            selectors.append(
                pn.widgets.Select(
                    name="Select AAE Score",
                    options=[e.value for e in AAEScoreType],
                    value=AAEScoreType.GRADIENT.value,
                ),
            )
        if self.do_raes:
            selectors.append(
                pn.widgets.Select(
                    name="Select RAE Score",
                    options=[e.value for e in RAEScoreType],
                    value=RAEScoreType.SHAP.value,
                ),
            )

        bokeh_pane = pn.pane.Bokeh(
            sizing_mode="stretch_both",
            min_height=400,
        )  # Set a fixed height for the pane

        def update(event=None) -> None:  # noqa: ANN001, ARG001
            """Update the graph based on the selected options."""
            selected_sparsity = selectors[0].value
            selected_aae_score = selectors[1].value if self.do_aaes else None
            selected_rae_score = selectors[2].value if self.do_raes else None

            if selected_sparsity == 0:
                graph = self.nx_graphs[0]
                aae = self.aaes[0] if self.do_aaes else None
                rae = self.raes[0] if self.do_raes else None
            else:
                idx = self.sparse_percents.index(selected_sparsity) + 1
                graph = self.nx_graphs[idx]
                aae = self.aaes[idx] if self.do_aaes else None
                rae = self.raes[idx] if self.do_raes else None

            plot = self.render_graph(
                graph,
                aae=aae,
                rae=rae,
                aae_score_type=AAEScoreType(selected_aae_score)
                if selected_aae_score
                else None,
                rae_score_type=RAEScoreType(selected_rae_score)
                if selected_rae_score
                else None,
            )
            bokeh_pane.object = plot

        update_button = pn.widgets.Button(name="Update", button_type="primary")
        update_button.on_click(update)

        page = pn.Column(
            pn.pane.Markdown("## QBAF Visualiser"),
            pn.pane.Markdown("### Select a QBAF to visualise"),
            selectors,
            update_button,
            bokeh_pane,
        )

        update()

        page.show("Visualise QBAF and Explanations")
