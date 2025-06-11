# Code used for MEng Final Year Project: Arguing with Neural Networks

The project contains code from submodules: clone with the `--recurse-submodules` option. i.e.
```bash
git clone git@github.com:DanielPeacock/ArguingWithNeuralNetworksPublic.git --recurse-submodules
```
To initialise the correct `Python` version and packages, [`uv`](https://docs.astral.sh/uv/) can be used. Once installed (if not already) run `uv sync` from the project root.

## Structure

- The code for computing explanations, translation to QBAFs etc. is available in the [mlp_to_qbaf_converter](src/mlp_to_qbaf_converter) folder. Tests for the translation process are in the [test](src/test) folder.
- The modified code for SpArX (local) is in the [sparx](src/sparx) folder.
- The code for analysing explanation methods (AAEs/ RAEs/ CEs - see the report for more information) is available in the [analysis_scripts](src/analysis_scripts) folder. The outputs of running these scripts (for different sized MLPs) is available in the [outputs](outputs) folder.
- The plots and graphs in the report are in the [notebooks](src/notebooks) folder. See these files also for examples of using the library and the file [visualiser_example.ipynb](src/notebooks/visualiser_example.ipynb) for an example of using the visualiser pipeline.
