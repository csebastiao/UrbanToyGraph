# UrbanToyGraph

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Toy graph to play with network metrics and algorithms mimicking typical urban patterns. The actual templates that can be modified by removing edges (and isolated nodes) are:

- Grid graph: A grid of m columns and n nodes. Can select a different width and height.
- Radial graph: Roads coming from a central node in a star-shaped structure. Can select the number of radial roads that will be evenly distributed around the central node.
- Concentric graph: Nodes that are on connected circles. Can choose the number of radial roads and of circles (called zones). Can choose to put a central node connected to the first circle.

## Installation

First clone the repository in the folder of your choosing:

```
git clone https://github.com/csebastiao/UrbanToyGraph.git
```

Locate yourself within the cloned folder, and create a new virtual environment. You can either create a new virtual environment then install the necessary dependencies with `pip` using the `requirements.txt` file:

```
pip install -r requirements.txt
```

Or create a new environment with the dependencies with `conda` or `mamba` using the `environment.yml` file:

```
mamba env create -f environment.yml
```

Once your environment is ready, you can locally install the package using:

```
pip install -e .
```
