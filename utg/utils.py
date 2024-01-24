"""Useful functions."""

import math

from matplotlib import pyplot as plt
import geopandas as gpd
import numpy as np
import shapely
import networkx as nx


def save_graph(G, filepath):
    """Save the graph in the corresponding filepath, converting geometry to WKT string."""
    G = G.copy()
    for e in G.edges:
        G.edges[e]["geometry"] = shapely.to_wkt(G.edges[e]["geometry"])
    nx.write_graphml(G, filepath)


def load_graph(filepath):
    """Load the graph from the corresponding filepath, creating geometry from WKT string."""
    G = nx.read_graphml(filepath)
    for e in G.edges:
        G.edges[e]["geometry"] = shapely.from_wkt_wkt(G.edges[e]["geometry"])
    return G


def plot_graph(G, bb="square", show=True, save=False, filepath=None):
    fig, ax = plt.subplots()
    geom_node = [shapely.Point(get_node_coord(G, node)) for node in G.nodes]
    geom_edge = list(nx.get_edge_attributes(G, "geometry").values())
    gdf_node = gpd.GeoDataFrame(geometry=geom_node)
    gdf_edge = gpd.GeoDataFrame(geometry=geom_edge)
    gdf_edge.plot(ax=ax, color="steelblue", zorder=1, linewidth=2)
    gdf_node.plot(ax=ax, color="black", zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    bounds = gdf_node.total_bounds
    if bb == "square":
        BUFFER = 0.1
        side_length = max(bounds[3] - bounds[1], bounds[2] - bounds[0])
        mean_x = (bounds[0] + bounds[2]) / 2
        mean_y = (bounds[1] + bounds[3]) / 2
        xmin = mean_x - (1 + BUFFER) * side_length / 2
        xmax = mean_x + (1 + BUFFER) * side_length / 2
        ymin = mean_y - (1 + BUFFER) * side_length / 2
        ymax = mean_y + (1 + BUFFER) * side_length / 2
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    if show:
        plt.show()
    if save:
        if filepath is None:
            raise ValueError("If save is True, need to specify a filepath")
        fig.savefig(filepath)


def make_true_zero(vec):
    """Round to zero when values are very close to zero in a list."""
    return [round(val) if math.isclose(val, 0, abs_tol=1e-10) else val for val in vec]


def get_node_coord(G, n):
    """Return the coordinates of the node."""
    return [G.nodes[n]["x"], G.nodes[n]["y"]]


def normalize(vec):
    """Normalize the vector."""
    return vec / np.linalg.norm(vec)


def find_angle(vec):
    """Find the angle of the vector to the origin and the horizontal axis."""
    normvec = make_true_zero(normalize(vec))
    if normvec[1] >= 0:
        return np.arccos(normvec[0])
    elif normvec[0] >= 0:
        angle = np.arcsin(normvec[1])
        if angle < 0:
            angle += 2 * np.pi
        return angle
    else:
        return np.arccos(normvec[0]) + np.pi / 2
