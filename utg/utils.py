"""Useful functions."""

import math

from matplotlib import pyplot as plt
import geopandas as gpd
import numpy as np
import shapely
import scipy.spatial as sp
import networkx as nx

__all__ = [
    "make_osmnx_compatible",
    "save_graph",
    "load_graph",
    "plot_graph",
    "get_node_coord",
]


def make_osmnx_compatible(G):
    """
    Make the graph osmnx-compatible.

    Args:
        G (networkx.Graph or networkx.MultiDiGraph): Graph generated from any .create_graph function that will be made osmnx compatible.

    Returns:
        networkx.MultiDiGraph: Graph G made osmnx compatible.
    """
    G = G.copy()
    for c, edge in enumerate(G.edges):
        G.edges[edge]["osmid"] = c
    G.graph["crs"] = "epsg:2154"
    G.graph["simplified"] = True
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)
    return G


def save_graph(G, filepath):
    """
    Save the graph in the corresponding filepath, converting geometry to WKT string.

    Args:
        G (networkx.Graph or networkx.MultiDiGraph): Graph to be saved.
        filepath (str): Path where the graph will be saved.
    """
    G = G.copy()
    for e in G.edges:
        G.edges[e]["geometry"] = shapely.to_wkt(G.edges[e]["geometry"])
    nx.write_graphml(G, filepath)


def load_graph(filepath):
    """
    Load the graph from the corresponding filepath, creating geometry from WKT string.

    Args:
        filepath (str): Path locating the graph to be loaded.

    Returns:
        networkx.Graph or networkx.MultiDiGraph: Loaded graph.
    """
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, lambda x: int(x))
    for e in G.edges:
        G.edges[e]["geometry"] = shapely.from_wkt(G.edges[e]["geometry"])
    return G


def plot_graph(
    G,
    square_bb=True,
    figsize=(6, 6),
    show_voronoi=False,
    voronoi_color="firebrick",
    voronoi_alpha=0.7,
    show=True,
    save=False,
    close=False,
    filepath=None,
    rel_buff=0.1,
    edge_color="black",
    node_color="black",
    edge_linewidth=2,
    node_size=20,
    dpi=300,
    no_border=True,
    tight=True,
):
    """
    Plot the graph using geopandas plotting function, with the option to save the picture and see the Voronoi cells.

    Args:
        G (nx.Graph or nx.MultiDiGraph): Graph we want to plot.
        square_bb (bool, optional): If True, limits of the figure are a square centered around the graph. Defaults to True.
        figsize (tuple, optional): Size of the figure. Defaults to (10, 10).
        show_voronoi (bool, optional): If True, show the Voronoi cells for each nodes. Defaults to False.
        voronoi_color (str, optional): Color of the Voronoi cells. Defaults to firebrick.
        voronoi_alpha (float, optional): Transparency of the Voronoi cells. Defaults to 0.7.
        show (bool, optional): If True, show the figure in Python. Defaults to True.
        save (bool, optional): If True, save the figure at the designated filepath. Defaults to False.
        filepath (_type_, optional): Path for the saved figure. Defaults to None.
        rel_buff (float, optional): Relative buffer around the nodes, creating padding for the square bounding box. For instance a padding of 10% around each side of the graph for a value of 0.1. Defaults to 0.1.
        edge_color (str, optional): Color of the edges. Defaults to black.
        node_color (str, optional): Color of the nodes. Defaults to black.
        edge_linewidth (int, optional): Width of the edges. Defaults to 2.
        node_size (int, optional): Size of the nodes. Defaults to 20.
        dpi (int, optional): Dots per inches of the figure. Defaults to 300.
        no_border (str, optional): If True, remove borders of the plot. Defaults to True.
        tight (bool, optional): If True, use tight layout for the figure. Defaults to True.

    Raises:
        ValueError: If save is True, need to specify a filepath. Filepath can't be None.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if tight:
        fig.set_layout_engine("tight")
    geom_node = [shapely.Point(get_node_coord(G, node)) for node in G.nodes]
    geom_edge = list(nx.get_edge_attributes(G, "geometry").values())
    gdf_node = gpd.GeoDataFrame(geometry=geom_node)
    gdf_edge = gpd.GeoDataFrame(geometry=geom_edge)
    gdf_edge.plot(ax=ax, color=edge_color, zorder=1, linewidth=edge_linewidth)
    gdf_node.plot(ax=ax, color=node_color, zorder=2, markersize=node_size)
    ax.set_xticks([])
    ax.set_yticks([])
    bounds = gdf_node.total_bounds
    if square_bb:
        # Find if graph is larger in width or height
        side_length = max(bounds[3] - bounds[1], bounds[2] - bounds[0])
        # Find center of the graph
        mean_x = (bounds[0] + bounds[2]) / 2
        mean_y = (bounds[1] + bounds[3]) / 2
        # Add padding
        xmin = mean_x - (1 + rel_buff) * side_length / 2
        xmax = mean_x + (1 + rel_buff) * side_length / 2
        ymin = mean_y - (1 + rel_buff) * side_length / 2
        ymax = mean_y + (1 + rel_buff) * side_length / 2
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    if show_voronoi:
        # Create a bounding box to create bounded Voronoi cells that can easily be drawn
        vor_buff = max(xmax - xmin, ymax - ymin)
        bb = np.array(
            [xmin - vor_buff, xmax + vor_buff, ymin - vor_buff, ymax + vor_buff]
        )
        bounded_vor = bounded_voronoi([get_node_coord(G, node) for node in G.nodes], bb)
        vor_cells = create_voronoi_polygons(bounded_vor)
        gdf_voronoi = gpd.GeoDataFrame(geometry=vor_cells)
        gdf_voronoi.geometry = gdf_voronoi.geometry.exterior
        gdf_voronoi.plot(ax=ax, color=voronoi_color, alpha=voronoi_alpha, zorder=0)
    if no_border:
        ax.axis("off")
    if show:
        plt.show()
    if save:
        if filepath is None:
            raise ValueError("If save is True, need to specify a filepath")
        fig.savefig(filepath, dpi=dpi)
    if close:
        plt.close()
    return fig, ax


def make_true_zero(vec):
    """Round to zero when values are very close to zero in a list."""
    return [round(val) if math.isclose(val, 0, abs_tol=1e-6) else val for val in vec]


def get_node_coord(G, n):
    """Return the coordinates of the node."""
    return [G.nodes[n]["x"], G.nodes[n]["y"]]


def normalize(vec):
    """Normalize the vector."""
    return vec / np.linalg.norm(vec)


# TODO: Look at shapely voronoi to maybe make a change for better written code
# TODO: Verify code
def bounded_voronoi(points, bb):
    """
    Make bounded voronoi cells for points by creating a large square of artifical points far away.

    Args:
        points (list): List of coordinates, as [[x1, y1], [x2, y2], ..., [xn, yn]].
        bb (list): Bounding box constructed as [xmin, xmax, ymin, ymax]

    Returns:
        scipy.spatial.Voronoi: Voronoi cells created from the list of points, bounded by the bounding box.
    """
    # TODO add test to bb and points
    artificial_points = []
    # Make artifical points outside of the bounding box
    artificial_points.append([bb[0], bb[2]])
    artificial_points.append([bb[0], bb[3]])
    artificial_points.append([bb[1], bb[2]])
    artificial_points.append([bb[1], bb[3]])
    for x in np.linspace(bb[0], bb[1], num=100, endpoint=False)[1:]:
        artificial_points.append([x, bb[2]])
        artificial_points.append([x, bb[3]])
    for y in np.linspace(bb[2], bb[3], num=100, endpoint=False)[1:]:
        artificial_points.append([bb[0], y])
        artificial_points.append([bb[1], y])
    points = np.concatenate([points, artificial_points])
    # Find Voronoi regions
    vor = sp.Voronoi(points)
    regions = []
    points_ordered = []
    # Keep regions for points that are within the bounding box so only the original points
    for c, region in enumerate(vor.regions):
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (bb[0] <= x and x <= bb[1] and bb[2] <= y and y <= bb[3]):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
            points_ordered.append(np.where(vor.point_region == c)[0][0])
    # Create filtered attributes to keep in memory the original points and related Voronoi regions
    vor.filtered_points = points_ordered
    vor.filtered_regions = regions
    return vor


# TODO: Redo but instead start from voronoi, and do intersection with bounding box to avoid curved shapes
def create_voronoi_polygons(vor, filtered=True):
    """
    Create polygons from Voronoi regions. Use the filtered attributes from bounded_voronoi.

    Args:
        vor (scipy.spatial.Voronoi): Voronoi cells.
        filtered (bool, optional): If True, use filtered regions instead of the true regions. Defaults to True.

    Returns:
        list: List of shapely.Polygons corresponding to the Voronoi cells.
    """
    vor_poly = []
    attr = vor.regions
    if filtered:
        attr = vor.filtered_regions
    for region in attr:
        vertices = vor.vertices[region, :]
        vor_poly.append(shapely.Polygon(vertices))
    return vor_poly
