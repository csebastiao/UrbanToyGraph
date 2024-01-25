"""Useful functions."""

import math

from matplotlib import pyplot as plt
import geopandas as gpd
import numpy as np
import shapely
import scipy.spatial as sp
import networkx as nx


def make_osmnx_compatible(G):
    """Make the graph osmnx-compatible."""
    G = G.copy()
    for c, edge in enumerate(G.edges):
        G.edges["osmid"] = c
    G.graph["crs"] = "epsg:2154"
    G.graph["simplified"] = True
    if type(G) != nx.MultiDiGraph:
        G = nx.MultiDiGraph(G)
    return G


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
        G.edges[e]["geometry"] = shapely.from_wkt(G.edges[e]["geometry"])
    return G


def plot_graph(
    G,
    bb="square",
    show_voronoi=False,
    show=True,
    save=False,
    filepath=None,
    rel_buff=0.1,
):
    """Plot the graph using geopandas plotting function, with the option to save the picture."""
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
        side_length = max(bounds[3] - bounds[1], bounds[2] - bounds[0])
        mean_x = (bounds[0] + bounds[2]) / 2
        mean_y = (bounds[1] + bounds[3]) / 2
        xmin = mean_x - (1 + rel_buff) * side_length / 2
        xmax = mean_x + (1 + rel_buff) * side_length / 2
        ymin = mean_y - (1 + rel_buff) * side_length / 2
        ymax = mean_y + (1 + rel_buff) * side_length / 2
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    if show_voronoi:
        vor_buff = max(xmax - xmin, ymax - ymin)
        bb = np.array(
            [xmin - vor_buff, xmax + vor_buff, ymin - vor_buff, ymax + vor_buff]
        )
        bounded_vor = bounded_voronoi([get_node_coord(G, node) for node in G.nodes], bb)
        vor_cells = create_voronoi_polygons(bounded_vor)
        gdf_voronoi = gpd.GeoDataFrame(geometry=vor_cells)
        gdf_voronoi.geometry = gdf_voronoi.geometry.exterior
        gdf_voronoi.plot(ax=ax, color="firebrick", alpha=0.7, zorder=0)
    if show:
        plt.show()
    if save:
        if filepath is None:
            raise ValueError("If save is True, need to specify a filepath")
        fig.savefig(filepath, dpi=300)


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
    vec = np.array(vec[1]) - np.array(vec[0])
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


def bounded_voronoi(points, bb):
    """Make bounded voronoi cells for points. Made using https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells"""
    artificial_points = []
    xsl = bb[1] - bb[0]
    ysl = bb[3] - bb[2]
    artificial_points.append([bb[0], bb[2]])
    artificial_points.append([bb[0], bb[3]])
    artificial_points.append([bb[1], bb[2]])
    artificial_points.append([bb[1], bb[3]])
    for x in np.linspace(bb[0] - xsl, bb[1] + xsl, num=100, endpoint=False)[1:]:
        artificial_points.append([x, bb[2]])
        artificial_points.append([x, bb[3]])
    for y in np.linspace(bb[2] - ysl, bb[3] + ysl, num=100, endpoint=False)[1:]:
        artificial_points.append([bb[0], y])
        artificial_points.append([bb[1], y])
    points = np.concatenate([points, artificial_points])
    vor = sp.Voronoi(points)
    regions = []
    points_ordered = []
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
    vor.filtered_points = points_ordered
    vor.filtered_regions = regions
    return vor


def create_voronoi_polygons(vor, filtered=True):
    """Create polygons from Voronoi regions."""
    vor_poly = []
    attr = vor.regions
    if filtered:
        attr = vor.filtered_regions
    for region in attr:
        vertices = vor.vertices[region, :]
        vor_poly.append(shapely.Polygon(vertices))
    return vor_poly
