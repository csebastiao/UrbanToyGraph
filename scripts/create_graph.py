"""Script to create networkx graphs and save them."""

import warnings

import networkx as nx
import numpy as np
import math
import shapely


def create_grid_graph(m=3, n=3, width=50, height=None):
    """Create a grid graph of arbitrary size.

    Args:
        m (int, optional): Number or columns. Defaults to 3.
        n (int, optional): Number of rows. Defaults to 3.
        width (int or float, optional): Length in the x coordinate. If height is not defined, is the square's length. Defaults to 50.
        height (int or float, optional): If not None, length of the y coordinate. Defaults to None.

    Raises:
        ValueError: width needs to be positive.

    Returns:
        G (networkx.MultiDiGraph): Grid-like graph.
    """
    G = nx.grid_2d_graph(m, n, create_using=nx.MultiDiGraph)
    if width <= 0:
        raise ValueError("Width needs to be positive.")
    if height is None:
        height = width
    else:
        warnings.warn(
            "Height value selected, if different than width, will create rectangles instead of squares."
        )
    count = 0
    for node in G.nodes:
        x, y = node
        G.nodes[node]["x"] = x * width
        G.nodes[node]["y"] = y * height
        for edge in list(G.in_edges(node)) + list(G.out_edges(node)):
            count += 1
            first, second = edge
            fx, fy = first
            sx, sy = second
            # Edges' geometry is straight line between the linked nodes
            G.edges[(first, second, 0)]["geometry"] = shapely.LineString(
                [(fx * width, fy * height), (sx * width, sy * height)]
            )
            # Added to make it osmnx-compatible
            G.edges[(first, second, 0)]["osmid"] = count
    # To make easier node labels
    G = nx.convert_node_labels_to_integers(G)
    # Added to make it osmnx-compatible
    G.graph["crs"] = "epsg:2154"
    G.graph["simplified"] = True
    return G


def create_radial_graph(radial=4, length=50):
    """Create a radial graph where roads are radiating from a center.

    Args:
        radial (int, optional): Number of roads arranged evenly around the center. Needs to be at least 2. Defaults to 4.
        length (int, optional): Lengths of the roads. Defaults to 50.

    Raises:
        ValueError: Radial graph needs at least 2 radial roads to work.

    Returns:
        G (networkx.MultiDiGraph): Radial graph.
    """
    if radial < 2:
        raise ValueError("Radial graph needs at least 2 radial roads to work.")
    G = nx.MultiDiGraph()
    G.add_node(0, x=0, y=0)
    count = 0
    # Nodes are evenly distributed on a circle
    for i in range(radial):
        G.add_node(
            i + 1,
            x=length * np.cos(i * 2 * np.pi / radial),
            y=length * np.sin(i * 2 * np.pi / radial),
        )
        pos = coord_node(G, i + 1)
        G.add_edge(0, i + 1, geometry=shapely.LineString([(0, 0), pos]), osmid=count)
        count += 1
        # Add edge in both directions
        G.add_edge(i + 1, 0, geometry=shapely.LineString([pos, (0, 0)]), osmid=count)
        count += 1
    # Added to make it osmnx-compatible
    G.graph["crs"] = "epsg:2154"
    G.graph["simplified"] = True
    return G


def make_true_zero(vec):
    """Round to zero when values are very close to zero in a list."""
    return [round(val) if math.isclose(val, 0, abs_tol=1e-10) else val for val in vec]

def coord_node(G, n):
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
        return np.arcsin(normvec[1])
    else:
        # TODO: FIX IT
        return np.arccos(normvec[0]) + np.pi / 2


def create_concentric_graph(radial=8, zones=3, radius=30, center=True):
    """Create a concentric graph, where nodes are on circular zones, connected to their nearest neighbors and to the next zone.

    Args:
        radial (int, optional): Number of nodes per zone. Nodes are evenly distributed on each circle. Needs to be at least 2. Defaults to 8.
        zones (int, optional): Number of zones. Needs to be at least 1. Defaults to 3.
        radius (int, optional): Radius between zones. Defaults to 30.
        center (bool, optional): If True, add a node at the center of the graph.

    Raises:
        ValueError: Needs two node per zone at least.
        ValueError: Needs one zone at least.

    Returns:
        G (networkx.MultiDiGraph): Concentric graph.
    """
    if radial < 2:
        raise ValueError("Concentric graph needs at least 2 radial positions to work.")
    if zones < 1:
        raise ValueError("Number of zones needs to be positive.")
    G = nx.MultiDiGraph()
    count = 0
    if center is True:
        G.add_node(count, x=0, y=0)
        count += 1
    # Zones increase the radius
    for i in range(zones):
        # Nodes are evenly distributed on a circle
        for j in range(radial):
            G.add_node(
                count,
                x=radius * (i + 1) * np.cos(j * 2 * np.pi / radial),
                y=radius * (i + 1) * np.sin(j * 2 * np.pi / radial),
            )
            count += 1
    count = 0
    # If there is a center node, shift the ID of the nodes in the zones by 1.
    startnum = 0
    if center is True:
        startnum += 1
        for i in range(radial):
            pos = [G.nodes[i + 1]["x"], G.nodes[i + 1]["y"]]
            G.add_edge(0, i, geometry=shapely.LineString([(0, 0), pos]), osmid=count)
            count += 1
            # Add edge in both directions
            G.add_edge(i, 0, geometry=shapely.LineString([pos, (0, 0)]), osmid=count)
            count += 1
    for i in range(zones):
        for j in range(startnum, startnum + radial - 1):
            fn = i * radial + j
            sn = i * radial + j + 1
            fc = coord_node(G, fn)
            sc = coord_node(G, sn)
            # Add edge in both directions
            G.add_edge(
                fn,
                sn,
                geometry=create_curved_linestring(fc, sc, radius * (i + 1)),
                osmid=count,
            )
            count += 1
            G.add_edge(
                sn,
                fn,
                geometry=create_curved_linestring(sc, fc, radius * (i + 1)),
                osmid=count,
            )
            count += 1
            # Connect nodes to next zone if there is one
            if i < zones - 1:
                tn = (i + 1) * radial + j
                tc = coord_node(G, tn)
                G.add_edge(fn, tn, geometry=shapely.LineString([fc, tc]), osmid=count)
                count += 1
                # Add edge in both directions
                G.add_edge(tn, fn, geometry=shapely.LineString([tc, fc]), osmid=count)
                count += 1
    # Added to make it osmnx-compatible
    G.graph["crs"] = "epsg:2154"
    G.graph["simplified"] = True
    return G


# Simpler but less general function in the meantime
def create_curved_linestring(startpoint, endpoint, radius):
    N = 100
    coords = [startpoint]
    start_angle = find_angle(startpoint)
    end_angle = find_angle(endpoint)
    angle_coords = np.linspace(start_angle, end_angle, num=N - 1, endpoint=False)
    for i in range(1, N - 1):
        coords.append(
            [radius * np.cos(angle_coords[i]), radius * np.sin(angle_coords[i])]
        )
    coords.append(endpoint)
    return shapely.LineString(coords)


# TODO: Find way to know the coordinates of the n_coord points. Maybe find the center of the circle and draw from here ?
# Need to be able to specify which side because there are always two center (except if right in the middle) !
def WIP_create_curved_linestring(
    startpoint, endpoint, radius, side="smaller", n_coord=100
):
    """Create a curved linestring between two points.

    The function suppose that the startpoint and the endpoint are both on a circle of a given radius and create the corresponding shapely.LineString, with the number of points on the LineString being n_coord.

    Args:
        startpoint (array-like): (x,y) coordinates of the first point.
        endpoint (array-like): (x,y) coordinates of the last point.
        radius (int or float): Radius of the circle on which the points are.
        side (str, optional):  Side on which the center of the circle is. The options are smaller and bigger, meaning if the sum of the coordinates of the center is smaller or bigger than the average sum of the coordinates of the two points. Defaults to "smaller".
        n_coord (int, optional): Number of coordinates of the linestring. A higher number means a better, more refined curve. Defaults to 100.

    Raises:
        ValueError: The radius needs to be at least as long as the Euclidean distance between the points.

    Returns:
        curve (shapely.LineString): Curved linestring between the two points.
    """
    if np.linalg.norm([startpoint, endpoint]) > radius:
        raise ValueError(
            "Radius needs to be larger than the Euclidean distance between the points."
        )
    coords = []
    for i in range(n_coord):
        coords.append([i, i])
    curve = shapely.LineString(coords)
    return curve

# TODO: Function to randomly remove a number of edges, while keeping the network connected. To break the perfection of the toy graphs.
def remove_random_edges(G, N=1, prevent_disconnect=True):
    return G