"""Script to create networkx graphs and save them."""

import math
import random
import warnings

import networkx as nx
import numpy as np
import shapely

from . import utils


def create_grid_graph(
    rows=3, cols=3, width=50, height=None, multidigraph=True, diagonal=False
):
    """Create a grid graph of arbitrary size.

    Args:
        rows (int, optional): Number or rows. Defaults to 3.
        cols (int, optional): Number of columns. Defaults to 3.
        width (int or float, optional): Length in the x coordinate. If height is not defined, is the square's length. Defaults to 50.
        height (int or float, optional): If not None, length of the y coordinate. Defaults to None.
        multidigraph (bool, optional): If True, return Graph as MultiDiGraph. Graph is better for computations and ease of use, MultiDiGraph is more general and osmnx-compatible. Defaults to True.
        diagonal (bool, optional): If True, create diagonal edges along the square. Works only if there is an equal amount of rows and columns. Defaults to False.

    Raises:
        ValueError: width needs to be positive.

    Returns:
        G (networkx.Graph or networkx.MultiDiGraph): Grid-like graph.
    """
    G = nx.grid_2d_graph(rows, cols, create_using=nx.Graph)
    if width <= 0:
        raise ValueError("Width needs to be positive.")
    if height is None:
        height = width
    else:
        warnings.warn(
            "Height value selected, if different than width, will create rectangles instead of squares."
        )
    for node in G.nodes:
        x, y = node
        G.nodes[node]["x"] = x * width
        G.nodes[node]["y"] = y * height
        for c, edge in enumerate(list(G.edges(node))):
            first, second = edge
            fx, fy = first
            sx, sy = second
            # Edges' geometry is straight line between the linked nodes
            G.edges[(first, second)]["geometry"] = shapely.LineString(
                [(fx * width, fy * height), (sx * width, sy * height)]
            )
            G.edges[(first, second)]["length"] = G.edges[(first, second)][
                "geometry"
            ].length
            # Added to make it osmnx-compatible
            G.edges[(first, second)]["osmid"] = c
    # To make easier node labels
    G = nx.convert_node_labels_to_integers(G)
    count = len(G.edges)
    if diagonal:
        if cols == rows:
            for i in range(rows - 1):
                first = i * (rows + 1)
                second = (i + 1) * (rows + 1)
                G.add_edge(first, second)
                G.edges[first, second]["geometry"] = shapely.LineString(
                    [
                        utils.get_node_coord(
                            G,
                            first,
                        ),
                        utils.get_node_coord(G, second),
                    ]
                )
                G.edges[first, second]["length"] = G.edges[first, second][
                    "geometry"
                ].length
                G.edges[first, second]["osmid"] = count + i + 1
        else:
            warnings.warn(
                "Diagonal is only possible if the number of rows is the same as the number of colums for now."
            )
    # Added to make it osmnx-compatible
    G.graph["crs"] = "epsg:2154"
    G.graph["simplified"] = True
    if multidigraph:
        return nx.MultiDiGraph(G)
    return G


def create_radial_graph(radial=4, length=50, multidigraph=True):
    """Create a radial graph where roads are radiating from a center.

    Args:
        radial (int, optional): Number of roads arranged evenly around the center. Needs to be at least 2. Defaults to 4.
        length (int, optional): Lengths of the roads. Defaults to 50.
        multidigraph (bool, optional): If True, return Graph as MultiDiGraph. Graph is better for computations and ease of use, MultiDiGraph is more general and osmnx-compatible. Defaults to True.

    Raises:
        ValueError: Radial graph needs at least 3 radial roads to work.

    Returns:
        G (networkx.Graph or networkx.MultiDiGraph): Radial graph.
    """
    if radial < 3:
        raise ValueError("Radial graph needs at least 3 radial roads to work.")
    G = nx.Graph()
    G.add_node(0, x=0, y=0)
    # Nodes are evenly distributed on a circle
    for i in range(radial):
        G.add_node(
            i + 1,
            x=length * np.cos(i * 2 * np.pi / radial),
            y=length * np.sin(i * 2 * np.pi / radial),
        )
        pos = utils.get_node_coord(G, i + 1)
        G.add_edge(0, i + 1, geometry=shapely.LineString([(0, 0), pos]), osmid=i)
        G.edges[(0, i + 1)]["length"] = G.edges[(0, i + 1)]["geometry"].length
    # Added to make it osmnx-compatible
    G.graph["crs"] = "epsg:2154"
    G.graph["simplified"] = True
    if multidigraph:
        return nx.MultiDiGraph(G)
    return G


def create_concentric_graph(
    radial=8, zones=3, radius=30, center=True, multidigraph=True
):
    """Create a concentric graph, where nodes are on circular zones, connected to their nearest neighbors and to the next zone.

    Args:
        radial (int, optional): Number of nodes per zone. Nodes are evenly distributed on each circle. Needs to be at least 2. Defaults to 8.
        zones (int, optional): Number of zones. Needs to be at least 1. Defaults to 3.
        radius (int, optional): Radius between zones. Defaults to 30.
        center (bool, optional): If True, add a node at the center of the graph.
        multidigraph (bool, optional): If True, return Graph as MultiDiGraph. Graph is better for computations and ease of use, MultiDiGraph is more general and osmnx-compatible. Defaults to True.

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
    G = nx.Graph()
    count = 0
    if center:
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
    # If there is a center node, shift the ID of the nodes in the zones by 1
    startnum = 0
    # And shift the modulo parameter to link last node to first node of a zone
    mod = radial - 1
    if center:
        startnum += 1
        mod = 0
        for i in range(1, radial + 1):
            pos = [G.nodes[i]["x"], G.nodes[i]["y"]]
            G.add_edge(0, i, geometry=shapely.LineString([(0, 0), pos]), osmid=count)
            G.edges[(0, i)]["length"] = G.edges[(0, i)]["geometry"].length
            count += 1
    for i in range(zones):
        for j in range(startnum, startnum + radial):
            fn = i * radial + j
            sn = i * radial + j + 1
            # At last node of a zone, link to first node
            offset = 0
            if j % radial == mod:
                sn -= radial
                offset += 2 * np.pi
            fc = utils.get_node_coord(G, fn)
            sc = utils.get_node_coord(G, sn)
            # Add edge in both directions
            geom = create_curved_linestring(fc, sc, radius * (i + 1), offset=offset)
            G.add_edge(
                fn,
                sn,
                geometry=geom,
                osmid=count,
            )
            G.edges[(fn, sn)]["length"] = G.edges[(fn, sn)]["geometry"].length
            count += 1
            # Connect nodes to next zone if there is one
            if i < zones - 1:
                tn = (i + 1) * radial + j
                tc = utils.get_node_coord(G, tn)
                G.add_edge(fn, tn, geometry=shapely.LineString([fc, tc]), osmid=count)
                G.edges[(fn, tn)]["length"] = G.edges[(fn, tn)]["geometry"].length
                count += 1
    # Added to make it osmnx-compatible
    G.graph["crs"] = "epsg:2154"
    G.graph["simplified"] = True
    if multidigraph:
        return nx.MultiDiGraph(G)
    return G


# Simpler but less general function in the meantime
def create_curved_linestring(startpoint, endpoint, radius, offset=0):
    """Create a curved linestring between the two selected points.

    The curvature is given by the radius. The two points are supposed to be on a circle of the given radius. The offset allows to change the endpoint angle, to avoid issues of negative values and periodicity.

    Args:
        startpoint (array-like): coordinates of the first point
        endpoint (array-like): coordinates of the second point
        radius (int or float): radius of the circle on which the points are.
        offset (int, optional): Added angle in radian to the endpoint angle. Defaults to 0.

    Raises:
        ValueError: The radius needs to be at least as long as the Euclidean distance between the points.

    Returns:
        shapely.LineString : A geometric curved line between the two points.
    """
    if np.linalg.norm([startpoint, endpoint]) > 2 * radius:
        if math.isclose(np.linalg.norm([startpoint, endpoint]), radius):
            warnings.warn("Given radius is very close to the minimum value")
        else:
            raise ValueError(
                "Radius needs to be larger than the Euclidean distance between the points."
            )
    N = 100
    coords = []
    start_angle = utils.find_angle(startpoint)
    end_angle = utils.find_angle(endpoint) + offset
    angle_coords = np.linspace(start_angle, end_angle, num=N)
    for i in range(N):
        coords.append(
            [radius * np.cos(angle_coords[i]), radius * np.sin(angle_coords[i])]
        )
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
    if np.linalg.norm([startpoint, endpoint]) > 2 * radius:
        raise ValueError(
            "Radius needs to be larger than the Euclidean distance between the points."
        )
    coords = []
    for i in range(n_coord):
        coords.append([i, i])
    curve = shapely.LineString(coords)
    return curve


def remove_random_edges(
    G, N=1, keep_all_nodes=True, prevent_disconnect=True, is_directed=True
):
    """Remove random edges from a graph.

    Args:
        G (neworkx.MultiDiGraph): Graph on which we want to remove edges.
        N (int, optional): Number of edges we want to remove. Defaults to 1.
        prevent_disconnect (bool, optional): If True, will keep the network as connected as it was initially. Defaults to True.
        is_directed (bool, optional): Need to be True if the graph is directed. Defaults to True.

    Raises:
        ValueError: N is too large for the graph, pick a smaller N.

    Returns:
        G (networkx.MultiDiGraph): Graph with edges removed.
    """
    removed = 0
    # Avoid mutate original graph
    G = G.copy()
    # Make it undirected to avoid needing to remove two edges and to use the nx.number_connected_components function
    if is_directed is True:
        G = nx.MultiGraph(G)
    edgelist = list(G.edges)
    if keep_all_nodes:
        if prevent_disconnect:
            if len(G) - 1 > len(edgelist) - N:
                raise ValueError("N is too large for the graph, pick a smaller N")
        else:
            if len(G) // 2 + 1 > len(edgelist) - N:
                raise ValueError("N is too large for the graph, pick a smaller N")
    else:
        if N >= len(edgelist):
            raise ValueError("N is too large for the graph, pick a smaller N")
    # Test random edges and see if they are a valid choice
    while removed < N:
        valid = True
        tested = random.choice(edgelist)
        # Wrong choice if node of degree 1
        if keep_all_nodes:
            for node in tested[:2]:
                if G.degree(node) == 1:
                    valid = False
        # Create copy to test removal
        H = G.copy()
        H.remove_edge(*tested)
        # Wrong choice if creating an additional component
        if prevent_disconnect:
            if keep_all_nodes is False:
                for node in tested[:2]:
                    if H.degree(node) == 0:
                        H.remove_node(node)
            if nx.number_connected_components(H) > nx.number_connected_components(G):
                valid = False
        if valid:
            G = H
            edgelist = list(G.edges)
            removed += 1
    if is_directed:
        G = nx.MultiDiGraph(G)
    return G
