"""Script to create networkx graphs and save them."""

import warnings
import networkx as nx
import shapely

def create_grid(m=3, n=3, width=50, height=None):
    G = nx.grid_2d_graph(m, n, create_using=nx.MultiDiGraph)
    if width <= 0:
        raise ValueError("Width needs to be positive.")
    if height is None:
        height = width
    else:
        warnings.warn("Height value selected, if different than width, will create rectangles instead of squares.")
    count = 0
    for node in G.nodes:
        x, y = node
        G.nodes[node]['x'] = x*width
        G.nodes[node]['y'] = y*height
        for edge in list(G.in_edges(node)) + list(G.out_edges(node)):
            count += 1
            first, second = edge
            fx, fy = first
            sx, sy = second
            G.edges[(first, second, 0)]["geometry"] = shapely.LineString([(fx*width, fy*height), (sx*width, sy*height)])
            G.edges[(first, second, 0)]["osmid"] = count
    G = nx.convert_node_labels_to_integers(G)
    G.graph["crs"] = "epsg:4326"
    G.graph["simplified"] = True
    return G

def create_concentric(radial=8, zones=3, radius=30):
    if radial < 2:
        raise ValueError("Concentric graph needs at least 2 radial to work.")
    G = nx.MultiDiGraph()
    return G