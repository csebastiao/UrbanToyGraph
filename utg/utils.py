"""Useful functions."""

import math

import numpy as np


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
