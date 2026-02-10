import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def barycentric_weights_simplex(alpha, anchors):
    '''
    Return the barycentric weights w_i s.t. alpha = \sum_i w_i * a_i and \sum_i w_i = 1
    '''
    
    # Applying \sum_i w_i = 1, w_1 = 1 - \sum_i w_i 
    # Using this, we form a system of linear equations to solve for w_i
    num_anchors         = anchors.shape[0]
    M                   = np.column_stack([anchors[i] - anchors[0] for i in range(1, num_anchors)])
    y                   = alpha - anchors[0]
    weights_arr         = np.linalg.solve(M, y)
    sum_weights_arr     = np.sum(weights_arr)
    w1                  = 1.0 - sum_weights_arr
    weights_arr         = np.insert(weights_arr, 0, w1)
    return np.array(weights_arr)

def return_measure(xarray):
    xmin = np.amin(xarray)
    xmax = np.amax(xarray)
    num = len(xarray)

    return (xmax-xmin)/(num-1)

def inner_product(q1, q2, xarray):
    '''
    Inner product between two vectors 
    '''
    dx = return_measure(xarray)

    if callable(q1) and callable(q2):
        return np.sum(q1(xarray) * q2(xarray))*dx
    else:
        return np.sum(q1 * q2) * dx

def getChordDistance(q1, q2, xarray):
    return 2.*np.sin( np.arccos( inner_product(q1, q2, xarray) ) /2. )

def l2_normalize(q, xarray):
    norm_squared = inner_product(q, q, xarray)
    return q / np.sqrt(norm_squared)

def compute_l2_norm(q, xarray):
    norm_squared = inner_product(q, q, xarray)
    return norm_squared

def choose_base_vertex(inner_product_matrix):
    """
    Choose a base vertex for numerical stability.
    We pick the vertex that maximizes the *minimum* overlap to the other two.
    """

    m = inner_product_matrix.shape[0]
    scores = [np.min(inner_product_matrix[i, np.arange(m)!=i]) for i in range(m)]
    return int(np.argmax(scores))


def embed_points_on_unit_sphere_from_chord_distances(chord_dist: np.array):
    """
    Reconstruct m points on S^{d-1} given chord distances. 
    Cosmetically modified code of the original in https://github.com/anjishnu1991/interpolation/tree/master
    """
    m = chord_dist.shape[0]
    sphere_dim = m - 1

    pts = np.zeros((m, m), dtype=float)
    pts[0, -1] = -1.0  # south pole in last coordinate

    for k in range(1, m):
        M = np.empty((k, k), dtype=float)
        y = np.zeros(k, dtype=float)

        for j in range(k):
            d = chord_dist[j, k]
            y[j] = 1.0 - 0.5 * d ** 2      # dot product computation
            M[j, :] = pts[j, -k:]         # last k coordinates of previous point

        tail = np.linalg.solve(M, y)
        rem_sq = 1.0 - np.dot(tail, tail)
        first = -np.sqrt(rem_sq)        # choose southern hemisphere

        active = np.concatenate([[first], tail])      # length k+1
        pts[k, :] = np.concatenate([np.zeros(sphere_dim - k), active])

    return pts


def intrinsic_gnomonic_from_triangle(q_array, xarray, barycentric_weights, xobs = None):
    '''
    Given unit-norm sqrt densities q_i and barycentric weights for the target alpha, 
    with q_0 mapped to 0 in its own tangent place, the function returns interpolated pdf at alpha.

    This is the direct Hilber-space modification to the original AF.

    Gnomonic projection formula in the intrinsic space: 
    g_i = q_i / <q_0, q_i> - q_0

    And the interpolation is done using the standard AF formula:
    g = \sum_{i=1} w_i*g_i 
    q(alpha) = normalize(q_0 + g)
    p(alpha) = q(alpha)**2
    '''

    gnomonic_projections = []
    anchor_length = q_array.shape[0]

    for i in range(1, anchor_length):

        c_i0 = inner_product(q_array[0], q_array[i], xarray)
        if callable(q_array[0]):
            gnomonic_projections.append((q_array[i](xarray) / c_i0) - q_array[0](xarray))
        else:
            gnomonic_projections.append((q_array[i] / c_i0 ) - q_array[0])

    gnomonic_projections =  np.array(gnomonic_projections)

    g = np.dot(barycentric_weights[1:], gnomonic_projections)

    if callable(q_array[0]):
        q_alpha = q_array[0](xarray) + g
    else:
        q_alpha = q_array[0] + g

    q_alpha = l2_normalize(q_alpha, xarray)

    p_alpha = q_alpha**2

    if xobs is not None:
        if xobs.any() not in xarray.tolist():
            raise Exception("x not in domain")
        indices_xobs = [xarray.tolist().index(x) for x in xobs]
        p_alpha = p_alpha[indices_xobs].copy()
    
    return p_alpha
