import numpy as np
import time
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from ambientfisher.utils import barycentric_weights_simplex, \
    inner_product, choose_base_vertex, l2_normalize, \
    intrinsic_gnomonic_from_triangle, getChordDistance, \
    embed_points_on_unit_sphere_from_chord_distances

class AmbientFisherInterpolator:
    def __init__(self,
                 anchor_alphas: np.array, 
                 anchor_pdfs: np.array, 
                 x_grid: np.ndarray):
        '''
        This class of AF is derived from work by Cranmer, Streets and Bandyopadhyay
        '''
        self.ambient_dim = len(anchor_alphas[0]) + 1
        self.sphere_dim = self.ambient_dim - 1

        self.anchor_alphas = anchor_alphas
        self.anchor_pdfs = list(anchor_pdfs)
        self.x = x_grid

        # Delauney triangulation
        self.triangulation = Delaunay(self.anchor_alphas)
        self.simplices = self.triangulation.simplices
        self.neighbors = self.triangulation.neighbors

        self.q = []
        for p in self.anchor_pdfs:
            q_alpha = np.sqrt(p(self.x))
            q_alpha = l2_normalize(q_alpha, self.x)
            self.q.append(q_alpha)
        
        self.q = np.array(self.q)

    def enclosing_simplex_indices(self, 
                     alpha: np.array):
        simplex_indices = int(self.triangulation.find_simplex(alpha))
        if simplex_indices==-1:
            raise Exception("Needs extrapolation")
        return simplex_indices

    def predict_intrinsic(self, alpha: np.ndarray):
        '''
        Predict interpolated PDF at alpha, evaluated on self.x
        Done intrinsically in Hilbert space - no Euclidean embedding needed
        '''
        simplex_indices = self.enclosing_simplex_indices(alpha)

        verts_init      = self.simplices[simplex_indices]
        anchors         = self.anchor_alphas[verts_init]

        barycentric_weights_init = barycentric_weights_simplex(alpha, anchors)

        simplex_qs = self.q[verts_init]

        inner_product_matrix=[]
        for f1 in simplex_qs:
            inner_product_arr = []
            for f2 in simplex_qs:
                inner_product_arr.append(inner_product(f1, f2, self.x))
            inner_product_matrix.append(inner_product_arr)
        inner_product_matrix=np.array(inner_product_matrix)

        base_vertex = choose_base_vertex(inner_product_matrix)

        simplex_qs_reordered = np.concatenate((simplex_qs[base_vertex:base_vertex+1], 
                                                simplex_qs[:base_vertex], 
                                                simplex_qs[base_vertex+1:]))

        barycentric_weights = np.concatenate((barycentric_weights_init[base_vertex:base_vertex+1],
                                              barycentric_weights_init[:base_vertex],
                                              barycentric_weights_init[base_vertex+1:]))
        
        pdf_alpha = intrinsic_gnomonic_from_triangle(simplex_qs_reordered, 
                                                     self.x, 
                                                     barycentric_weights)

        return pdf_alpha

    def predict_extrinsic(self, alpha: np.ndarray, followKyle = True):
        '''
        Predict interpolated PDF at alpha, evaluated on self.x
        Original AF formulation with Euclidean embedding step
        '''
        simplex_indices = self.enclosing_simplex_indices(alpha)

        verts_init      = self.simplices[simplex_indices]
        anchors         = self.anchor_alphas[verts_init]

        barycentric_weights_init          = barycentric_weights_simplex(alpha, anchors)

        simplex_qs = self.q[verts_init]

        inner_product_matrix=[]
        for f1 in simplex_qs:
            inner_product_arr = []
            for f2 in simplex_qs:
                inner_product_arr.append(inner_product(f1, f2, self.x))
            inner_product_matrix.append(inner_product_arr)
        inner_product_matrix=np.array(inner_product_matrix)

        # choose base vertex for stability, then reorder (base first)
        base_vertex = choose_base_vertex(inner_product_matrix)

        simplex_qs_reordered = np.concatenate((simplex_qs[base_vertex:base_vertex+1], 
                                                simplex_qs[:base_vertex], 
                                                simplex_qs[base_vertex+1:]))

        barycentric_weights = np.concatenate((barycentric_weights_init[base_vertex:base_vertex+1],
                                              barycentric_weights_init[:base_vertex],
                                              barycentric_weights_init[base_vertex+1:]))

        # Step 1: chord distances in Hilbert space
        chordDistMatrix=[]
        for f1 in simplex_qs_reordered:
            chord_distance_arr = []
            for f2 in simplex_qs_reordered:
                chord_distance_arr.append(getChordDistance(f1, f2, self.x))
            chordDistMatrix.append(chord_distance_arr)
        chordDistMatrix=np.array(chordDistMatrix)

        # Step 2: Euclidean embedding of the N points onto S^N-1 sphere 
        sphere_embedded_pts = embed_points_on_unit_sphere_from_chord_distances(chordDistMatrix) 

        # Step 3: gnomonic projection to plane z = -1 (south tangent plane) 
        z = sphere_embedded_pts[:, -1]
        gnomonic_projection = (-sphere_embedded_pts / z[:, None])     # scale so last coord is -1
        gnomonic_projection_vertices = gnomonic_projection[:, :self.sphere_dim].copy()

        baryCoords_ = self.triangulation.transform[simplex_indices, :self.sphere_dim].dot(np.transpose(alpha - self.triangulation.transform[simplex_indices, self.sphere_dim]))
        baryCoords = np.concatenate([baryCoords_, [1.0 - baryCoords_.sum()]])

        gnomonicTarget = np.zeros(self.sphere_dim)
        for i in range(self.ambient_dim):
            gnomonicTarget      += baryCoords[i] * gnomonic_projection_vertices[i]

        normedVertices          = gnomonic_projection_vertices.copy()
        for i in range(1, normedVertices.shape[0]):
            normedVertices[i] /= np.linalg.norm(normedVertices[i])
        normedSimplex           = Delaunay(normedVertices)
        normedBaryCoords_       = normedSimplex.transform[0, :self.sphere_dim].dot(np.transpose(gnomonicTarget - normedSimplex.transform[0, self.sphere_dim]))
        normedBaryCoords        = np.concatenate([normedBaryCoords_, [1.0 - normedBaryCoords_.sum()]])
        t                       = np.arctan(np.linalg.norm(gnomonicTarget))

        # Get tangents on the Hilbert plane
        tangents_to_vertices_on_plane = []
        for i in range(1, self.ambient_dim):
            ci = inner_product(simplex_qs_reordered[0], simplex_qs_reordered[i], self.x)
            tangents_to_vertices_on_plane.append(simplex_qs_reordered[i] - ci * simplex_qs_reordered[0])

        # Step 4: Find the tangent direction in Hilbert space plane
        u = []
        for i in range(self.ambient_dim-1):
            u.append(l2_normalize(tangents_to_vertices_on_plane[i], self.x))
        
        tangent_dir_target = np.zeros_like(simplex_qs_reordered[0])
        for i in range(self.sphere_dim):
            tangent_dir_target += u[i] * normedBaryCoords[i]

        tangent_dir_target = l2_normalize(tangent_dir_target, self.x)

        # Step 5: exp map on Hilbert sphere
        q_alpha         = l2_normalize(np.cos(t) * simplex_qs_reordered[0] + np.sin(t) * tangent_dir_target, self.x)
        pdf_alpha       = q_alpha ** 2

        return pdf_alpha