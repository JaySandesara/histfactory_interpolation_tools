import numpy as np
import time
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import jax.numpy as jnp
import evermore as evm
from flax import nnx


from ambientfisher.utils import barycentric_weights_simplex, \
    inner_product, choose_base_vertex, l2_normalize, \
    intrinsic_gnomonic_from_triangle, getChordDistance, \
    embed_points_on_unit_sphere_from_chord_distances


class GaussianProcessInterpolator:
    def __init__(
        self,
        anchor_alphas: np.ndarray,
        anchor_ratios: np.ndarray,
        length_scale: float = 10.0,
        variance: float = 1.0,
        noise: float = 0.0
    ):
        """
        Gaussian Process interpolator for systematic variation ratios
        """
        self.anchor_alphas = np.asarray(anchor_alphas, dtype=float)
        self.anchor_ratios = np.asarray(anchor_ratios, dtype=float)

        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.noise = float(noise)

        # Precompute K(train, train) factorization-ready matrix
        K = self._sq_exp_kernel(self.anchor_alphas, self.anchor_alphas)
        diag = (self.noise**2) * np.eye(K.shape[0])
        self._K_train = K + diag

        self._K_train_inverse = np.linalg.inv(self._K_train)

    def _sq_exp_kernel(self, A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
        """
        Squared exponential (RBF) kernel.
        """
        A1 = np.asarray(A1, dtype=float)
        A2 = np.asarray(A2, dtype=float)

        # Pairwise squared distances: ||alpha - alpha'||^2
        diff = A1[:, None, :] - A2[None, :, :]
        sq_distance = np.sum(diff * diff, axis=-1)

        return self.variance * np.exp(-0.5 * sq_distance / (self.length_scale**2))

    def predict(self, alpha: Union[np.ndarray, list], prior_mean = 1.0) -> np.ndarray:
        """
        Predict ratio(s) at alpha using the GP posterior mean.
        """
        alpha = np.asarray(alpha, dtype=float)
        single_point = (alpha.ndim == 1)
        if single_point:
            alpha = alpha[None, :]

        K_eval_train = self._sq_exp_kernel(alpha, self.anchor_alphas)  # (n_eval, n_train)

        _anchor_minus_prior = self.anchor_ratios - prior_mean 

        # Posterior mean 
        posterior_mean = prior_mean + K_eval_train @ self._K_train_inverse @ _anchor_minus_prior

        return posterior_mean



class PoissonAFInterpolator:
    def __init__(self,
                 anchor_alphas: np.ndarray, 
                 anchor_nus: np.ndarray,
                 plot_simplex=False):
        '''
        Bin-by-bin interpolator in the manifold of Poisson probabilities
        '''
        self.anchor_alphas = anchor_alphas
        self.anchor_nus = list(anchor_nus)

        self.ambient_dim = len(anchor_alphas[0]) + 1
        self.sphere_dim = self.ambient_dim - 1

        # Delauney triangulation
        self.triangulation = Delaunay(self.anchor_alphas)
        self.simplices = self.triangulation.simplices
        self.neighbors = self.triangulation.neighbors

        if plot_simplex:
            if self.sphere_dim>2: 
                print(f"Cannot plot simplex in high-dimensional space")
            else:
                plt.triplot(anchor_alphas[:,0], anchor_alphas[:,1], self.simplices)
                plt.plot(anchor_alphas[:,0], anchor_alphas[:,1], 'o')
                plt.xlabel(r"$\alpha_1$")
                plt.ylabel(r"$\alpha_2$")
                plt.show()

        self.sqrtnus = []
        for nu in self.anchor_nus:
            sqrtnu = np.sqrt(nu)
            self.sqrtnus.append(sqrtnu)

        self.sqrtnus = np.array(self.sqrtnus)

    def enclosing_simplex_indices(self, 
                                alpha: np.array):
        simplex_indices = int(self.triangulation.find_simplex(alpha))
        if simplex_indices==-1:
            raise Exception("Needs extrapolation")
        return simplex_indices

    def predict(self, alpha: np.ndarray) -> float:
        '''
        Predict interpolated nu at alpha
        '''
        simplex_indices = self.enclosing_simplex_indices(alpha)

        verts_init = self.simplices[simplex_indices]
        anchors = self.anchor_alphas[verts_init]

        barycentric_weights = barycentric_weights_simplex(alpha, anchors)

        sqrtnu_anchors = self.sqrtnus[verts_init]

        sqrtnu_interpolated = np.dot(barycentric_weights, sqrtnu_anchors)

        nu_interpolated = sqrtnu_interpolated ** 2

        return nu_interpolated

class AmbientFisherInterpolator:
    def __init__(self,
                 anchor_alphas: np.array, 
                 anchor_pdfs: np.array, 
                 x_grid: np.ndarray,
                 plot_simplex = False):
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

        if plot_simplex:
            if self.sphere_dim>2: 
                print(f"Cannot plot simplex in high-dimensional space")
            else:
                plt.triplot(anchor_alphas[:,0], anchor_alphas[:,1], self.simplices)
                plt.plot(anchor_alphas[:,0], anchor_alphas[:,1], 'o')
                plt.xlabel(r"$\alpha_1$")
                plt.ylabel(r"$\alpha_2$")
                plt.show()

        self.q = []
        for p in self.anchor_pdfs:
            q_alpha = np.sqrt(p(self.x))
            q_alpha = l2_normalize(q_alpha, self.x)
            self.q.append(q_alpha)
        
        self.q = np.array(self.q)

    def enclosing_simplex_indices(self, 
                     alpha: np.array):
        simplex_indices = int(self.triangulation.find_simplex(alpha))
        return simplex_indices

    def predict_intrinsic(self, alpha: np.ndarray, xobs=None):
        '''
        Predict interpolated PDF at alpha, evaluated on self.x
        Done intrinsically in Hilbert space - no Euclidean embedding needed
        '''
        simplex_indices = self.enclosing_simplex_indices(alpha)

        if simplex_indices==-1:
            return None

        verts_init      = self.simplices[simplex_indices]
        anchors         = self.anchor_alphas[verts_init]

        barycentric_weights_init = barycentric_weights_simplex(alpha, anchors)
        # barycentric_weights_init_ = self.triangulation.transform[simplex_indices, :self.sphere_dim].dot(np.transpose(alpha - self.triangulation.transform[simplex_indices, self.sphere_dim]))
        # barycentric_weights_init = np.concatenate([barycentric_weights_init_, [1.0 - barycentric_weights_init_.sum()]])

        simplex_qs = self.q[verts_init]

        inner_product_matrix=[]
        for f1 in simplex_qs:
            inner_product_arr = []
            for f2 in simplex_qs:
                inner_product_arr.append(inner_product(f1, f2, self.x))
            inner_product_matrix.append(inner_product_arr)
        inner_product_matrix=np.array(inner_product_matrix)

        # base_vertex = choose_base_vertex(inner_product_matrix)
        base_vertex = 0

        simplex_qs_reordered = np.concatenate((simplex_qs[base_vertex:base_vertex+1], 
                                                simplex_qs[:base_vertex], 
                                                simplex_qs[base_vertex+1:]))

        barycentric_weights = np.concatenate((barycentric_weights_init[base_vertex:base_vertex+1],
                                              barycentric_weights_init[:base_vertex],
                                              barycentric_weights_init[base_vertex+1:]))
        
        pdf_alpha = intrinsic_gnomonic_from_triangle(simplex_qs_reordered, 
                                                     self.x, 
                                                     barycentric_weights,
                                                     xobs = xobs)

        return pdf_alpha

    def predict_extrinsic(self, alpha: np.ndarray, xobs = None):
        '''
        Predict interpolated PDF at alpha, evaluated on self.x
        Original AF formulation with Euclidean embedding step
        '''
        simplex_indices = self.enclosing_simplex_indices(alpha)

        if simplex_indices==-1:
            return None
        
        verts_init      = self.simplices[simplex_indices]
        anchors         = self.anchor_alphas[verts_init]

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

        # baryCoords_ = self.triangulation.transform[simplex_indices, :self.sphere_dim].dot(np.transpose(alpha - self.triangulation.transform[simplex_indices, self.sphere_dim]))
        # baryCoords = np.concatenate([baryCoords_, [1.0 - baryCoords_.sum()]])

        baryCoords = barycentric_weights_simplex(alpha, anchors)

        baryCoords_reordered = np.concatenate((baryCoords[base_vertex:base_vertex+1], 
                                                baryCoords[:base_vertex], 
                                                baryCoords[base_vertex+1:]))

        gnomonicTarget = np.zeros(self.sphere_dim)
        for i in range(self.ambient_dim):
            gnomonicTarget      += baryCoords_reordered[i] * gnomonic_projection_vertices[i]

        t                       = np.arctan(np.linalg.norm(gnomonicTarget))

        # Get tangents on the Hilbert plane
        tangents_to_vertices_on_plane = []
        for i in range(1, self.ambient_dim):
            ci = inner_product(simplex_qs_reordered[0], simplex_qs_reordered[i], self.x)
            if not callable(simplex_qs_reordered[0]):
                tangents_to_vertices_on_plane.append(simplex_qs_reordered[i] - ci * simplex_qs_reordered[0])
            else:
                tangents_to_vertices_on_plane.append(simplex_qs_reordered[i](self.x) - ci * simplex_qs_reordered[0](self.x))

        # Step 4: Find the tangent direction in Hilbert space plane
        u = []
        for i in range(self.ambient_dim-1):
            u.append(l2_normalize(tangents_to_vertices_on_plane[i], self.x))
        
        normedVertices          = gnomonic_projection_vertices.copy()
        for i in range(1, normedVertices.shape[0]):
            normedVertices[i] /= np.linalg.norm(normedVertices[i])

        # # This Delaunay business seems to break the ordering from original Barycentric coords calculation
        # normedSimplex           = Delaunay(normedVertices)
        # normedBaryCoords_       = normedSimplex.transform[0, :self.sphere_dim].dot(np.transpose(gnomonicTarget - normedSimplex.transform[0, self.sphere_dim]))
        # normedBaryCoords        = np.concatenate([normedBaryCoords_, [1.0 - normedBaryCoords_.sum()]])
        normedBaryCoords = barycentric_weights_simplex(gnomonicTarget, normedVertices)

        tangent_dir_target = np.zeros_like(simplex_qs_reordered[0])
        for i in range(self.sphere_dim):
            tangent_dir_target += u[i] * normedBaryCoords[i+1]

        tangent_dir_target = l2_normalize(tangent_dir_target, self.x)

        # Step 5: exp map on Hilbert sphere
        q_alpha         = l2_normalize(np.cos(t) * simplex_qs_reordered[0] + np.sin(t) * tangent_dir_target, self.x)
        pdf_alpha       = q_alpha ** 2

        if xobs is not None:
            if xobs.any() not in self.x.tolist():
                raise Exception("x not in domain")
            indices_xobs = [self.x.tolist().index(x) for x in xobs]
            pdf_alpha = pdf_alpha[indices_xobs].copy()

        return pdf_alpha