# Interpolation with Ambient Fisher and Gaussian Process Regression

Repository to explore and develop the Ambient Fisher and Gaussian Process Regression ideas for the goal of building a vertical interpolation algorithm to be used in Simulation-Based Inference and HistFactory-style analysis. 

## Ambient Fisher

Ambient Fisher algorithm heavily derived from work by Kyle Cranmer and Jeff Streets: [git link](https://github.com/cranmer/ambient-fisher/tree/483c23597ceef4111791f3e1cd0ca6566f7fdc15). The sphere embedding algorithm has been modified using the technique from [this work](https://github.com/anjishnu1991/interpolation/tree/master).

New work in this repository includes a modification that allows Gnomonic interpolation directly in the intrinsic Hilbert space without the Euclidean embedding step, as well as a new interpolation scheme using closed-form geodesics on the manifold of Poisson probability densities when working with binned Poisson fits.

### Intrinsic Gnomonic Interpolation

The original Ambient Fisher algorithm involves a step that embeds the anchor points to a low-dimensional Euclidean sphere, preserving chord distances from the original Hilber-space representation. This is followed by Delaunay triangulation and then Gnomonic projection onto the tangent plane at one of the vertices of the simplex containing the target parameter value.

In this work, we compare this original algorithm with a modified one, where the Gnomonic projection is performed onto the appropriate tangent plane, directly in the infinite-dimensional Hilbert space representation instead of the low-dimensional Euclidean space. This approach can be significantly faster as well as more robust since the Euclidean embedding step can become potentially unstable when large-dimensional parameter spaces are involved.

Key formulae that make this possible:

1. Gnomonic projection generalized to Hilbert-space representation:
   
  $$g\left(\sqrt{p(\alpha)}\right) = \frac{\sqrt{p(\alpha)} - \langle \sqrt{p(\alpha_0)}, \sqrt{p(\alpha)} \rangle \sqrt{p(\alpha_0)}}{\langle \sqrt{p(\alpha_0)}, \sqrt{p(\alpha)} \rangle}$$

2. Inverse Gnomonic Projection:

$$\phi = \frac{\sqrt{p(\alpha)}}{\langle \sqrt{p(\alpha)}, \sqrt{p(\alpha_0)} \rangle}$$

All the rest of the steps are more or less identical to the original algorithm.

### Ambient Fisher Interpolation in space of Poisson probabilities

The closed-form expression associated with two Poisson distributions computed with expected yields $\nu_1$ and $\nu_2$, which is given by

$$d_\text{FR}(p(x\mid \nu_1), p(x\mid \nu_1)) = 2 \cdot  \Big| \sqrt{\nu_1} - \sqrt{\nu_2} \Big|$$

The key observation is that the FR metric for a Poisson family becomes Euclidean in the $\sqrt{\nu}$ space (ignoring the factor of two, which is just a constant scaling of the geometry. This means the gnomonic projection step is unnecessary and we can already start Delauney triangualation after the square-root transformation of the yield.

## Gaussian Process Regression

# Library

Eventual goal to build a library around Ambient Fisher and GP regression for regression using arbitrary (alpha, p(x|alpha)) pairs - regardless of whether they come from binned approximations or NN estimated PDFs.


