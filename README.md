# Interpolation with Ambient Fisher and Gaussian Process Regression

Repository to explore and develop the Ambient Fisher and Gaussian Process Regression ideas for the goal of building a vertical interpolation algorithm to be used in Simulation-Based Inference and HistFactory-style analysis. 

## Ambient Fisher

Ambient Fisher algorithm heavily derived from work by Kyle Cranmer and Jeff Streets: [git link](https://github.com/cranmer/ambient-fisher/tree/483c23597ceef4111791f3e1cd0ca6566f7fdc15). The sphere embedding algorithm has been modified using the technique from [this work](https://github.com/anjishnu1991/interpolation/tree/master).

New work in this repository includes a modification that allows Gnomonic interpolation directly in the intrinsic Hilbert space without the Euclidean embedding step, as well as a new interpolation scheme using closed-form geodesics on the manifold of Poisson probability densities when working with binned Poisson fits.

## Gaussian Process Regression

# Library

Eventual goal to build a library around Ambient Fisher and GP regression for regression using arbitrary (alpha, p(x|alpha)) pairs - regardless of whether they come from binned approximations or NN estimated PDFs.


