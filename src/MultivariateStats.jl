module MultivariateStats_

export RidgeRegressor, PCA, KernelPCA, ICA, LDA

import MLJBase
import MLJBase: @mlj_model, metadata_model, metadata_pkg
import StatsBase:proportions
using  CovarianceEstimation
using Distances
using LinearAlgebra
using ScientificTypes
using Tables

import ..MultivariateStats # lazy loading

const MS = MultivariateStats

struct LinearFitresult{F} <: MLJBase.MLJType
    coefficients::Vector{F}
    bias::F
end

const RIDGE_DESCR = "Ridge regressor with regularization parameter lambda."
const PCA_DESCR = "Principal component analysis."
const KPCA_DESCR = "Kernel principal component analysis."
const ICA_DESCR = "Independent component analysis."
const LDA_DESCR = "Linear discriminant analysis, learns a projection matrix `W` that projects the feature matrix `Xtrain` unto a lower dimensional space of dimension `out_dim` such that the between-class variance is maximized relative to the  within-class variance. Classification is done by applying Bayes' rule to the transformed test matrix `W'Xtest`."

####
#### RIDGE
####

"""
RidgeRegressor(; lambda=1.0)

$RIDGE_DESCR

## Parameters

* `lambda=1.0`: non-negative parameter for the regularization strength.
"""
@mlj_model mutable struct RidgeRegressor <: MLJBase.Deterministic
    lambda::Real = 1.0::(_ ≥ 0)
end

function MLJBase.fit(model::RidgeRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    features = Tables.schema(X).names

    weights      = MS.ridge(Xmatrix, y, model.lambda)
    coefficients = weights[1:end-1]
    bias         = weights[end]

    fitresult = LinearFitresult(coefficients, bias)
    report    = NamedTuple()
    cache     = nothing

    return fitresult, cache, report
end

MLJBase.fitted_params(::RidgeRegressor, fitresult) =
    (coefficients=fitresult.coefficients, bias=fitresult.bias)

function MLJBase.predict(::RidgeRegressor, fitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return Xmatrix * fitresult.coefficients .+ fitresult.bias
end

####
#### PCA
####

const PCAFitResultType = MS.PCA

"""
PCA(; maxoutdim=nothing, method=:auto, pratio=0.99, mean=nothing)

$PCA_DESCR

## Parameters

* `maxoutdim=nothing`: maximum number of output dimensions, unconstrained if nothing.
* `method=:auto`: method to use to solve the problem, one of `:auto`, `:cov` or `:svd`
* `pratio=0.99`: ratio of variance preserved
* `mean=nothing`: if set to nothing centering will be computed and applied, if set to `0` no centering (assumed pre-centered), if a vector is passed, the centering is done with that vector.
"""
@mlj_model mutable struct PCA <: MLJBase.Unsupervised
    maxoutdim::Int  = typemax(Int)::(_ === nothing || _ ≥ 1)
    method::Symbol  = :auto::(_ in (:auto, :cov, :svd))
    pratio::Float64 = 0.99::(0.0 < _ ≤ 1.0)
    mean::Union{Nothing, Real, Vector{Float64}} = nothing::(_ === nothing || (_ isa Real && iszero(_)) || true)
end

function MLJBase.fit(model::PCA, verbosity::Int, X)
    Xarray = MLJBase.matrix(X)
    mindim = minimum(size(Xarray))

    maxoutdim = ifelse(model.maxoutdim == typemax(Int), mindim, model.maxoutdim)

    # NOTE: copy/transpose
    fitresult = MS.fit(MS.PCA, permutedims(Xarray)
                     ; method=model.method
                     , pratio=model.pratio
                     , maxoutdim=maxoutdim
                     , mean=model.mean)

    cache = nothing
    report = (indim=MS.indim(fitresult),
              outdim=MS.outdim(fitresult),
              mean=MS.mean(fitresult),
              principalvars=MS.principalvars(fitresult),
              tprincipalvar=MS.tprincipalvar(fitresult),
              tresidualvar=MS.tresidualvar(fitresult),
              tvar=MS.tvar(fitresult))

    return fitresult, cache, report
end

MLJBase.fitted_params(::PCA, fitresult) = (projection=fitresult,)


function MLJBase.transform(::PCA, fitresult::PCAFitResultType, X)
    Xarray = MLJBase.matrix(X)
    # X is n x d, need to transpose and copy twice...
    return MLJBase.table(
                permutedims(MS.transform(fitresult, permutedims(Xarray))),
                prototype=X)
end

####
#### KernelPCA
####

const KernelPCAFitResultType = MS.KernelPCA

default_kernel = (x, y) -> x'y

"""
KernelPCA(; maxoutdim=nothing, kernel=(x,y)->x'y, solver=:auto, pratio=0.99, mean=nothing)

$KPCA_DESCR

## Parameters

* `maxoutdim=nothing`: maximum number of output dimensions, unconstrained if nothing.
* `kernel=nothing`: kernel function of 2 vector arguments x and y, returns a scalar value, (x,y)->x'y if nothing
* `solver=:auto`: solver to use for the eigenvalues, one of `:eig` (default), `:eigs`
* `inverse=false`: perform calculation for inverse transform
* `beta=1.0`: strength of the ridge regression that learns the inverse transform when inverse is true
* `tol=0.0`: Convergence tolerance for eigs solver
* `maxiter=300`: maximum number of iterations for eigs solver
"""
@mlj_model mutable struct KernelPCA <: MLJBase.Unsupervised
    maxoutdim::Int   = typemax(Int)::(_ ≥ 1)
    kernel::Function = default_kernel
    solver::Symbol   = :eig::(_ in (:eig, :eigs))
    inverse::Bool    = false
    beta::Real       = 1.0::(_ ≥ 0.0)
    tol::Real        = 1e-6::(_ ≥ 0.0)
    maxiter::Int     = 300::(_ ≥ 1)
end

function MLJBase.fit(model::KernelPCA
                   , verbosity::Int
                   , X)

    Xarray = MLJBase.matrix(X)
    mindim = minimum(size(Xarray))

    maxoutdim = ifelse(model.maxoutdim == typemax(Int), mindim, model.maxoutdim)

    fitresult = MS.fit(MS.KernelPCA, permutedims(Xarray)
                     ; kernel=model.kernel
                     , maxoutdim=maxoutdim
                     , solver=model.solver
                     , inverse=model.inverse
                     , β=model.beta
                     , tol=model.tol
                     , maxiter=model.maxiter)

    cache = nothing
    report = (indim=MS.indim(fitresult)
            , outdim=MS.outdim(fitresult)
            , projection=MS.projection(fitresult)
            , principalvars=MS.principalvars(fitresult))

    return fitresult, cache, report
end

MLJBase.fitted_params(::KernelPCA, fitresult) = (projection=fitresult,)

function MLJBase.transform(::KernelPCA, fitresult::KernelPCAFitResultType, X)
    # X is n x d, need to transpose and copy twice...
    Xarray = MLJBase.matrix(X)
    return MLJBase.table(
                permutedims(MS.transform(fitresult, permutedims(Xarray))),
                prototype=X)
end

####
#### ICA
####

const ICAFitResultType = MS.ICA

"""
ICA(; k=0, alg=:fastica, fun=:tanh, do_whiten=true, maxiter=100,
      tol=1e-6, mean=nothing, winit=zeros(0,0))

$ICA_DESCR

## Parameters

* `k=0`: number of independent components to recover, set automatically if `0`
* `alg=:fastica`: algorithm to use (only `:fastica` is supported at the moment)
* `fun=:tanh`: approximate neg-entropy functor, via the function `MultivariateStats.icagfun`, one of `:tanh` and `:gaus`
* `do_whiten=true`: whether to perform pre-whitening
* `maxiter=100`: maximum number of iterations
* `tol=1e-6`:  convergence tolerance for change in matrix W
* `mean=nothing`: mean to use, if nothing centering is computed and applied, if zero, no centering, a vector of means can be passed
* `winit=zeros(0,0)`: initial guess for matrix `W` either an empty matrix (random init), a matrix of size `k × k` (if `do_whiten` is true), a matrix of size `m × k` otherwise
"""
@mlj_model mutable struct ICA <: MLJBase.Unsupervised
    k::Int          = 0::(_ ≥ 0)
    alg::Symbol     = :fastica::(_ in (:fastica,))
    fun::Symbol     = :tanh::(_ in (:tanh, :gaus))
    do_whiten::Bool = true
    maxiter::Int    = 100
    tol::Real       = 1e-6::(_ ≥ 0.0)
    winit::Union{Nothing,Matrix{<:Real}} = nothing
    mean::Union{Nothing, Real, Vector{Float64}} = nothing::(_ === nothing || (_ isa Real && iszero(_)) || true)
end

function MLJBase.fit(model::ICA, verbosity::Int, X)

    Xarray = MLJBase.matrix(X)
    m, n = size(Xarray)

    k = (model.k <= min(m, n)) ? min(m, n) : model.k

    fitresult = MS.fit(MS.ICA, permutedims(Xarray), k
                     ; alg=model.alg
                     , fun=MS.icagfun(model.fun)
                     , do_whiten=model.do_whiten
                     , maxiter=model.maxiter
                     , tol=model.tol
                     , mean=model.mean
                     , winit=ifelse(model.winit === nothing, zeros(0,0), model.winit))

    cache = nothing
    report = (indim=MS.indim(fitresult),
              outdim=MS.outdim(fitresult),
              mean=MS.mean(fitresult))

    return fitresult, cache, report
end

MLJBase.fitted_params(::ICA, fitresult) = (projection=fitresult,)

function MLJBase.transform(model::ICA
                         , fitresult::ICAFitResultType
                         , X)

    Xarray = MLJBase.matrix(X)
    # X is n x d, need to transpose and copy twice...
    return MLJBase.table(
                permutedims(MS.transform(fitresult, permutedims(Xarray))),
                prototype=X)
end

####
#### MulticlassLDA
####

"""
LDA(; method=:gevd, shrinkage=:None , out_dim=1, regcoef=1e-6)

$LDA_DESCR

## Parameters

* `method`    : The choice of methods, one of
    * `:gevd`   : based on generalized eigenvalue decomposition
    * `:whiten` : first derive a whitening transform from Sw and then solve the problem based on
                  eigenvalue decomposition of the whiten Sb
* `shrinkage` : Choice of shrinkage parameter for Linear shrinkage covariance estimator using a
                Diagonal target matrix, one of
    * `:None` : No shrinkage. use `SimpleCovariance` estimator instead
    * `:lw`  : select optimal shrinkage using the Ledoit-Wolf formula.
* `out_dim`   : The output dimension, i.e dimension of the transformed space, automatically set if 0
* `regcoef`   : The regularization coefficient. A positive value regcoef * eigmax(Sw) is added to
                the diagonal of Sw to improve numerical stability.

See also the [package documentation](https://multivariatestatsjl.readthedocs.io/en/latest/lda.html).
"""
@mlj_model mutable struct LDA <: MLJBase.Probabilistic
    method::Symbol                = :gevd::(_ in (:gevd, :whiten))
    shrinkage::Union{Symbol,Real} = :none::(_ in (:none,) || (_ isa Real && 0 ≤ _ ≤ 1))
    out_dim::Int                  = 0::(_ ≥ 0)
    regcoef::Real                 = 1e-6::(_ ≥ 0)
end

function MLJBase.fit(model::LDA, verbosity::Int, X, y)
    class_list = MLJBase.classes(y[1]) #class list containing unique entries in y
    nclasses   = length(class_list)

    # NOTE: copy/transpose
    Xmatrix = MLJBase.matrix(X, transpose=true) # now p x n
    dims    = size(Xmatrix, 1)

    # convert y into ints and estimate prior probabilities
    yplain = MLJBase.int(y)
    πk     = proportions(yplain)

    def_outdim = min(dims, nclasses - 1)
    out_dim    = ifelse(model.out_dim == 0, def_outdim, model.out_dim)
    out_dim ≤ def_outdim || throw(ArgumentError("`out_dim` must not be larger than `min(p, nc-1)` where `p` is the dimension of `X` and `nc` is the number of classes."))

    covestimator = model.shrinkage == :none ?
                     MS.SimpleCovariance() :
                     LinearShrinkage(target=DiagonalCommonVariance(), shrinkage=model.shrinkage)

    cache  = nothing
    report = NamedTuple{}()

    core_fitresult = MS.fit(MS.MulticlassLDA, nclasses, Xmatrix, Int.(yplain);
                            method=model.method,
                            outdim=out_dim,
                            regcoef=model.regcoef,
                            covestimator_within=covestimator,
                            covestimator_between=covestimator)

    fitresult = (class_list, core_fitresult, πk, dims)
    return fitresult, cache, report
end

function MLJBase.fitted_params(::LDA, (class_list, core_fitresult, πk, dims))
    ## Note The projection matrix that projects data unto
    ## a lower dimensional subspace and whitens the lower dimensional data .
    return (class_means = MS.classmeans(core_fitresult),
            projection_matrix = MS.projection(core_fitresult),
            prior_probabilities = πk)
end

function MLJBase.predict(model::LDA , (class_list, core_fitresult, πk, dims), Xnew)
    nclasses = length(class_list)

    ##Transpose the Xnew matrix and reduce its dimension
    Xmatrix = MS.transform(core_fitresult, MLJBase.matrix(Xnew, transpose=true)) # p x n

    ## Estimated the probabilities of each column in Xmatrix belonging to each class in
    ## class_list storing the results in a probability_matrix
    pmeans = MS.transform(core_fitresult, MS.classmeans(core_fitresult))

    probability_matrix   = pairwise(SqEuclidean(), pmeans, Xmatrix, dims=2)
    probability_matrix  .= πk .* exp.(-0.5 .* probability_matrix .* (dims - nclasses) )
    probability_matrix ./= sum(probability_matrix, dims=1)

    return [MLJBase.UnivariateFinite(class_list, probability_matrix[:, j]) for j in 1:size(Xmatrix, 2)]
end


####
#### METADATA
####

metadata_pkg.((RidgeRegressor, PCA, KernelPCA, ICA, LDA),
              name="MultivariateStats",
              uuid="6f286f6a-111f-5878-ab1e-185364afe411",
              url="https://github.com/JuliaStats/MultivariateStats.jl",
              license="MIT",
              julia=true,
              is_wrapper=false)

metadata_model(RidgeRegressor,
               input=MLJBase.Table(MLJBase.Continuous),
               target=AbstractVector{MLJBase.Continuous},
               weights=false,
               descr=RIDGE_DESCR)

metadata_model(PCA,
               input=MLJBase.Table(MLJBase.Continuous),
               target=MLJBase.Table(MLJBase.Continuous),
               weights=false,
               descr=PCA_DESCR)

metadata_model(KernelPCA,
               input=MLJBase.Table(MLJBase.Continuous),
               target=MLJBase.Table(MLJBase.Continuous),
               weights=false,
               descr=KPCA_DESCR)

metadata_model(ICA,
               input=MLJBase.Table(MLJBase.Continuous),
               target=MLJBase.Table(MLJBase.Continuous),
               weights=false,
               descr=ICA_DESCR)

metadata_model(LDA,
               input=MLJBase.Table(MLJBase.Continuous),
               target=AbstractVector{<:MLJBase.Finite},
               weights=false,
               descr=LDA_DESCR)

end # of module+
