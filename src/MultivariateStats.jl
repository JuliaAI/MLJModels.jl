module MultivariateStats_

export RidgeRegressor, PCA, KernelPCA, ICA, LDA, BayesianLDA

import MLJBase
import MLJBase: @mlj_model, metadata_model, metadata_pkg
import StatsBase: proportions, CovarianceEstimator
using Distances, LinearAlgebra
using Tables, ScientificTypes

import MultivariateStats

const MS = MultivariateStats

struct LinearFitresult{F} <: MLJBase.MLJType
    coefficients::Vector{F}
    intercept::F
end

const RIDGE_DESCR = "Ridge regressor with regularization parameter lambda. Learns a linear regression with a penalty on the l2 norm of the coefficients."
const PCA_DESCR = "Principal component analysis. Learns a linear transformation to project the data  on a lower dimensional space while preserving most of the initial variance."
const KPCA_DESCR = "Kernel principal component analysis."
const ICA_DESCR = "Independent component analysis."
const LDA_DESCR = "Multiclass linear discriminant analysis. The algorithm learns a projection matrix `W` that projects the feature matrix `Xtrain` onto a lower dimensional space of dimension `out_dim` such that the between-class variance in the transformed space is maximized relative to the within-class variance."
const BayesianLDA_DESCR= "Bayesian Multiclass linear discriminant analysis. The algorithm learns a projection matrix `W` that projects the feature matrix `Xtrain` onto a lower dimensional space of dimension `out_dim` such that the between-class variance in the transformed space is maximized relative to the within-class variance and classifies using Bayes rule."

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
    Xmatrix   = MLJBase.matrix(X)
    features  = Tables.schema(X).names
    θ         = MS.ridge(Xmatrix, y, model.lambda)
    coefs     = θ[1:end-1]
    intercept = θ[end]

    fitresult = LinearFitresult(coefs, intercept)
    report    = NamedTuple()
    cache     = nothing

    return fitresult, cache, report
end

MLJBase.fitted_params(::RidgeRegressor, fr) =
    (coefficients=fr.coefficients, intercept=fr.intercept)

function MLJBase.predict(::RidgeRegressor, fr, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return Xmatrix * fr.coefficients .+ fr.intercept
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
    maxoutdim::Union{Nothing,Int} = nothing::(_ === nothing || _ ≥ 1)
    method::Symbol  = :auto::(_ in (:auto, :cov, :svd))
    pratio::Float64 = 0.99::(0.0 < _ ≤ 1.0)
    mean::Union{Nothing, Real, Vector{Float64}} = nothing::(_ === nothing || (_ isa Real && iszero(_)) || true)
end

function MLJBase.fit(model::PCA, verbosity::Int, X)
    Xarray = MLJBase.matrix(X)
    mindim = minimum(size(Xarray))

    maxoutdim = model.maxoutdim === nothing ? mindim : model.maxoutdim

    # NOTE: copy/transpose
    fitresult = MS.fit(MS.PCA, permutedims(Xarray);
                       method=model.method,
                       pratio=model.pratio,
                       maxoutdim=maxoutdim,
                       mean=model.mean)

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

MLJBase.fitted_params(::PCA, fr) = (projection=fr,)


function MLJBase.transform(::PCA, fr::PCAFitResultType, X)
    # X is n x d, need to transpose and copy twice...
    Xarray = MLJBase.matrix(X)
    Xnew   = permutedims(MS.transform(fr, permutedims(Xarray)))
    return MLJBase.table(Xnew, prototype=X)
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
    maxoutdim::Union{Nothing,Int} = nothing::(_ === nothing || _ ≥ 1)
    kernel::Function = default_kernel
    solver::Symbol   = :eig::(_ in (:eig, :eigs))
    inverse::Bool    = false
    beta::Real       = 1.0::(_ ≥ 0.0)
    tol::Real        = 1e-6::(_ ≥ 0.0)
    maxiter::Int     = 300::(_ ≥ 1)
end

function MLJBase.fit(model::KernelPCA, verbosity::Int, X)
    Xarray = MLJBase.matrix(X)
    mindim = minimum(size(Xarray))
    # default max out dim if not given
    maxoutdim = model.maxoutdim === nothing ? mindim : model.maxoutdim

    fitresult = MS.fit(MS.KernelPCA, permutedims(Xarray);
                       kernel=model.kernel,
                       maxoutdim=maxoutdim,
                       solver=model.solver,
                       inverse=model.inverse,
                       β=model.beta,
                       tol=model.tol,
                       maxiter=model.maxiter)

    cache  = nothing
    report = (indim=MS.indim(fitresult),
              outdim=MS.outdim(fitresult),
              projection=MS.projection(fitresult),
              principalvars=MS.principalvars(fitresult))

    return fitresult, cache, report
end

MLJBase.fitted_params(::KernelPCA, fr) = (projection=fr,)

function MLJBase.transform(::KernelPCA, fr::KernelPCAFitResultType, X)
    # X is n x d, need to transpose and copy twice...
    Xarray = MLJBase.matrix(X)
    Xnew   = permutedims(MS.transform(fr, permutedims(Xarray)))
    return MLJBase.table(Xnew, prototype=X)
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
    n, p   = size(Xarray)

    m = min(n, p)
    k = ifelse(model.k ≤ m, model.k, m)

    fitresult = MS.fit(MS.ICA, permutedims(Xarray), k;
                       alg=model.alg,
                       fun=MS.icagfun(model.fun),
                       do_whiten=model.do_whiten,
                       maxiter=model.maxiter,
                       tol=model.tol,
                       mean=model.mean,
                       winit=ifelse(model.winit === nothing, zeros(0,0), model.winit))

    cache = nothing
    report = (indim=MS.indim(fitresult),
              outdim=MS.outdim(fitresult),
              mean=MS.mean(fitresult))

    return fitresult, cache, report
end

MLJBase.fitted_params(::ICA, fr) = (projection=fr,)

function MLJBase.transform(::ICA, fr::ICAFitResultType, X)
    # X is n x d, need to transpose and copy twice...
    Xarray = MLJBase.matrix(X)
    Xnew   = permutedims(MS.transform(fr, permutedims(Xarray)))
    return MLJBase.table(Xnew, prototype=X)
end

####
#### MulticlassLDA
####

"""
LDA(; kwargs...)

$LDA_DESCR

## Parameters

* `method=:gev`: choice of solver, one of `:gevd` or `:whiten` methods
* `cov_w=SimpleCovariance()`: an estimator for the within-class covariance, by default set to the standard `MultivariateStats.CovarianceEstimator` but could be set to any robust estimator from `CovarianceEstimation.jl`.
* `cov_b=SimpleCovariance()`: same as `cov_w` but for the between-class covariance.
* `out_dim`: the output dimension, i.e dimension of the transformed space, automatically set if 0 is given (default).
* `regcoef`: regularization coefficient. A positive value `regcoef * eigmax(Sw)` where `Sw` is the within-class covariance estimator, is added to the diagonal of Sw to improve numerical stability. This can be useful if using the standard covariance estimator.
* `dist=SqEuclidean`: the distance metric to use when performing classification (to compare the distance between a new point and centroids in the transformed space), an alternative choice can be the `CosineDist`.

See also the [package documentation](https://multivariatestatsjl.readthedocs.io/en/latest/lda.html).
For more information about the algorithm, see the paper by Li, Zhu and Ogihara, [Using Discriminant Analysis for Multi-class Classification: An Experimental Investigation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.7068&rep=rep1&type=pdf).
"""
@mlj_model mutable struct LDA <: MLJBase.Probabilistic
    method::Symbol   = :gevd::(_ in (:gevd, :whiten))
    cov_w::CovarianceEstimator = MS.SimpleCovariance()
    cov_b::CovarianceEstimator = MS.SimpleCovariance()
    out_dim::Int     = 0::(_ ≥ 0)
    regcoef::Real    = 0.0::(_ ≥ 0)
    dist::SemiMetric = SqEuclidean()
end

function MLJBase.fit(model::LDA, ::Int, X, y)
    class_list = MLJBase.classes(y[1]) #class list containing unique entries in y
    nclasses   = length(class_list)

    # NOTE: copy/transpose
    Xm_t   = MLJBase.matrix(X, transpose=true) # now p x n matrix
    yplain = MLJBase.int(y) # vector of n ints in {1,..., nclasses}
    p      = size(Xm_t, 1)

    # check output dimension default is min(p, nc-1)
    def_outdim = min(p, nclasses - 1)
    # if unset (0) use the default; otherwise try to use the provided one
    out_dim = ifelse(model.out_dim == 0, def_outdim, model.out_dim)
    # check if the given one is sensible
    out_dim ≤ def_outdim || throw(ArgumentError("`out_dim` must not be larger than `min(p, nc-1)` where `p` is the dimension of `X` and `nc` is the number of classes."))

    core_res = MS.fit(MS.MulticlassLDA, nclasses, Xm_t, Int.(yplain);
                      method=model.method,
                      outdim=out_dim,
                      regcoef=model.regcoef,
                      covestimator_within=model.cov_w,
                      covestimator_between=model.cov_b)

    cache     = nothing
    report    = NamedTuple{}()
    fitresult = (core_res, class_list)

    return fitresult, cache, report
end

function MLJBase.fitted_params(::LDA, (core_res, class_list))
    return (class_means       = MS.classmeans(core_res),
            projection_matrix = MS.projection(core_res))
end

function MLJBase.predict(m::LDA, (core_res, class_list), Xnew)
    # projection of Xnew XWt is n x o  where o = number of out dims
    XWt = MLJBase.matrix(Xnew) * core_res.proj
    # centroids in the transformed space, nc x o
    centroids = permutedims(core_res.pmeans)

    # compute the distances in the transformed space between pairs of rows
    # the probability matrix is `n x nc` and normalised accross rows
    P = pairwise(m.dist, XWt, centroids, dims=1)
    # apply a softmax transformation
    P .-= maximum(P, dims=2)
    P  .= exp.(-P)
    P ./= sum(P, dims=2)

    return [MLJBase.UnivariateFinite(class_list, P[j, :]) for j in 1:size(P, 1)]
end

####
#### BayesianLDA
####

"""
BayesianLDA(; kwargs...)

$BayesianLDA_DESCR

## Parameters

* `method=:gev`: choice of solver, one of `:gevd` or `:whiten` methods
* `cov_w=SimpleCovariance()`: an estimator for the within-class covariance, by default set to the standard `MultivariateStats.CovarianceEstimator` but could be set to any robust estimator from `CovarianceEstimation.jl`.
* `cov_b=SimpleCovariance()`: same as `cov_w` but for the between-class covariance.
* `out_dim`: the output dimension, i.e dimension of the transformed space, automatically set if 0 is given (default).
* `regcoef`: regularization coefficient. A positive value `regcoef * eigmax(Sw)` where `Sw` is the within-class covariance estimator, is added to the diagonal of Sw to improve numerical stability. This can be useful if using the standard covariance estimator.
* `priors=nothing`: if `priors = nothing` estimates the prior probabilities from the data else it uses the user specified `Univariate` prior probabilities.
See also the [package documentation](https://multivariatestatsjl.readthedocs.io/en/latest/lda.html).
For more information about the algorithm, see the paper by Li, Zhu and Ogihara, [Using Discriminant Analysis for Multi-class Classification: An Experimental Investigation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.7068&rep=rep1&type=pdf).
"""
@mlj_model mutable struct BayesianLDA <: MLJBase.Probabilistic
    method::Symbol   = :gevd::(_ in (:gevd, :whiten))
    cov_w::CovarianceEstimator = MS.SimpleCovariance()
    cov_b::CovarianceEstimator = MS.SimpleCovariance()
    out_dim::Int     = 0::(_ ≥ 0)
    regcoef::Real    = 0.0::(_ ≥ 0)
    priors::Union{Nothing, MLJBase.UnivariateFinite} = nothing
end

function MLJBase.fit(model::BayesianLDA, ::Int, X, y)
    class_list = MLJBase.classes(y[1]) #class list containing unique entries in y
    nclasses   = length(class_list)

    # NOTE: copy/transpose
    Xm_t   = MLJBase.matrix(X, transpose=true) # now p x n matrix
    yplain = MLJBase.int(y) # vector of n ints in {1,..., nclasses}
    p, n   = size(Xm_t)

    # check output dimension default is min(p, nc-1)
    def_outdim = min(p, nclasses - 1)
    # if unset (0) use the default; otherwise try to use the provided one
    out_dim = ifelse(model.out_dim == 0, def_outdim, model.out_dim)
    # check if the given one is sensible
    out_dim ≤ def_outdim || throw(ArgumentError("`out_dim` must not be larger than `min(p, nc-1)` where `p` is the dimension of `X` and `nc` is the number of classes."))

    ## Estimates prior probabilities is unspecified by user.
    if model.priors == nothing
        priors = proportions(yplain)
    else
        #check if the length of priors is same as nclasses
        size(MLJBase.classes(model.priors)) == nclasses || throw(ArgumentError("Invalid size of `priors`"))
        priors = MLJBase.pdf.(model.priors, class_list)
    end

    core_res = MS.fit(MS.MulticlassLDA, nclasses, Xm_t, Int.(yplain);
                      method=model.method,
                      outdim=out_dim,
                      regcoef=model.regcoef,
                      covestimator_within=model.cov_w,
                      covestimator_between=model.cov_b)

    ## The original projection matrix satisfies Pᵀ*Sw*P=I
    ## scaled projection_matrix and core_res.proj by multiplying by sqrt(n - nclasses) this ensures Pᵀ*Σ*P=I
    ## where covariance estimate Σ = Sw / (n - nclasses)
    core_res.proj   .*= sqrt(n - nclasses)
    core_res.pmeans .*= sqrt(n - nclasses)

    cache     = nothing
    report    = NamedTuple{}()
    fitresult = (core_res, class_list, priors)

    return fitresult, cache, report
end

function MLJBase.fitted_params(::BayesianLDA, (core_res, class_list, priors))
    return (class_means       = MS.classmeans(core_res),
            projection_matrix = MS.projection(core_res),
            priors            = priors)
end

function MLJBase.predict(m::BayesianLDA, (core_res, class_list, priors), Xnew)
    # projection of Xnew XWt is n x o  where o = number of out dims
    XWt = MLJBase.matrix(Xnew) * core_res.proj
    # centroids in the transformed space, nc x o
    centroids = permutedims(core_res.pmeans)

    n  = core_res.stats.tweight # n is the Number of training examples
    nclasses = size(class_list)

    # compute the distances in the transformed space between pairs of rows
    # The discriminant matrix `P` is of dimension `n x nc`
    #  P[i,k] = -0.5*(xᵢ −  µₖ)ᵀΣ⁻¹(xᵢ −  µₖ) + log(priorsₖ) and Σ⁻¹ = I due to the nature of the projection_matrix
    P = pairwise(SqEuclidean(), XWt, centroids, dims=1)
    P .*= -0.5
    P .+= log.(priors)'

    # apply a softmax transformation to convert P to a probability matrix
    P  .= exp.(P)
    P ./= sum(P, dims=2)

    return [MLJBase.UnivariateFinite(class_list, P[j, :]) for j in 1:size(P, 1)]
end


####
#### METADATA
####

metadata_pkg.((RidgeRegressor, PCA, KernelPCA, ICA, LDA, BayesianLDA),
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

metadata_model.((LDA, BayesianLDA),
               input=MLJBase.Table(MLJBase.Continuous),
               target=AbstractVector{<:MLJBase.Finite},
               weights=false,
               descr=LDA_DESCR)

end # of module+
