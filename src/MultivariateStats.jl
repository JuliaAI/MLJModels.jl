module MultivariateStats_

export RidgeRegressor, PCA, KernelPCA, ICA, LDA, BayesianLDA

import MLJModelInterface
import MLJModelInterface: @mlj_model, metadata_pkg, metadata_model,
    Table, Continuous, Finite

const MMI = MLJModelInterface

import StatsBase: CovarianceEstimator 
using Distances
using LinearAlgebra

import MultivariateStats

const MS = MultivariateStats

struct LinearFitresult{F} <: MMI.MLJType
    coefficients::Vector{F}
    intercept::F
end

const RIDGE_DESCR = "Ridge regressor with regularization parameter lambda. Learns a linear regression with a penalty on the l2 norm of the coefficients."
const PCA_DESCR = "Principal component analysis. Learns a linear transformation to project the data  on a lower dimensional space while preserving most of the initial variance."
const KPCA_DESCR = "Kernel principal component analysis."
const ICA_DESCR = "Independent component analysis."
const LDA_DESCR = "Multiclass linear discriminant analysis. The algorithm learns a projection matrix `P` that projects a feature matrix `Xtrain` onto a lower dimensional space of dimension `out_dim` such that the trace of the transformed between-class scatter matrix(`Pᵀ*Sb*P`) is maximized relative to the trace of the transformed within-class scatter matrix (`Pᵀ*Sw*P`).The projection matrix is scaled such that `Pᵀ*Sw*P=I` or `Pᵀ*Σw*P=I`(where `Σw` is the within-class covariance matrix) . \nPredicted class posterior probability for feature matrix `Xtest` are derived by applying a softmax transformation to a matrix `Pr`, such that  rowᵢ of `Pr` contains computed distances(based on a distance metric) in the transformed space of rowᵢ in `Xtest` to the centroid of each class. "
const BayesianLDA_DESCR = "Bayesian Multiclass linear discriminant analysis. The algorithm learns a projection matrix `P` that projects a feature matrix `Xtrain` onto a lower dimensional space of dimension `out_dim` such that the trace of the transformed between-class scatter matrix(`Pᵀ*Sb*P`) is maximized relative to the trace of the transformed within-class scatter matrix (`Pᵀ*Sw*P`). The projection matrix is scaled such that `Pᵀ*Sw*P = n` or `Pᵀ*Σw*P=I` (Where `n` is the number of training samples and `Σw` is the within-class covariance matrix).\nPredicted class posterior probability distibution are derived by applying Bayes rule with a multivariate Gaussian class-conditional distribution."
const BayesianSubspaceLDA_DESCR = "Bayesian Multiclass linear discriminant analysis. Suitable for high dimensional data(Avoids computing scatter matrices `Sw` ,`Sb`). The algorithm learns a projection matrix `P = W*L` (`Sw`), that projects a feature matrix `Xtrain` onto a lower dimensional space of dimension `nc-1` such that the trace of the transformed between-class scatter matrix(`Pᵀ*Sb*P`) is maximized relative to the trace of the transformed within-class scatter matrix (`Pᵀ*Sw*P`). The projection matrix is scaled such that `Pᵀ*Sw*P = mult*I` or `Pᵀ*Σw*P=mult/(n-nc)*I` (where `n` is the number of training samples, `mult` is  one of `n` or `1` depending on whether `Sb` is normalized, `Σw` is the within-class covariance matrix, and `nc` is the number of unique classes in `y`) and also obeys `Wᵀ*Sb*p = λ*Wᵀ*Sw*p`, for every column `p` in `P`. \nPosterior class probability distibution are derived by applying Bayes rule with a multivariate Gaussian class-conditional distribution"
const SubspaceLDA_DESCR = "Multiclass linear discriminant analysis. Suitable for high dimensional data (Avoids computing scatter matrices `Sw` ,`Sb`). The algorithm learns a projection matrix `P = W*L` that projects a feature matrix `Xtrain` onto a lower dimensional space of dimension `nc - 1 ` such that the trace of the transformed between-class scatter matrix(`Pᵀ*Sb*P`) is maximized relative to the trace of the transformed within-class scatter matrix (`Pᵀ*Sw*P`). The projection matrix is scaled such that `Pᵀ*Sw*P = mult*I` or `Pᵀ*Σw*P=mult/(n-nc)*I` (where `n` is the number of training samples, mult` is  one of `n` or `1` depending on whether `Sb` is normalized, `Σw` is the within-class covariance matrix, and `nc` is the number of unique classes in `y`) and also obeys `Wᵀ*Sb*p = λ*Wᵀ*Sw*p`, for every column `p` in `P`.\nPredicted class posterior probability for feature matrix `Xtest` are derived by applying a softmax transformation to a matrix `Pr`, such that  rowᵢ of `Pr` contains computed distances(based on a distance metric) in the transformed space of rowᵢ in `Xtest` to the centroid of each class. "

####
#### RIDGE
####

"""
    RidgeRegressor(; lambda=1.0)

$RIDGE_DESCR

## Parameters

* `lambda=1.0`: non-negative parameter for the regularization strength.
"""
@mlj_model mutable struct RidgeRegressor <: MMI.Deterministic
    lambda::Real = 1.0::(_ ≥ 0)
end

function MMI.fit(model::RidgeRegressor, verbosity::Int, X, y)
    Xmatrix   = MMI.matrix(X)
    features  = MMI.schema(X).names
    θ         = MS.ridge(Xmatrix, y, model.lambda)
    coefs     = θ[1:end-1]
    intercept = θ[end]

    fitresult = LinearFitresult(coefs, intercept)
    report    = NamedTuple()
    cache     = nothing

    return fitresult, cache, report
end

MMI.fitted_params(::RidgeRegressor, fr) =
    (coefficients=fr.coefficients, intercept=fr.intercept)

function MMI.predict(::RidgeRegressor, fr, Xnew)
    Xmatrix = MMI.matrix(Xnew)
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

* `maxoutdim=nothing`: maximum number of output dimensions, unconstrained if
                       nothing.
* `method=:auto`:      method to use to solve the problem, one of `:auto`,
                       `:cov` or `:svd`
* `pratio=0.99`:       ratio of variance preserved
* `mean=nothing`:      if set to nothing centering will be computed and
                       applied, if set to `0` no centering (assumed pre-
                       centered), if a vector is passed, the centering is done
                       with that vector.
"""
@mlj_model mutable struct PCA <: MMI.Unsupervised
    maxoutdim::Union{Nothing,Int} = nothing::(_ === nothing || _ ≥ 1)
    method::Symbol  = :auto::(_ in (:auto, :cov, :svd))
    pratio::Float64 = 0.99::(0.0 < _ ≤ 1.0)
    mean::Union{Nothing, Real, Vector{Float64}} = nothing::(_ === nothing || (_ isa Real && iszero(_)) || true)
end

function MMI.fit(model::PCA, verbosity::Int, X)
    Xarray = MMI.matrix(X)
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

MMI.fitted_params(::PCA, fr) = (projection=fr,)


function MMI.transform(::PCA, fr::PCAFitResultType, X)
    # X is n x d, need to transpose and copy twice...
    Xarray = MMI.matrix(X)
    Xnew   = permutedims(MS.transform(fr, permutedims(Xarray)))
    return MMI.table(Xnew, prototype=X)
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

* `maxoutdim=nothing`: maximum number of output dimensions, unconstrained if
                       nothing.
* `kernel=nothing`:    kernel function of 2 vector arguments x and y, returns a
                       scalar value, (x,y)->x'y if nothing
* `solver=:auto`:      solver to use for the eigenvalues, one of `:eig`
                       (default), `:eigs`
* `inverse=false`:     perform calculation for inverse transform
* `beta=1.0`:          strength of the ridge regression that learns the inverse
                       transform when inverse is true
* `tol=0.0`:           Convergence tolerance for eigs solver
* `maxiter=300`:       maximum number of iterations for eigs solver
"""
@mlj_model mutable struct KernelPCA <: MMI.Unsupervised
    maxoutdim::Union{Nothing,Int} = nothing::(_ === nothing || _ ≥ 1)
    kernel::Function = default_kernel
    solver::Symbol   = :eig::(_ in (:eig, :eigs))
    inverse::Bool    = false
    beta::Real       = 1.0::(_ ≥ 0.0)
    tol::Real        = 1e-6::(_ ≥ 0.0)
    maxiter::Int     = 300::(_ ≥ 1)
end

function MMI.fit(model::KernelPCA, verbosity::Int, X)
    Xarray = MMI.matrix(X)
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

MMI.fitted_params(::KernelPCA, fr) = (projection=fr,)

function MMI.transform(::KernelPCA, fr::KernelPCAFitResultType, X)
    # X is n x d, need to transpose and copy twice...
    Xarray = MMI.matrix(X)
    Xnew   = permutedims(MS.transform(fr, permutedims(Xarray)))
    return MMI.table(Xnew, prototype=X)
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

* `k=0`:              number of independent components to recover, set
                      automatically if `0`
* `alg=:fastica`:     algorithm to use (only `:fastica` is supported at the
                      moment)
* `fun=:tanh`:        approximate neg-entropy functor, via the function
                      `MultivariateStats.icagfun`, one of `:tanh` and `:gaus`
* `do_whiten=true`:   whether to perform pre-whitening
* `maxiter=100`:      maximum number of iterations
* `tol=1e-6`:         convergence tolerance for change in matrix W
* `mean=nothing`:     mean to use, if nothing centering is computed and
                      applied, if zero, no centering, a vector of means can be
                      passed
* `winit=zeros(0,0)`: initial guess for matrix `W` either an empty matrix
                      (random init), a matrix of size `k × k` (if `do_whiten`
                      is true), a matrix of size `m × k` otherwise
"""
@mlj_model mutable struct ICA <: MMI.Unsupervised
    k::Int          = 0::(_ ≥ 0)
    alg::Symbol     = :fastica::(_ in (:fastica,))
    fun::Symbol     = :tanh::(_ in (:tanh, :gaus))
    do_whiten::Bool = true
    maxiter::Int    = 100
    tol::Real       = 1e-6::(_ ≥ 0.0)
    winit::Union{Nothing,Matrix{<:Real}} = nothing
    mean::Union{Nothing, Real, Vector{Float64}} = nothing::(_ === nothing || (_ isa Real && iszero(_)) || true)
end

function MMI.fit(model::ICA, verbosity::Int, X)
    Xarray = MMI.matrix(X)
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

MMI.fitted_params(::ICA, fr) = (projection=fr,)

function MMI.transform(::ICA, fr::ICAFitResultType, X)
    # X is n x d, need to transpose and copy twice...
    Xarray = MMI.matrix(X)
    Xnew   = permutedims(MS.transform(fr, permutedims(Xarray)))
    return MMI.table(Xnew, prototype=X)
end

####
#### MulticlassLDA
####

"""
    LDA(; kwargs...)

$LDA_DESCR

## Parameters

* `method=:gevd`:      choice of solver, one of `:gevd` or `:whiten` methods
* `cov_w=SimpleCovariance()`: an estimator for the within-class covariance (used in computing
                              within-class scatter matrix, Sw), by
                              default set to the standard
                              `MultivariateStats.CovarianceEstimator` but could
                              be set to any robust estimator from `CovarianceEstimation.jl`.
* `cov_b=SimpleCovariance()`: same as `cov_w` but for the between-class
                              covariance(used in computing between-class scatter matrix, Sb).
* `out_dim`:          the output dimension, i.e dimension of the transformed
                      space, automatically set if 0 is given (default).
* `regcoef`:          regularization coefficient (default value 1e-6). A
                      positive value `regcoef * eigmax(Sw)` where `Sw` is the
                      within-class scatter matrix, is added to the
                      diagonal of Sw to improve numerical stability. This can
                      be useful if using the standard covariance estimator.
* `dist=SqEuclidean`: the distance metric to use when performing classification
                      (to compare the distance between a new point and
                      centroids in the transformed space), an alternative
                      choice can be the `CosineDist`.

See also the [package documentation](https://multivariatestatsjl.readthedocs.io/en/latest/lda.html).
For more information about the algorithm, see the paper by Li, Zhu and Ogihara, [Using Discriminant Analysis for Multi-class Classification: An Experimental Investigation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.7068&rep=rep1&type=pdf).
"""
@mlj_model mutable struct LDA <: MMI.Probabilistic
    method::Symbol   = :gevd::(_ in (:gevd, :whiten))
    cov_w::CovarianceEstimator = MS.SimpleCovariance()
    cov_b::CovarianceEstimator = MS.SimpleCovariance()
    out_dim::Int     = 0::(_ ≥ 0)
    regcoef::Float64    = 1e-6::(_ ≥ 0)
    dist::SemiMetric = SqEuclidean()
end

function MMI.fit(model::LDA, ::Int, X, y)
    class_list = MMI.classes(y[1]) #class list containing entries in pool of y
    classes_seen  = filter(in(unique(y)), class_list) #class list containing unique entries in y
    nc   = length(classes_seen) #number of classes in pool of y
    nclasses = length(class_list)
    nc == nclasses || (integers_seen = MMI.int(classes_seen))
    
    # NOTE: copy/transpose
    Xm_t   = MMI.matrix(X, transpose=true) # now p x n matrix
    yplain = MMI.int(y) # vector of n ints in {1,..., nclasses}
    p, n   = size(Xm_t)
    #recode yplain to be in {1,..., nc}
    nc == nclasses ||  replace!(yplain, (integers_seen .=> 1:nc)...)
    
    #check to make sure we have more than one class in training sample 
    # This is to prevent Sb from being a zero matrix
    nc >= 2 || throw(ArgumentError("The number of unique classes in traning sample"*
                                        " `nc` has to be greater than one"))
    
    #check to make sure we have more samples than classes
    # This is to prevent Sw from being the zero matrix
    n > nc || throw(ArgumentError("The number of training samples `n` has to be"*
                                    " greater than the number of unique classes `nc`"))


    # check output dimension default is min(p, nc-1)
    def_outdim = min(p, nc - 1)
    # if unset (0) use the default; otherwise try to use the provided one
    out_dim = ifelse(model.out_dim == 0, def_outdim, model.out_dim)
    # check if the given one is sensible
    out_dim ≤ p || throw(ArgumentError("`out_dim` must not be larger than `p`"*
                                        "where `p` is the number of features in `X`"))


    core_res = MS.fit(MS.MulticlassLDA, nc, Xm_t, Int.(yplain);
                      method=model.method,
                      outdim=out_dim,
                      regcoef=model.regcoef,
                      covestimator_within=model.cov_w,
                      covestimator_between=model.cov_b)

    cache     = nothing
    report    = (classes       = classes_seen,
                 out_dim       = MS.outdim(core_res),
                 class_means   = MS.classmeans(core_res),
                 mean          = MS.mean(core_res),
                 class_weights = MS.classweights(core_res),
                 Sw            = MS.withclass_scatter(core_res),
                 Sb            = MS.betweenclass_scatter(core_res),
                 nc            = nc)
    fitresult = (core_res, classes_seen)

    return fitresult, cache, report
end

function MMI.fitted_params(::LDA, (core_res, classes_seen))
    return (projected_class_means = MS.classmeans(core_res),
            projection_matrix     = MS.projection(core_res))
end

function MMI.predict(m::LDA, (core_res, classes_seen), Xnew)
    # projection of Xnew XWt is n x o  where o = number of out dims
    XWt = MMI.matrix(Xnew) * core_res.proj
    # centroids in the transformed space, nc x o
    centroids = permutedims(core_res.pmeans)

    # compute the distances in the transformed space between pairs of rows
    # the probability matrix is `n x nc` and normalised accross rows
    P = pairwise(m.dist, XWt, centroids, dims=1)
    # apply a softmax transformation
    P .-= maximum(P, dims=2)
    P  .= exp.(-P)
    P ./= sum(P, dims=2)

    return MMI.UnivariateFinite(classes_seen, P)
end

####
#### BayesianLDA
####

SymORStr = Union{Symbol,String}
"""
    BayesianLDA(; kwargs...)

$BayesianLDA_DESCR

## Parameters

* `method=:gevd`:    choice of solver, one of `:gevd` or `:whiten`
                    methods
* `cov_w=SimpleCovariance()`: an estimator for the within-class covariance (used in computing
                              within-class scatter matrix, Sw), by
                              default set to the standard
                              `MultivariateStats.CovarianceEstimator` but could
                              be set to any robust estimator from `CovarianceEstimation.jl`.
* `cov_b=SimpleCovariance()`: same as `cov_w` but for the between-class
                              covariance(used in computing between-class scatter matrix, Sb).
* `out_dim`:                  the output dimension, i.e dimension of the
                    transformed space, automatically set if 0 is
                    given (default).
* `regcoef`:        regularization coefficient (default value 1e-6).
                    A positive value `regcoef * eigmax(Sw)` where `Sw` is the
                    within-class covariance estimator, is added to the diagonal
                    of Sw to improve numerical stability. This can be useful if
                    using the standard covariance estimator.
* `priors=nothing`: For use in prediction with Bayes rule. if `priors = nothing` estimates the prior probabilities
                    from the data else it uses a user specified Dictionary(`::AbstractDict{<:Union{Symbol, String}, <:Real}}`)
                    of `class => probability` pairs for each unique class found in training data.

See also the [package documentation](https://multivariatestatsjl.readthedocs.io/en/latest/lda.html).
For more information about the algorithm, see the paper by Li, Zhu and Ogihara, [Using Discriminant Analysis for Multi-class Classification: An Experimental Investigation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.7068&rep=rep1&type=pdf).
"""
@mlj_model mutable struct BayesianLDA <: MMI.Probabilistic
    method::Symbol   = :gevd::(_ in (:gevd, :whiten))
    cov_w::CovarianceEstimator = MS.SimpleCovariance()
    cov_b::CovarianceEstimator = MS.SimpleCovariance()
    out_dim::Int     = 0::(_ ≥ 0)
    regcoef::Float64    = 1e-6::(_ ≥ 0)
    priors::Union{Nothing, AbstractDict{<:SymORStr, <:Real} } = nothing
end

function MMI.fit(model::BayesianLDA, ::Int, X, y)
    class_list = MMI.classes(y[1]) #class list containing entries in pool of y
    classes_seen  = filter(in(unique(y)), class_list) #class list containing unique entries in y
    nc   = length(classes_seen) #number of classes in pool of y
    nclasses = length(class_list)
    nc == nclasses || (integers_seen = MMI.int(classes_seen))
    
    # NOTE: copy/transpose
    Xm_t   = MMI.matrix(X, transpose=true) # now p x n matrix
    yplain = MMI.int(y) # vector of n ints in {1,..., nclasses}
    p, n   = size(Xm_t)
    #recode yplain to be in {1,..., nc}
    nc == nclasses ||  replace!(yplain, (integers_seen .=> 1:nc)...)
   
    #check to make sure we have more than one class in training sample
    #This is to prevent Sb from being a zero matrix
    nc >= 2 || throw(ArgumentError("The number of unique classes in "*
                                            "traning sample has to be greater than one"))
    #check to make sure we have more samples than classes
    #This is to prevent Sw from being the zero matrix
    n > nc || throw(ArgumentError("The number of training samples `n` has"*
                                    " to be greater than number of unique classes `nc`"))

    # check output dimension default is min(p, nc-1)
    def_outdim = min(p, nc - 1)
    # if unset (0) use the default; otherwise try to use the provided one
    out_dim = ifelse(model.out_dim == 0, def_outdim, model.out_dim)
    # check if the given one is sensible
    out_dim ≤ p || throw(ArgumentError("`out_dim` must not be larger than `p`"*
                                            "where `p` is the number of features in `X`"))

    core_res = MS.fit(MS.MulticlassLDA, nc, Xm_t, Int.(yplain);
                      method=model.method,
                      outdim=out_dim,
                      regcoef= model.regcoef,
                      covestimator_within=model.cov_w,
                      covestimator_between=model.cov_b)

    ## Estimates prior probabilities if unspecified by user
    ## Or check if user specified prior makes sense.
    if isa(model.priors, Nothing)
        weights = MS.classweights(core_res)
        total = core_res.stats.tweight
        priors = weights ./ total
    else
        prior_classes = sort!(collect(keys(model.priors)))
        length(prior_classes) == nc || throw(ArgumentError("Invalid size of `priors`"))
        all(prior_classes .== classes_seen) || throw(ArgumentError("classes used as `keys` in"*
                                            "constructing `priors` must match unique classes of target vector `y` "))        
        priors = getindex((model.priors,), prior_classes)
        isapprox(sum(priors), 1) || throw(ArgumentError("probabilities in `priors` must sum to 1"))
        all(priors .>= 0) || throw(ArgumentError("probabilities in `priors` must non-negative"))      
    end

    cache     = nothing
    report    = (classes       = classes_seen,
                 out_dim       = MS.outdim(core_res),
                 class_means   = MS.classmeans(core_res),
                 mean          = MS.mean(core_res),
                 class_weights = MS.classweights(core_res),
                 Sw            = MS.withclass_scatter(core_res),
                 Sb            = MS.betweenclass_scatter(core_res),
                 nc            = nc)

    fitresult = (core_res, classes_seen, priors, n)

    return fitresult, cache, report
end

function MMI.fitted_params(::BayesianLDA, (core_res, classes_seen, priors, n))
   return ( projected_class_means = MS.classmeans(core_res),
            projection_matrix     = MS.projection(core_res),
            priors                = priors)
end

function MMI.predict(m::BayesianLDA, (core_res, classes_seen, priors, n), Xnew)
    # projection of Xnew XWt is nt x o  where o = number of out dims
    XWt = MMI.matrix(Xnew) * core_res.proj
    # centroids in the transformed space, nc x o
    centroids = permutedims(core_res.pmeans)
  
    # compute the distances in the transformed space between pairs of rows
    # The discriminant matrix `Pr` is of dimension `nt x nc`
    # Pr[i,k] = -0.5*(xᵢ −  µₖ)ᵀ(Σw⁻¹)(xᵢ −  µₖ) + log(priorsₖ) where (Σw = Sw/n)
    # In the transformed space this becomes
    # Pr[i,k] = -0.5*(Pᵀxᵢ −  Pᵀµₖ)ᵀ(PᵀΣw⁻¹P)(Pᵀxᵢ −  Pᵀµₖ) + log(priorsₖ)
    # PᵀSw⁻¹P = I and PᵀΣw⁻¹P = n*I due to the nature of the projection_matrix, P
    # Giving Pr[i,k] = -0.5*n*(Pᵀxᵢ −  Pᵀµₖ)ᵀ(Pᵀxᵢ −  Pᵀµₖ) + log(priorsₖ)
    # (Pᵀxᵢ −  Pᵀµₖ)ᵀ(Pᵀxᵢ −  Pᵀµₖ) is the SquaredEquclidean distance in the transformed space
    Pr = pairwise(SqEuclidean(), XWt, centroids, dims=1)
    Pr .*= (-0.5*n)
    Pr .+= log.(priors)'

    # apply a softmax transformation to convert Pr to a probability matrix
    Pr .-= maximum(Pr, dims=2)
    Pr  .= exp.(Pr)
    Pr ./= sum(Pr, dims=2)

    return MMI.UnivariateFinite(classes_seen, Pr)
end

####
#### BayesianSubspaceLDA
####

"""
    BayesianSubspaceLDA(; kwargs...)

$BayesianSubspaceLDA_DESCR

## Parameters

* `normalize=true`: Option to normalize the between class variance for the number of
                    observations in each class, one of `true` or `false`.
* `out_dim`:        the dimension of the transformed space
                    to be used by `predict` and `transform` methods, automatically set if 0 is given (default).
* `priors=nothing`: For use in prediction with Bayes rule. if `priors = nothing` estimates the prior probabilities
                    from the data else it uses a user specified Dictionary(`::AbstractDict{<:Union{Symbol, String}, <:Real}}`)
                    of `class => probability` pairs for each unique class found in training data.

For more information about the algorithm, see the paper by Howland & Park (2006), 
"Generalizing discriminant analysis using the generalized singular value decomposition",IEEE Trans. Patt.
 Anal. & Mach. Int., 26: 995-1006.
"""
@mlj_model mutable struct BayesianSubspaceLDA <: MMI.Probabilistic
    normalize::Bool=false
    out_dim::Int   = 0::(_ ≥ 0)
    priors::Union{Nothing, AbstractDict{SymORStr, <:Real} } = nothing
end

function MMI.fit(model::BayesianSubspaceLDA, ::Int, X, y)
    class_list = MMI.classes(y[1]) #class list containing entries in pool of y
    classes_seen  = filter(in(unique(y)), class_list) #class list containing unique entries in y
    nc   = length(classes_seen) #number of classes in pool of y
    nclasses = length(class_list)
    nc == nclasses || (integers_seen = MMI.int(classes_seen))
    
    # NOTE: copy/transpose
    Xm_t   = MMI.matrix(X, transpose=true) # now p x n matrix
    yplain = MMI.int(y) # vector of n ints in {1,..., nclasses}
    p, n   = size(Xm_t)
    #recode yplain to be in {1,..., nc}
    nc == nclasses ||  replace!(yplain, (integers_seen .=> 1:nc)...)
   
    #check to make sure we have more than one class in training sample
    #This is to prevent Sb from being a zero matrix
    nc >= 2 || throw(ArgumentError("The number of unique classes in traning sample"*
                                    " has to be greater than one"))
    #check to make sure we have more samples than classes
    #This is to prevent Sw from being the zero matrix
    n > nc || throw(ArgumentError("The number of training samples has to be greater"*
                                    " than the number of unique classes in `y`"))

    # check output dimension default is  min(p, nc-1)
    def_outdim = min(p, nc - 1)
    # if unset (0) use the default; otherwise try to use the provided one
    out_dim = ifelse(model.out_dim == 0, def_outdim, model.out_dim)
    # check if the given one is sensible
    out_dim ≤ nc - 1 || throw(ArgumentError("`out_dim` must not be larger than `nc - 1`"*
                                            " where  `nc` is the number of unique classes in `y`"))

    core_res = MS.fit(MS.SubspaceLDA, Xm_t,
                        Int.(yplain),
                        nc;
                        normalize = model.normalize)
    
    λ = core_res.λ # λ is a (nc -1) x 1 vector containing the eigen values sorted in descending order.
    explained_variance_ratio = λ ./ sum(λ) #proportions of variance
    
    mult = model.normalize ? n : 1 #used in prediction

    ## Estimates prior probabilities if specified by user.
    ## Or check if user specified prior makes sense.
    if isa(model.priors, Nothing)
        weights = MS.classweights(core_res)
        priors = weights ./ n
    else 
       prior_classes = sort!(collect(keys(model.priors)))
        length(prior_classes) == nc || throw(ArgumentError("Invalid size of `priors`"))
        all(prior_classes .== classes_seen) || throw(ArgumentError("classes used as `keys` in"*
                                            "constructing `priors` must match unique classes of target vector `y` "))        
        priors = getindex((model.priors,), prior_classes)
        isapprox(sum(priors), 1) || throw(ArgumentError("probabilities in `priors` must sum to 1"))
        all(priors .>= 0) || throw(ArgumentError("probabilities in `priors` must non-negative"))  
    end

    cache     = nothing
    report    = (explained_variance_ratio  = explained_variance_ratio,
                 classes       = classes_seen,
                 class_means   = MS.classmeans(core_res),
                 mean          = MS.mean(core_res),
                 class_weights = MS.classweights(core_res),
                 nc            = nc)
    fitresult = (core_res, out_dim, classes_seen, priors, n, mult)

    return fitresult, cache, report
end

function MMI.fitted_params(::BayesianSubspaceLDA, (core_res, _, _, priors,_))
    return (projected_class_means  = MS.classmeans(core_res),
            projection_matrix      = MS.projection(core_res),
            priors                 = priors)
end

function MMI.predict(m::BayesianSubspaceLDA, (core_res, out_dim, classes_seen, priors, n, mult), Xnew)
    # projection of Xnew XWt is nt x o  where o = number of out dims
    proj = core_res.projw * view(core_res.projLDA, :, 1:out_dim) #proj is the projection_matrix
    XWt = MMI.matrix(Xnew) * proj
    
    # centroids in the transformed space, nc x o
    centroids = permutedims(proj' * core_res.cmeans)
    nc = length(classes_seen)

    # compute the distances in the transformed space between pairs of rows
    # The discriminant matrix `Pr` is of dimension `nt x nc`
    # Pr[i,k] = -0.5*(xᵢ −  µₖ)ᵀ(Σw⁻¹)(xᵢ −  µₖ) + log(priorsₖ) where (Σw = Sw/(n-nc)
    # In the transformed space this becomes
    # Pr[i,k] = -0.5*(Pᵀxᵢ −  Pᵀµₖ)ᵀ(PᵀΣw⁻¹P)(Pᵀxᵢ −  Pᵀµₖ) + log(priorsₖ)
    # PᵀSw⁻¹P = (1/mult)*I and PᵀΣw⁻¹P = (n-nc)/mult*I due to the nature of the projection_matrix, P
    # Giving Pr[i,k] = -0.5*n*(Pᵀxᵢ −  Pᵀµₖ)ᵀ(Pᵀxᵢ −  Pᵀµₖ) + log(priorsₖ)
    # (Pᵀxᵢ −  Pᵀµₖ)ᵀ(Pᵀxᵢ −  Pᵀµₖ) is the SquaredEquclidean distance in the transformed space  
    Pr = pairwise(SqEuclidean(), XWt, centroids, dims=1)
    Pr .*= (-0.5 * (n-nc)/mult)
    Pr .+= log.(priors)'

    # apply a softmax transformation to convert Pr to a probability matrix
    Pr .-= maximum(Pr, dims=2)
    Pr  .= exp.(Pr)
    Pr ./= sum(Pr, dims=2)

    return MMI.UnivariateFinite(classes_seen, Pr)
end

####
#### SubspaceLDA
####

"""
    SubspaceLDA(; kwargs...)

$SubspaceLDA_DESCR

## Parameters

* `normalize=true`:   Option to normalize the between class variance for the number
                      of observations in each class, one of `true` or `false`.
* `out_dim`:        the dimension of the transformed space
                    to be used by `predict` and `transform` methods, automatically set if 0 is given (default).
* `dist=SqEuclidean`: the distance metric to use when performing classification
                      (to compare the distance between a new point and
                      centroids in the transformed space), an alternative
                      choice can be the `CosineDist`.

See also the [package documentation](https://multivariatestatsjl.readthedocs.io/en/latest/lda.html).
For more information about the algorithm, see the paper by Howland & Park (2006), "Generalizing discriminant analysis using the generalized singular value decomposition", IEEE Trans. Patt. Anal. & Mach. Int., 26: 995-1006.
"""
@mlj_model mutable struct SubspaceLDA <: MMI.Probabilistic
    normalize::Bool=true
    out_dim::Int   = 0::(_ ≥ 0)
    dist::SemiMetric = SqEuclidean()
end

function MMI.fit(model::SubspaceLDA, ::Int, X, y)
    class_list = MMI.classes(y[1]) #class list containing entries in pool of y
    classes_seen  = filter(in(unique(y)), class_list) #class list containing unique entries in y
    nc   = length(classes_seen) #number of classes in pool of y
    nclasses = length(class_list)
    nc == nclasses || (integers_seen = MMI.int(classes_seen))
    
    # NOTE: copy/transpose
    Xm_t   = MMI.matrix(X, transpose=true) # now p x n matrix
    yplain = MMI.int(y) # vector of n ints in {1,..., nclasses}
    p, n   = size(Xm_t)
    #recode yplain to be in {1,..., nc}
    nc == nclasses ||  replace!(yplain, (integers_seen .=> 1:nc)...)

    #check to make sure we have more than one class in training sample
    #This is to prevent Sb from being a zero matrix
    nc >= 2 || throw(ArgumentError("The number of unique classes in traning sample"*
                                    " has to be greater than one"))
    #check to make sure we have more samples than classes
    #This is to prevent Sw from being the zero matrix
    n > nc || throw(ArgumentError("The number of training samples has to be greater"*
                                    " than the number of unique classes in `y`"))

    # check output dimension default is  min(p, nc-1)
    def_outdim = min(p, nc - 1)
    # if unset (0) use the default; otherwise try to use the provided one
    out_dim = ifelse(model.out_dim == 0, def_outdim, model.out_dim)
    # check if the given one is sensible
    out_dim ≤ nc - 1 || throw(ArgumentError("`out_dim` must not be larger than `nc - 1`"*
                                            " where  `nc` is the number of unique classes in `y`"))

    core_res = MS.fit(MS.SubspaceLDA, Xm_t,
                        Int.(yplain),
                        nc;
                        normalize = model.normalize)

    λ = core_res.λ # λ is a (nc -1) x 1 vector containing the eigen values sorted in descending order.
    explained_variance_ratio = λ ./ sum(λ) #proportions of variance

    cache     = nothing
    report    = (explained_variance_ratio  = explained_variance_ratio,
                 classes       = classes_seen,
                 class_means   = MS.classmeans(core_res),
                 mean          = MS.mean(core_res),
                 class_weights = MS.classweights(core_res),
                 nc            = nc)
    fitresult = (core_res, out_dim, classes_seen)

    return fitresult, cache, report
end

function MMI.fitted_params(::SubspaceLDA, (core_res, _))
    return (class_means       = MS.classmeans(core_res),
            projection_matrix = MS.projection(core_res))
end

function MMI.predict(m::SubspaceLDA, (core_res, out_dim, classes_seen), Xnew)
    # projection of Xnew, XWt is nt x o  where o = number of out dims
    proj = core_res.projw * view(core_res.projLDA, :, 1:out_dim) #proj is the projection_matrix
    XWt = MMI.matrix(Xnew) * proj
    # centroids in the transformed space, nc x o
    centroids = permutedims(proj' * core_res.cmeans)

    # compute the distances in the transformed space between pairs of rows
    # the probability matrix is `nt x nc` and normalised accross rows
    P = pairwise(m.dist, XWt, centroids, dims=1)

    # apply a softmax transformation
    P .-= maximum(P, dims=2)
    P  .= exp.(-P)
    P ./= sum(P, dims=2)

    return MMI.UnivariateFinite(classes_seen, P)
end

function MMI.transform(m::T, (core_res, out_dim, _), X) where T<:Union{SubspaceLDA, BayesianSubspaceLDA}
    # projection of X, XWt is nt x o  where o = out dims
    proj = core_res.projw * view(core_res.projLDA, :, 1:out_dim) #proj is the projection_matrix
    XWt = MMI.matrix(X) * proj
    return MMI.table(XWt, prototype = X)
end

function MMI.transform(m::T, (core_res,), X) where T<:Union{LDA, BayesianLDA}
    # projection of X, XWt is nt x o  where o = out dims
    proj = core_res.projw * core_res.projLDA #proj is the projection_matrix
    XWt = MMI.matrix(X) * proj
    return MMI.table(XWt, prototype = X)
end


####
#### METADATA
####

metadata_pkg.(
    (RidgeRegressor, PCA, KernelPCA, ICA, LDA,
     BayesianLDA, SubspaceLDA,  BayesianSubspaceLDA),
    name       = "MultivariateStats",
    uuid       = "6f286f6a-111f-5878-ab1e-185364afe411",
    url        = "https://github.com/JuliaStats/MultivariateStats.jl",
    license    = "MIT",
    julia      = true,
    is_wrapper = false)

metadata_model(RidgeRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Continuous},
    weights = false,
    descr   = RIDGE_DESCR)

metadata_model(PCA,
    input   = Table(Continuous),
    target  = Table(Continuous),
    weights = false,
    descr   = PCA_DESCR)

metadata_model(KernelPCA,
    input   = Table(Continuous),
    target  = Table(Continuous),
    weights = false,
    descr   = KPCA_DESCR)

metadata_model(ICA,
    input   = Table(Continuous),
    target  = Table(Continuous),
    weights = false,
    descr   = ICA_DESCR)

metadata_model(LDA,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    output  = Table(Continuous),
    descr   = LDA_DESCR)

metadata_model(BayesianLDA,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    output  = Table(Continuous),
    descr   = BayesianLDA_DESCR)

metadata_model(SubspaceLDA,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    output  = Table(Continuous),
    descr   = SubspaceLDA_DESCR)

metadata_model(BayesianSubspaceLDA,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    output  = Table(Continuous),
    descr   = BayesianSubspaceLDA_DESCR)

end
