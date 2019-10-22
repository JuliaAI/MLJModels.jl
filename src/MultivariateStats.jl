module MultivariateStats_

export RidgeRegressor, PCA, KernelPCA, ICA, MulticlassLDA

import MLJBase
import MLJBase: @mlj_model
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

####
#### RIDGE
####

@mlj_model mutable struct RidgeRegressor <: MLJBase.Deterministic
    lambda::Float64 = 0.0::(_ ≥ 0)
end

function MLJBase.fit(model::RidgeRegressor,
                     verbosity::Int,
                     X,
                     y)

    Xmatrix = MLJBase.matrix(X)
    features = Tables.schema(X).names

    weights = MS.ridge(Xmatrix, y, model.lambda)

    coefficients = weights[1:end-1]
    bias = weights[end]

    fitresult = LinearFitresult(coefficients, bias)

    report= nothing
    cache = nothing

    return fitresult, cache, report

end

MLJBase.fitted_params(::RidgeRegressor, fitresult) =
    (coefficients=fitresult.coefficients, bias=fitresult.bias)

function MLJBase.predict(model::RidgeRegressor, fitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return Xmatrix*fitresult.coefficients .+ fitresult.bias
end

# metadata:
MLJBase.load_path(::Type{<:RidgeRegressor}) = "MLJModels.MultivariateStats_.RidgeRegressor"
MLJBase.package_name(::Type{<:RidgeRegressor}) = "MultivariateStats"
MLJBase.package_uuid(::Type{<:RidgeRegressor}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url(::Type{<:RidgeRegressor})  = "https://github.com/JuliaStats/MultivariateStats.jl"
MLJBase.is_pure_julia(::Type{<:RidgeRegressor}) = true
MLJBase.input_scitype(::Type{<:RidgeRegressor}) = Table(Continuous)
MLJBase.target_scitype(::Type{<:RidgeRegressor}) = AbstractVector{Continuous}

####
#### PCA
####

const PCAFitResultType = MS.PCA

mutable struct PCA <: MLJBase.Unsupervised
    maxoutdim::Union{Nothing, Int} # number of PCA components, all if nothing
    method::Symbol  # cov or svd (auto by default, choice based on dims)
    pratio::Float64 # ratio of variances preserved in the principal subspace
    mean::Union{Nothing, Real, Vector{Float64}} # 0 if pre-centered
end

function PCA(; maxoutdim=nothing
             , method=:auto
             , pratio=0.99
             , mean=nothing)

    model = PCA(maxoutdim, method, pratio, mean)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.clean!(model::PCA)
    warning = ""
    if model.maxoutdim isa Int && model.maxoutdim < 1
        warning *= "Need maxoutdim > 1. Resetting maxoutdim=p.\n"
        model.maxoutdim = nothing
    end
    if model.method ∉ [:auto, :cov, :svd]
        warning *= "Unknown method specification. Resetting to method=:auto.\n"
        model.method = :auto
    end
    if !(0.0 < model.pratio <= 1.0)
        warning *= "Need 0 < pratio < 1. Resetting to pratio=0.99.\n"
        model.pratio = 0.99
    end
    if (model.mean isa Real) && !iszero(model.mean)
        warning *= "Need mean to be nothing, zero or a vector." *
                   " Resetting to mean=nothing.\n"
        model.mean = nothing
    end
    return warning
end

function MLJBase.fit(model::PCA
                   , verbosity::Int
                   , X)

    Xarray = MLJBase.matrix(X)
    mindim = minimum(size(Xarray))

    maxoutdim = (model.maxoutdim === nothing) ? mindim : model.maxoutdim

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


function MLJBase.transform(model::PCA
                         , fitresult::PCAFitResultType
                         , X)

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

mutable struct KernelPCA <: MLJBase.Unsupervised
    maxoutdim::Union{Nothing, Int}       # number of KernelPCA components, all if nothing
    kernel::Union{Nothing, Function} # kernel function of 2 vector arguments x and y, returns a scalar value, (x,y)->x'y if nothing
    solver::Union{Nothing, Symbol}   # eig solver, :eig or :eigs, :eig if nothing
    inverse::Union{Nothing, Bool}    # perform calculation for inverse transform for, false if nothing
    β::Union{Nothing, Real}          # Hyperparameter of the ridge regression that learns the inverse transform when inverse is true, 1.0 if nothing
    tol::Union{Nothing, Real}        # Convergence tolerance for eigs solver, 0.0 if nothing
    maxiter::Union{Nothing, Int}     # maximu number of iterations for eigs solver, 300 if nothing
end

function KernelPCA(; maxoutdim=nothing
                   , kernel=(x,y)->x'y
                   , solver=:eig
                   , inverse=false
                   , β=1.0
                   , tol=0.0
                   , maxiter=300)

    model = KernelPCA(maxoutdim, kernel, solver, inverse, β, tol, maxiter)

    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.clean!(model::KernelPCA)
    warning = ""
    if model.maxoutdim isa Int && model.maxoutdim < 1
        warning *= "Need maxoutdim > 1. Resetting maxoutdim=p.\n"
        model.maxoutdim = nothing
    end
    if model.solver ∉ [:eig, :eigs]
        warning *= "Unknown eigen solver. Resetting to sovler=:eig.\n"
        model.solver = :eig
    end
    return warning
end

function MLJBase.fit(model::KernelPCA
                   , verbosity::Int
                   , X)

    Xarray = MLJBase.matrix(X)
    mindim = minimum(size(Xarray))

    maxoutdim = (model.maxoutdim === nothing) ? mindim : model.maxoutdim

    fitresult = MS.fit(MS.KernelPCA, permutedims(Xarray)
                     ; kernel=model.kernel
                     , maxoutdim=maxoutdim
                     , solver=model.solver
                     , inverse=model.inverse
                     , β=model.β
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


function MLJBase.transform(model::KernelPCA
                         , fitresult::KernelPCAFitResultType
                         , X)

    Xarray = MLJBase.matrix(X)
    # X is n x d, need to transpose and copy twice...
    return MLJBase.table(
                permutedims(MS.transform(fitresult, permutedims(Xarray))),
                prototype=X)
end

####
#### ICA
####

const ICAFitResultType = MS.ICA

mutable struct ICA <: MLJBase.Unsupervised
    k::Int # The number of independent components to recover
    alg::Union{Nothing, Symbol} # ICA algorithm (only :fastica is implemented upstream)j
    fun::Union{Nothing, MS.ICAGDeriv{T}} where T<:Real # The approx neg-entropy functor. It can be obtained using the function icagfun, currently accepting :tanh and :gaus functions
    do_whiten::Union{Nothing, Bool} # whether to perform pre-whitening, default true
    maxiter::Union{Nothing, Int}  # Maximum number of iterations, default 100
    tol::Union{Nothing, Real} # convergence tolerance for change in matrix W, default 1e-6
    mean::Union{Nothing, Real, Vector{Float64}} # can be 0 (the input data has already been centralized), nothing (it will be computed), vector (a pre-computed mean vector), or nothing
    winit::Union{Nothing, Matrix{Float64}} # initial guess of a matrix W, which should be an empty matrix (random initialisation) a matrix of size (k, k) (if do_whiten is true), a matrix of size (m, k) (if do_whiten is false), or zeros(0,0)
end

function ICA(k
           ; alg=:fastica
           , fun=MS.icagfun(:tanh)
           , do_whiten=true
           , maxiter=100
           , tol=1.0e-6
           , mean=nothing
           , winit=zeros(0,0))

    model = ICA(k, alg, fun, do_whiten, maxiter, tol, mean, winit)

    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.clean!(model::ICA)
    warning = ""
    if model.alg != :fastica
        warning *= "Unknown ICA algorithm. Resetting to alg=:fastica.\n"
        model.alg = :fastica
    end
    if (model.mean isa Real) && !iszero(model.mean)
        warning *= "Need mean to be nothing, zero or a vector." *
                   " Resetting to mean=nothing.\n"
        model.mean = nothing
    end
    return warning
end

function MLJBase.fit(model::ICA
                   , verbosity::Int
                   , X)

    Xarray = MLJBase.matrix(X)
    m, n = size(Xarray)

    k = (model.k <= min(m, n)) ? min(m, n) : model.k

    fitresult = MS.fit(MS.ICA, permutedims(Xarray), k
                     ; alg=model.alg
                     , fun=model.fun
                     , do_whiten=model.do_whiten
                     , maxiter=model.maxiter
                     , tol=model.tol
                     , mean=model.mean
                     , winit=model.winit)

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
#### METADATA
####

# TODO add license field from ridge when MLJBase 0.4 is used

MLJBase.load_path(::Type{<:PCA})  = "MLJModels.MultivariateStats_.PCA"
MLJBase.package_name(::Type{<:PCA})  = MLJBase.package_name(RidgeRegressor)
MLJBase.package_uuid(::Type{<:PCA})  = MLJBase.package_uuid(RidgeRegressor)
MLJBase.package_url(::Type{<:PCA})  = MLJBase.package_url(RidgeRegressor)
MLJBase.is_pure_julia(::Type{<:PCA}) = true
MLJBase.input_scitype(::Type{<:PCA}) = Table(Continuous)
MLJBase.output_scitype(::Type{<:PCA}) = Table(Continuous)

MLJBase.load_path(::Type{<:KernelPCA})  = "MLJModels.MultivariateStats_.KernelPCA"
MLJBase.package_name(::Type{<:KernelPCA})  = MLJBase.package_name(RidgeRegressor)
MLJBase.package_uuid(::Type{<:KernelPCA})  = MLJBase.package_uuid(RidgeRegressor)
MLJBase.package_url(::Type{<:KernelPCA})  = MLJBase.package_url(RidgeRegressor)
MLJBase.is_pure_julia(::Type{<:KernelPCA}) = true
MLJBase.input_scitype(::Type{<:KernelPCA}) = Table(Continuous)
MLJBase.output_scitype(::Type{<:KernelPCA}) = Table(Continuous)

MLJBase.load_path(::Type{<:ICA})  = "MLJModels.MultivariateStats_.ICA"
MLJBase.package_name(::Type{<:ICA})  = MLJBase.package_name(RidgeRegressor)
MLJBase.package_uuid(::Type{<:ICA})  = MLJBase.package_uuid(RidgeRegressor)
MLJBase.package_url(::Type{<:ICA})  = MLJBase.package_url(RidgeRegressor)
MLJBase.is_pure_julia(::Type{<:ICA}) = true
MLJBase.input_scitype(::Type{<:ICA}) = Table(Continuous)
MLJBase.output_scitype(::Type{<:ICA}) = Table(Continuous)

####
#### MulticlassLDA
####
const MulticlassLDA_Desc=
""" MulticlassLDA:
Learns a projection matrix `W` that transforms training feature matrix `Xtrain` unto a 
lower dimensional space of dimension `out_dim` such that the betweenclass variance is maximized 
relative to the within-class variance. Classification is done by applying bayes rule to 
the transformed test feature matrix `WᵀXtest`                          
"""
const MulticlassLDAFields = """
    ## Keywords

    * `method`    : The choice of methods
        * `:gevd`   : based on generalized eigenvalue decomposition
        * `:whiten` : first derive a whitening transform from Sw and then solve the problem based on 
                      eigenvalue decomposition of the whiten Sb
    * `shrinkage` : Choice of shrinkage parameter for Linear shrinkage covariance estimator using a 
                    Diagonal target matrix.
        * `:None` : No shrinkage. use `SimpleCovariance` estimator instead
        * `:lw`  : select optimal shrinkage using the Ledoit-Wolf formula.
    * `out_dim`   : The output dimension, i.e dimension of the transformed space
    * `regcoef`   : The regularization coefficient. A positive value regcoef * eigmax(Sw) is added to 
                    the diagonal of Sw to improve numerical stability.

    """

"""
    ` LDA( ; method=:gevd, shrinkage=:lw , out_dim=1 , regcoef=1e-6 ) `

$MulticlassLDA_Desc

$MulticlassLDAFields

See also the [package documentation](https://multivariatestatsjl.readthedocs.io/en/latest/lda.html)
"""
 mutable struct MulticlassLDA <: MLJBase.Probabilistic
    method::Symbol 
    shrinkage::Union{Symbol,Real}
    out_dim::Int64 
    regcoef::Float64
end

function MLJBase.clean!(model::MulticlassLDA)
    warning = ""
    if model.method ∉ [:gevd, :whiten]
        warning *= "Unknown method. Resetting method=:gevd.\n"
        model.method = :gevd
    end
    if typeof(model.shrinkage) <: Real && !(0 <= model.shrinkage <= 1)
        warning *= "shrinkage has to be ∈ [0,1]. Resetting to shrinkage=:lw. \n"
        model.shrinkage = :lw
    elseif model.shrinkage ∉ [ :lw, :None ]
        warning *= "Unknown model solver. Resetting to shrinkage=:None.\n"
        model.shrinkage = :lw
    end
    if model.out_dim < 1
        warning *= "Need out_dim ≥ 1 and <=min(no_of_features,no_of_classes). Resetting out_dim=1.\n"
        model.lambda = 1
    end
    if model.regcoef <= 0
        warning *= "Need regcoef > 0 . Resetting regcoef=1e-4.\n"
        model.regcoef = 1e-6
    end
    return warning
end


function MulticlassLDA(; method=:gevd, shrinkage=:lw, out_dim=1, regcoef=1e-6)
    model = MulticlassLDA(method, shrinkage, out_dim, regcoef)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.fit(model::MulticlassLDA, verbosity::Int, X, y)
 
    class_list = MLJBase.classes(y[1]) #class list containing unique entries in y
    
    # NOTE: copy/transpose
    Xmatrix  = MLJBase.matrix(X,transpose=true)
    training_sample_size = size(Xmatrix,2)

    #convert the target array of categoricals (y) to array of integers (yplain)
    #and get the number of classes from yplain
    yplain = MLJBase.int(y)
    no_of_classes = maximum(yplain)

    #check to make sure output dimension isn't too large or negative
    0 < model.out_dim <= min(no_of_classes-1, training_sample_size) ||
     throw(ArgumentError("out_dim must lie within 0 and min(no_of_features,training_sample_size)"))

     model.shrinkage == :None ?
            covestimator ::Union{MS.SimpleCovariance,LinearShrinkage}=MS.SimpleCovariance() :
            covestimator = LinearShrinkage(target=DiagonalCommonVariance(),shrinkage=model.shrinkage)

    
    πk = proportions(yplain) #estimate prior probabilities of each class in target vector y
    cache = nothing
    report = NamedTuple{}()
    core_fitresult=MS.fit(MS.MulticlassLDA, Int(no_of_classes), Xmatrix, Int.(yplain);
                            method = model.method,
                            outdim = model.out_dim,
                            regcoef = model.regcoef,
                            covestimator_within = covestimator,
                            covestimator_between = covestimator)


    fitresult = (class_list, core_fitresult, πk, training_sample_size)
    return fitresult, cache, report
end

function MLJBase.fitted_params(::MulticlassLDA, (class_list, core_fitresult, πk, training_sample_size))
    ## Note The projection matrix that projects data unto
    ## a lower dimensional subspace and whitens the lower dimensional data .
    return (class_means = MS.classmeans(core_fitresult),
            projection_matrix = MS.projection(core_fitresult),
            prior_probabilities = πk)
end

function MLJBase.predict(model::MulticlassLDA , (class_list , core_fitresult , πk , training_sample_size), Xnew)
    
    ##Transpose the Xnew matrix and reduce its dimension
    Xmatrix = MS.transform(core_fitresult ,MLJBase.matrix(Xnew,transpose=true))

    ## Estimated the probabilities of each column in Xmatrix belonging to each class in 
    ## class_list storing the results in a probability_matrix  
    pmeans = MS.transform(core_fitresult, MS.classmeans(core_fitresult))
    class_size = size(pmeans, 2)
    test_sample_size = size(Xmatrix, 2)
    probability_matrix = pairwise(SqEuclidean(), pmeans, Xmatrix, dims = 2)
    probability_matrix .= πk .* exp.(-0.5 .* probability_matrix .*(training_sample_size-class_size))
    probability_matrix ./= sum(probability_matrix,dims=1) 
    
    return [MLJBase.UnivariateFinite(class_list, probability_matrix[:,j]) for j in 1:test_sample_size]
end

# metadata:
MLJBase.metadata_pkg(MulticlassLDA; name = "MultivariateStats", uuid="6f286f6a-111f-5878-ab1e-185364afe411",
                         url = "https://github.com/JuliaStats/MultivariateStats.jl",
                         julia = true, license = "MIT",
                         is_wrapper = true)
                                            
MLJBase.metadata_model(MulticlassLDA; input = Table(Continuous), target = AbstractVector{<:Finite},
                             weights = false,
                            descr = MulticlassLDA_Desc)




end # of module+

