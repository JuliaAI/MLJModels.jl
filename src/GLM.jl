module GLM_

import MLJBase

export OLSRegressor, OLS,
       GLMRegressor, GLM

import ..GLM # strange syntax for lazy-loading
import ..Distributions

const LMFitResult  = GLM.LinearModel
const GLMFitResult = GLM.GeneralizedLinearModel

LMFitResult(coefs::Vector, b=nothing) = LMFitResult(coefs, b)
GLMFitResult(coefs::Vector, b=nothing) = GLMFitResult(coefs, b)

mutable struct OLSRegressor <: MLJBase.Deterministic{LMFitResult}
    fit_intercept::Bool
#    allowrankdeficient::Bool
end

OLSRegressor(;fit_intercept=true) = OLSRegressor(fit_intercept)

# shorthand
const OLS = OLSRegressor

mutable struct GLMRegressor <: MLJBase.Deterministic{GLMFitResult}
    fit_intercept::Bool
    distribution::Distributions.Distribution
    link::GLM.Link01
end

function GLMRegressor(;fit_intercept=true
                     , distribution=Binomial()
                     , link=GLM.LogitLink())
    return GLMRegressor(fit_intercept, distribution, link)
end

# shorthand
const GLM = GLMRegressor

####
#### fit/predict (note that OLS and GLM share predict function)
####

function MLJBase.fit(model::OLS, verbosity::Int, X, y::Vector)

    Xmatrix = MLJBase.matrix(X)
    features = MLJBase.schema(X).names
    # append columns of 1 to Xmatrix if we have to fit the intercept
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))

    fitresult = GLM.lm(Xmatrix, y)

    coefs = (fitresult |> coef)

    ## TODO: add feature importance curve to report using `features`
    report = Dict(:coef => coefs[1:end-Int(model.fit_intercept)]
                , :intercept => ifelse(model.fit_intercept, coefs[end], nothing)
                , :deviance => GLM.deviance(fitresult)
                , :dof_residual => GLM.dof_residual(fitresult)
                , :stderror => GLM.stderror(fitresult)
                , :vcov => GLM.vcov(fitresult))
    cache = nothing

    return fitresult, cache, report
end

function MLJBase.fit(model::GLM, verbosity::Int, X, y::Vector)

    Xmatrix  = MLJBase.matrix(X)
    features = MLJBase.schema(X).names
    # append columns of 1 to Xmatrix if we have to fit the intercept
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))

    fitresult = GLM.glm(Xmatrix, y, model.distribution, model.link)

    coefs = (fitresult |> coef)

    ## TODO: add feature importance curve to report using `features`
    report = Dict(:coef => coefs[1:end-Int(model.fit_intercept)]
                , :intercept => ifelse(model.fit_intercept, coefs[end], nothing)
                , :deviance => GLM.deviance(fitresult)
                , :dof_residual => GLM.dof_residual(fitresult)
                , :stderror => GLM.stderror(fitresult)
                , :vcov => GLM.vcov(fitresult))
    cache = nothing

    return fitresult, cache, report
end

function MLJBase.predict(model::Union{OLS, GLM}
                       , fitresult::Union{LMFitResult, GLMFitResult}
                       , Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))
    return GLM.predict(fitresult, Xmatrix)
end

####
#### METADATA
####

# metadata everything
const _GLM_TYPES = Union{Type{<:OLS}, Type{<:GLM}}
MLJBase.package_name(::_GLM_TYPES)  = "GLM"
MLJBase.package_uuid(::_GLM_TYPES)  = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MLJBase.package_url(::_GLM_TYPES)   = "https://github.com/JuliaStats/GLM.jl"
MLJBase.is_pure_julia(::_GLM_TYPES) = :yes

# metadata OLS
MLJBase.load_path(::Type{<:OLS})       = "MLJModels.GLM_.OLS"
MLJBase.input_kinds(::Type{<:OLS})     = [:continuous, ]
MLJBase.output_kind(::Type{<:OLS})     = :continuous
MLJBase.output_quantity(::Type{<:OLS}) = :univariate

# metadata GLM
MLJBase.load_path(::Type{<:GLM})       = "MLJModels.GLM_.GLM"
MLJBase.input_kinds(::Type{<:GLM})     = [:continuous, ]
MLJBase.output_kind(::Type{<:GLM})     = [:binary, :categorical, :count]
MLJBase.output_quantity(::Type{<:GLM}) = :univariate

end # module
