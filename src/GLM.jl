module GLM_

import MLJBase

export OLSRegressor, OLS,
       GLMCountRegressor, GLMCount

import ..GLM

const LMFitResult  = GLM.LinearModel
const GLMFitResult = GLM.GeneralizedLinearModel

LMFitResult(coefs::Vector, b=nothing) = LMFitResult(coefs, b)

####
#### REGRESSION TYPES
####

mutable struct OLSRegressor <: MLJBase.Probabilistic{LMFitResult}
    fit_intercept::Bool
# allowrankdeficient::Bool
end

OLSRegressor(;fit_intercept=true) = OLSRegressor(fit_intercept)

mutable struct GLMCountRegressor <: MLJBase.Probabilistic{GLMFitResult}
    fit_intercept::Bool
# link
end

GLMCountRegressor(;fit_intercept=true) = GLMCountRegressor(fit_intercept)

# synonyms
const OLS = OLSRegressor
const GLMCount = GLMCountRegressor

####
#### FIT FUNCTIONS
####

function MLJBase.fit(model::OLS, verbosity::Int, X, y::Vector)

    Xmatrix = MLJBase.matrix(X)
    features = MLJBase.schema(X).names
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))

    fitresult = GLM.lm(Xmatrix, y)

    coefs = GLM.coef(fitresult)

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

function MLJBase.fit(model::GLMCount, verbosity::Int, X, y::Vector)

    Xmatrix = MLJBase.matrix(X)
    features = MLJBase.schema(X).names
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))

    fitresult = GLM.glm(Xmatrix, y, GLM.Poisson()) # Log link

    coefs = GLM.coef(fitresult)

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

####
#### PREDICT FUNCTIONS
####

function MLJBase.predict_mean(model::Union{OLS, GLMCount}
                            , fitresult::Union{LMFitResult, GLMFitResult}
                            , Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))
    return GLM.predict(fitresult, Xmatrix)
end

function MLJBase.predict(model::OLS, fitresult::LMFitResult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))
    μ = GLM.predict(fitresult, Xmatrix)
    σ̂ = GLM.dispersion(fitresult, false)
    return [GLM.Normal(μᵢ, σ̂) for μᵢ ∈ μ]
end

function MLJBase.predict(model::GLMCount, fitresult::GLMFitResult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))
    λ = GLM.predict(fitresult, Xmatrix)
    return [GLM.Poisson(λᵢ) for λᵢ ∈ λ]
end

####
#### METADATA
####

# shared metadata
const GLM_REGS = Union{Type{<:OLS}, Type{<:GLMCount}}
MLJBase.package_name(::GLM_REGS)  = "GLM"
MLJBase.package_uuid(::GLM_REGS)  = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MLJBase.package_url(::GLM_REGS)   = "https://github.com/JuliaStats/GLM.jl"
MLJBase.is_pure_julia(::GLM_REGS) = true

MLJBase.load_path(::Type{<:OLS})       = "MLJModels.GLM_.OLSRegressor"
MLJBase.input_scitypes(::Type{<:OLS})     = MLJBase.Continuous
MLJBase.target_scitype(::Type{<:OLS})     = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:OLS}) = true

MLJBase.load_path(::Type{<:GLMCount})       = "MLJModels.GLM_.GLMCountRegressor"
MLJBase.input_scitypes(::Type{<:GLMCount})     = MLJBase.Continuous
MLJBase.target_scitype(::Type{<:GLMCount})     = MLJBase.Count
MLJBase.input_is_multivariate(::Type{<:GLMCount}) = true

end # module
