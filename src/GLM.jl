module GLM_

import MLJBase

export OLSRegressor, OLS,
       GLMCountRegressor, GLMCount

import ..GLM

####
#### REGRESSION TYPES
####

mutable struct OLSRegressor <: MLJBase.Probabilistic
    fit_intercept::Bool
# allowrankdeficient::Bool
end

OLSRegressor(;fit_intercept=true) = OLSRegressor(fit_intercept)

mutable struct GLMCountRegressor <: MLJBase.Probabilistic
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

function MLJBase.fit(model::OLS, verbosity::Int, X, y)

    Xmatrix = MLJBase.matrix(X)
    features = MLJBase.schema(X).names
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))

    fitresult = GLM.lm(Xmatrix, y)

    ## TODO: add feature importance curve to report using `features`
    report = (deviance=GLM.deviance(fitresult)
              , dof_residual=GLM.dof_residual(fitresult)
              , stderror=GLM.stderror(fitresult)
              , vcov=GLM.vcov(fitresult))
    cache = nothing

    return fitresult, cache, report
end

function MLJBase.fitted_params(model::OLS, fitresult)
    coefs = GLM.coef(fitresult)
    return (coef=coefs[1:end-Int(model.fit_intercept)],
            intercept=ifelse(model.fit_intercept, coefs[end], nothing))
end

function MLJBase.fit(model::GLMCount, verbosity::Int, X, y)

    Xmatrix = MLJBase.matrix(X)
    features = MLJBase.schema(X).names
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))

    fitresult = GLM.glm(Xmatrix, y, GLM.Poisson()) # Log link

    ## TODO: add feature importance curve to report using `features`
    report = (deviance=GLM.deviance(fitresult)
              , dof_residual=GLM.dof_residual(fitresult)
              , stderror=GLM.stderror(fitresult)
              , vcov=GLM.vcov(fitresult))
    cache = nothing

    return fitresult, cache, report
end

function MLJBase.fitted_params(model::GLMCount, fitresult)
    coefs = GLM.coef(fitresult)
    return (coef=coefs[1:end-Int(model.fit_intercept)],
            intercept=ifelse(model.fit_intercept, coefs[end], nothing))
end

####
#### PREDICT FUNCTIONS
####

function MLJBase.predict_mean(model::Union{OLS, GLMCount}
                            , fitresult
                            , Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))
    return GLM.predict(fitresult, Xmatrix)
end

function MLJBase.predict(model::OLS, fitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))
    μ = GLM.predict(fitresult, Xmatrix)
    σ̂ = GLM.dispersion(fitresult, false)
    return [GLM.Normal(μᵢ, σ̂) for μᵢ ∈ μ]
end

function MLJBase.predict(model::GLMCount, fitresult, Xnew)
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
MLJBase.input_scitype(::Type{<:OLS})     = MLJBase.Table(MLJBase.Continuous)
MLJBase.target_scitype(::Type{<:OLS})     = AbstractVector{MLJBase.Continuous}

MLJBase.load_path(::Type{<:GLMCount})       = "MLJModels.GLM_.GLMCountRegressor"
MLJBase.input_scitype(::Type{<:GLMCount})     = MLJBase.Table(MLJBase.Continuous)
MLJBase.target_scitype(::Type{<:GLMCount})     = AbstractVector{MLJBase.Count}

end # module
