module GLM_

import MLJBase
using Parameters

export OLSRegressor, OLS,
       GLMCountRegressor, GLMCount

import ..GLM

####
#### HELPER FUNCTIONS
####

intercept_X(m::MLJBase.Model, X::Matrix) =
    m.fit_intercept ? hcat(X, ones(eltype(X), size(X, 1), 1)) : X

## TODO: add feature importance curve to report using `features`
glm_report(fitresult) = ( deviance      = GLM.deviance(fitresult),
                          dof_residual  = GLM.dof_residual(fitresult),
                          stderror      = GLM.stderror(fitresult),
                          vcov          = GLM.vcov(fitresult))

####
#### REGRESSION TYPES
####

"""
OLSRegressor

Ordinary Least Square regression provided by the [GLM.jl](https://github.com/JuliaStats/GLM.jl)
package.
"""
@with_kw mutable struct OLSRegressor <: MLJBase.Probabilistic
    fit_intercept::Bool      = true
    allowrankdeficient::Bool = false
end

"""
GLMCountRegressor

Count regression
"""
mutable struct GLMCountRegressor <: MLJBase.Probabilistic
    fit_intercept::Bool
# link
end
GLMCountRegressor(;fit_intercept=true) = GLMCountRegressor(fit_intercept)

# synonyms
const OLS = OLSRegressor
const GLMCount = GLMCountRegressor

const GLMModels = Union{OLS, GLMCount}

####
#### FIT
####

## > OLS

function MLJBase.fit(model::OLS, verbosity::Int, X, y)
    features  = MLJBase.schema(X).names
    Xmatrix   = intercept_X(model, MLJBase.matrix(X))
    fitresult = GLM.lm(Xmatrix, y)
    report    = glm_report(fitresult)
    cache     = nothing
    return fitresult, cache, report
end

## > Count

function MLJBase.fit(model::GLMCount, verbosity::Int, X, y)

    features  = MLJBase.schema(X).names
    Xmatrix   = intercept_X(model, MLJBase.matrix(X))
    fitresult = GLM.glm(Xmatrix, y, GLM.Poisson()) # Log link
    report    = glm_report(fitresult)
    cache     = nothing
    return fitresult, cache, report
end

####
#### FITTED PARAMS
####

function MLJBase.fitted_params(model::GLMModels, fitresult)
    coefs = GLM.coef(fitresult)
    return (coef      = coefs[1:end-Int(model.fit_intercept)],
            intercept = ifelse(model.fit_intercept, coefs[end],
            nothing))
end

####
#### PREDICT FUNCTIONS
####

function MLJBase.predict_mean(model::GLMModels, fitresult, Xnew)
    Xmatrix = intercept_X(model, MLJBase.matrix(Xnew))
    return GLM.predict(fitresult, Xmatrix)
end

function MLJBase.predict(model::OLS, fitresult, Xnew)
    Xmatrix = intercept_X(model, MLJBase.matrix(Xnew))
    μ = GLM.predict(fitresult, Xmatrix)
    σ̂ = GLM.dispersion(fitresult, false)
    return [GLM.Normal(μᵢ, σ̂) for μᵢ ∈ μ]
end

function MLJBase.predict(model::GLMCount, fitresult, Xnew)
    Xmatrix = intercept_X(model, MLJBase.matrix(Xnew))
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
MLJBase.input_scitype_union(::Type{<:OLS})     = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:OLS})     = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:OLS}) = true

MLJBase.load_path(::Type{<:GLMCount})       = "MLJModels.GLM_.GLMCountRegressor"
MLJBase.input_scitype_union(::Type{<:GLMCount})     = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:GLMCount})     = MLJBase.Count
MLJBase.input_is_multivariate(::Type{<:GLMCount}) = true

end # module
