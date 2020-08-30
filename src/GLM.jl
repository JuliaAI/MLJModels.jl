module GLM_

# -------------------------------------------------------------------
# TODO
# - return feature names in the report
# - return feature importance curve to report using `features`
# - handle binomial case properly, needs MLJ API change for weighted
# samples (y/N ~ Be(p) with weights N)
# - handle levels properly (see GLM.jl/issues/240); if feed something
# with levels, the fit will fail.
# - revisit and test Poisson and Negbin regression once there's a clear
# example we can test on (requires handling levels which deps upon GLM)
# - test Logit, Probit etc on Binomial once binomial case is handled
# -------------------------------------------------------------------

import MLJModelInterface
import MLJModelInterface: metadata_pkg, metadata_model,
                          Table, Continuous, Count, Finite, OrderedFactor,
                          Multiclass
const MMI = MLJModelInterface

import Distributions
using Parameters
using Tables

import ..GLM

export LinearRegressor, LinearBinaryClassifier, LinearCountRegressor, GenericGLMModel

##
## DESCRIPTIONS
##

const LR_DESCR = "Linear regressor (OLS) with a Normal model."
const LBC_DESCR = "Linear binary classifier with specified link (e.g. logistic)."
const LCR_DESCR = "Linear count regressor with specified link and distribution (e.g. log link and poisson)."
const GENERIC_GLM_MODEL_DESCR = "A generic GLM model with specified formula, link, and distribution"


###
## Helper functions
###

"""
augment_X(X, b)

Augment the matrix `X` with a column of ones if the intercept is to be fitted (`b=true`), return
`X` otherwise.
"""
function augment_X(X::Matrix, b::Bool)::Matrix
    b && return hcat(X, ones(eltype(X), size(X, 1), 1))
    return X
end


"""
glm_report(fitresult)

Report based on the `fitresult` of a GLM model.
"""
glm_report(fitresult) = ( deviance     = GLM.deviance(fitresult),
                          dof_residual = GLM.dof_residual(fitresult),
                          stderror     = GLM.stderror(fitresult),
                          vcov         = GLM.vcov(fitresult) )

####
#### REGRESSION TYPES
####

# LinearRegressor        --> Probabilistic w Continuous Target
# LinearCountRegressor   --> Probabilistic w Count Target
# LinearBinaryClassifier --> Probabilistic w Binary target // logit,cauchit,..
# MulticlassClassifier   --> Probabilistic w Multiclass target
# GenericGLMModel        --> Probabilistic


@with_kw_noshow mutable struct LinearRegressor <: MMI.Probabilistic
    fit_intercept::Bool      = true
    allowrankdeficient::Bool = false
end

@with_kw_noshow mutable struct LinearBinaryClassifier{L<:GLM.Link01} <: MMI.Probabilistic
    fit_intercept::Bool = true
    link::L             = GLM.LogitLink()
end

@with_kw_noshow mutable struct LinearCountRegressor{D<:Distributions.Distribution,L<:GLM.Link} <: MMI.Probabilistic
    fit_intercept::Bool = true
    distribution::D     = Distributions.Poisson()
    link::L 			= GLM.LogLink()
end

@with_kw_noshow mutable struct GenericGLMModel{F, LA, D, LI} <: MMI.Probabilistic
    formula::F
    label::LA
    distribution::D
    link::LI
end

# Short names for convenience here

const GLM_MODELS = Union{<:LinearRegressor, <:LinearBinaryClassifier, <:LinearCountRegressor}

####
#### FIT FUNCTIONS
####

function MMI.fit(model::LinearRegressor, verbosity::Int, X, y)
    # apply the model
    features  = Tables.schema(X).names
    Xmatrix   = augment_X(MMI.matrix(X), model.fit_intercept)
    fitresult = GLM.glm(Xmatrix, y, Distributions.Normal(), GLM.IdentityLink())
    # form the report
    report    = glm_report(fitresult)
    cache     = nothing
    # return
    return fitresult, cache, report
end

function MMI.fit(model::LinearCountRegressor, verbosity::Int, X, y)
    # apply the model
    features  = Tables.schema(X).names
    Xmatrix   = augment_X(MMI.matrix(X), model.fit_intercept)
    fitresult = GLM.glm(Xmatrix, y, model.distribution, model.link)
    # form the report
    report    = glm_report(fitresult)
    cache     = nothing
    # return
    return fitresult, cache, report
end

function MMI.fit(model::LinearBinaryClassifier, verbosity::Int, X, y)
    # apply the model
    features  = Tables.schema(X).names
    Xmatrix   = augment_X(MMI.matrix(X), model.fit_intercept)
    decode    = y[1]
    y_plain   = MMI.int(y) .- 1 # 0, 1 of type Int
    fitresult = GLM.glm(Xmatrix, y_plain, Distributions.Bernoulli(), model.link)
    # form the report
    report    = glm_report(fitresult)
    cache     = nothing
    # return
    return (fitresult, decode), cache, report
end

function _merge_X_and_y(X, # `X` must be a table
                        y::AbstractVector,
                        label)
    return merge(Tables.columntable(X), NamedTuple{(label,)}((y,)))
end

function MMI.fit(model::GenericGLMModel,
                 verbosity::Int,
                 X, # `X` must be a table
                 y::AbstractVector)
    # apply the model
    data = _merge_X_and_y(X, y, model.label)
    fitresult = GLM.glm(model.formula, data, model.distribution, model.link)
    # form the report
    report    = glm_report(fitresult)
    cache     = nothing
    # return
    return fitresult, cache, report
end

glm_fitresult(::LinearRegressor, fitresult)  = fitresult
glm_fitresult(::LinearCountRegressor, fitresult)  = fitresult
glm_fitresult(::LinearBinaryClassifier, fitresult)  = fitresult[1]
glm_fitresult(::GenericGLMModel, fitresult)  = fitresult

function MMI.fitted_params(model::GLM_MODELS, fitresult)
    coefs = GLM.coef(glm_fitresult(model,fitresult))
    return (coef      = coefs[1:end-Int(model.fit_intercept)],
            intercept = ifelse(model.fit_intercept, coefs[end], nothing))
end

function MMI.fitted_params(model::GenericGLMModel, fitresult)
    return GLM.coef(fitresult)
end

####
#### PREDICT FUNCTIONS
####

# more efficient than MLJBase fallback
function MMI.predict_mean(model::Union{LinearRegressor,<:LinearCountRegressor}, fitresult, Xnew)
    Xmatrix = augment_X(MMI.matrix(Xnew), model.fit_intercept)
    return GLM.predict(fitresult, Xmatrix)
end

function MMI.predict_mean(model::LinearBinaryClassifier, (fitresult, _), Xnew)
    Xmatrix = augment_X(MMI.matrix(Xnew), model.fit_intercept)
    return GLM.predict(fitresult, Xmatrix)
end

function MMI.predict(model::LinearRegressor, fitresult, Xnew)
    μ = MMI.predict_mean(model, fitresult, Xnew)
    σ̂ = GLM.dispersion(fitresult)
    return [GLM.Normal(μᵢ, σ̂) for μᵢ ∈ μ]
end

function MMI.predict(model::LinearCountRegressor, fitresult, Xnew)
    λ = MMI.predict_mean(model, fitresult, Xnew)
    return [GLM.Poisson(λᵢ) for λᵢ ∈ λ]
end

function MMI.predict(model::LinearBinaryClassifier, (fitresult, decode), Xnew)
    π = MMI.predict_mean(model, (fitresult, decode), Xnew)
    return MMI.UnivariateFinite(MMI.classes(decode), π, augment=true)
end

function MMI.predict(model::GenericGLMModel,
                     fitresult,
                     Xnew, # `Xnew` must be a table
                     )
    return GLM.predict(fitresult, Xnew)
end

# NOTE: predict_mode uses MLJBase's fallback

####
#### METADATA
####

# shared metadata
const GLM_REGS = Union{Type{<:LinearRegressor}, Type{<:LinearBinaryClassifier}, Type{<:LinearCountRegressor}, Type{<:GenericGLMModel}}

metadata_pkg.((LinearRegressor, LinearBinaryClassifier, LinearCountRegressor, GenericGLMModel),
    name       = "GLM",
    uuid       = "38e38edf-8417-5370-95a0-9cbb8c7f171a",
    url        = "https://github.com/JuliaStats/GLM.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false
    )

metadata_model(LinearRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Continuous},
    weights = false,
    descr   = LR_DESCR
    )

metadata_model(LinearBinaryClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite{2}},
    weights = false,
    descr   = LBC_DESCR
    )

metadata_model(LinearCountRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Count},
    weights = false,
    descr   = LCR_DESCR
    )

end # module
