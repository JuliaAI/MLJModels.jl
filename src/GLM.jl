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

import MLJBase
import MLJBase: metadata_pkg, metadata_model
import Distributions
using Parameters
using Tables

import ..GLM

export LinearRegressor, LinearBinaryClassifier, LinearCountRegressor

##
## DESCRIPTIONS
##

const LR_DESCR = "Linear regressor (OLS) with a Normal model."
const LBC_DESCR = "Linear binary classifier with specified link (e.g. logistic)."
const LCR_DESCR = "Linear count regressor with specified link and distribution (e.g. log link and poisson)."


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


@with_kw_noshow mutable struct LinearRegressor <: MLJBase.Probabilistic
    fit_intercept::Bool      = true
    allowrankdeficient::Bool = false
end

@with_kw_noshow mutable struct LinearBinaryClassifier{L<:GLM.Link01} <: MLJBase.Probabilistic
    fit_intercept::Bool = true
    link::L             = GLM.LogitLink()
end

@with_kw_noshow mutable struct LinearCountRegressor{D<:Distributions.Distribution,L<:GLM.Link} <: MLJBase.Probabilistic
    fit_intercept::Bool = true
    distribution::D     = Distributions.Poisson()
    link::L 			= GLM.LogLink()
end

# Short names for convenience here

const GLM_MODELS = Union{<:LinearRegressor, <:LinearBinaryClassifier, <:LinearCountRegressor}

####
#### FIT FUNCTIONS
####

function MLJBase.fit(model::LinearRegressor, verbosity::Int, X, y)
    # apply the model
    features  = Tables.schema(X).names
    Xmatrix   = augment_X(MLJBase.matrix(X), model.fit_intercept)
    fitresult = GLM.glm(Xmatrix, y, Distributions.Normal(), GLM.IdentityLink())
    # form the report
    report    = glm_report(fitresult)
    cache     = nothing
    # return
    return fitresult, cache, report
end

function MLJBase.fit(model::LinearCountRegressor, verbosity::Int, X, y)
    # apply the model
    features  = Tables.schema(X).names
    Xmatrix   = augment_X(MLJBase.matrix(X), model.fit_intercept)
    fitresult = GLM.glm(Xmatrix, y, model.distribution, model.link)
    # form the report
    report    = glm_report(fitresult)
    cache     = nothing
    # return
    return fitresult, cache, report
end

function MLJBase.fit(model::LinearBinaryClassifier, verbosity::Int, X, y)
    # apply the model
    features  = Tables.schema(X).names
    Xmatrix   = augment_X(MLJBase.matrix(X), model.fit_intercept)
    decode    = y[1]
    y_plain   = MLJBase.int(y) .- 1 # 0, 1 of type Int
    fitresult = GLM.glm(Xmatrix, y_plain, Distributions.Bernoulli(), model.link)
    # form the report
    report    = glm_report(fitresult)
    cache     = nothing
    # return
    return (fitresult, decode), cache, report
end

function MLJBase.fitted_params(model::GLM_MODELS, fitresult)
    coefs = GLM.coef(fitresult)
    return (coef      = coefs[1:end-Int(model.fit_intercept)],
            intercept = ifelse(model.fit_intercept, coefs[end], nothing))
end

####
#### PREDICT FUNCTIONS
####

# more efficient than MLJBase fallback
function MLJBase.predict_mean(model::Union{LinearRegressor,<:LinearCountRegressor}, fitresult, Xnew)
    Xmatrix = augment_X(MLJBase.matrix(Xnew), model.fit_intercept)
    return GLM.predict(fitresult, Xmatrix)
end

function MLJBase.predict_mean(model::LinearBinaryClassifier, (fitresult, _), Xnew)
    Xmatrix = augment_X(MLJBase.matrix(Xnew), model.fit_intercept)
    return GLM.predict(fitresult, Xmatrix)
end

function MLJBase.predict(model::LinearRegressor, fitresult, Xnew)
    μ = MLJBase.predict_mean(model, fitresult, Xnew)
    σ̂ = GLM.dispersion(fitresult)
    return [GLM.Normal(μᵢ, σ̂) for μᵢ ∈ μ]
end

function MLJBase.predict(model::LinearCountRegressor, fitresult, Xnew)
    λ = MLJBase.predict_mean(model, fitresult, Xnew)
    return [GLM.Poisson(λᵢ) for λᵢ ∈ λ]
end

function MLJBase.predict(model::LinearBinaryClassifier, (fitresult, decode), Xnew)
    π = MLJBase.predict_mean(model, (fitresult, decode), Xnew)
    return [MLJBase.UnivariateFinite(MLJBase.classes(decode), [1-πᵢ, πᵢ]) for πᵢ in π]
end

# NOTE: predict_mode uses MLJBase's fallback

####
#### METADATA
####

# shared metadata
const GLM_REGS = Union{Type{<:LinearRegressor}, Type{<:LinearBinaryClassifier}, Type{<:LinearCountRegressor}}

metadata_pkg.((LinearRegressor, LinearBinaryClassifier, LinearCountRegressor),
    name="GLM",
    uuid="38e38edf-8417-5370-95a0-9cbb8c7f171a",
    url="https://github.com/JuliaStats/GLM.jl",
    julia=true,
    license="MIT",
    is_wrapper=false
    )

metadata_model(LinearRegressor,
    input=MLJBase.Table(MLJBase.Continuous),
    output=AbstractVector{MLJBase.Continuous},
    weights=false,
    descr=LR_DESCR
    )

metadata_model(LinearBinaryClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{MLJBase.UnivariateFinite},
    weights=false,
    descr=LBC_DESCR
    )

metadata_model(LinearCountRegressor,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{MLJBase.Count},
    weights=false,
    descr=LCR_DESCR
    )

end # module
