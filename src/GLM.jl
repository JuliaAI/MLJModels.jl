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
import Distributions
using Parameters

import ..GLM

export NormalRegressor, OLSRegressor, OLS,
        BinaryClassifier

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

# NormalRegressor       --> Probabilistic w Continuous Target
# PoissonRegressor      --> Probabilistic w Count Target
# BernoulliClassifier   --> Probabilistic w Binary target // logit,cauchit,..
# MultinomialClassifier --> Probabilistic w Multiclass target


@with_kw mutable struct NormalRegressor <: MLJBase.Probabilistic
    fit_intercept::Bool      = true
    allowrankdeficient::Bool = false
end

@with_kw mutable struct BinaryClassifier{L<:GLM.Link01} <: MLJBase.Probabilistic
    fit_intercept::Bool = true
    link::L             = GLM.LogitLink()
end

# Short names for convenience here

const OLS = NormalRegressor
const OLSRegressor = OLS
const BC = BinaryClassifier

const GLM_MODELS = Union{<:OLS, <:BC}

####
#### FIT FUNCTIONS
####

function MLJBase.fit(model::OLS, verbosity::Int, X, y)
	# apply the model
	features  = MLJBase.schema(X).names
	Xmatrix   = augment_X(MLJBase.matrix(X), model.fit_intercept)
	fitresult = GLM.glm(Xmatrix, y, Distributions.Normal(), GLM.IdentityLink())
	# form the report
    report    = glm_report(fitresult)
    cache     = nothing
	# return
    return fitresult, cache, report
end

function MLJBase.fit(model::BinaryClassifier, verbosity::Int, X, y)
	# apply the model
	features  = MLJBase.schema(X).names
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
function MLJBase.predict_mean(model::OLS, fitresult, Xnew)
    Xmatrix = augment_X(MLJBase.matrix(Xnew), model.fit_intercept)
    return GLM.predict(fitresult, Xmatrix)
end

function MLJBase.predict_mean(model::BC, (fitresult, _), Xnew)
    Xmatrix = augment_X(MLJBase.matrix(Xnew), model.fit_intercept)
    return GLM.predict(fitresult, Xmatrix)
end

function MLJBase.predict(model::OLS, fitresult, Xnew)
    μ = MLJBase.predict_mean(model, fitresult, Xnew)
    σ̂ = GLM.dispersion(fitresult)
    return [GLM.Normal(μᵢ, σ̂) for μᵢ ∈ μ]
end

function MLJBase.predict(model::BC, (fitresult, decode), Xnew)
	π = MLJBase.predict_mean(model, (fitresult, decode), Xnew)
	return [MLJBase.UnivariateFinite(MLJBase.classes(decode), [1-πᵢ, πᵢ]) for πᵢ in π]
end

# NOTE: predict_mode uses MLJBase's fallback

####
#### METADATA
####

# shared metadata
MLJBase.package_name(::Type{<:GLM_MODELS})     = "GLM"
MLJBase.package_url(::Type{<:GLM_MODELS})      = "https://github.com/JuliaStats/GLM.jl"
# TODO wait for 0.4 to be merged
#MLJBase.package_license(::Type{<:GLM_MODELS})  = "MIT"
MLJBase.package_uuid(::Type{<:GLM_MODELS})     = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MLJBase.is_pure_julia(::Type{<:GLM_MODELS})    = true
MLJBase.is_wrapper(::Type{<:GLM_MODELS})       = false
MLJBase.supports_weights(::Type{<:GLM_MODELS}) = false #  for now

MLJBase.load_path(::Type{<:OLS})      = "MLJModels.GLM_.NormalRegressor"
MLJBase.load_path(::Type{<:BC})       = "MLJModels.GLM_.BinaryClassifier"

using Pkg
if Pkg.installed()["MLJBase"] > v"0.3"
    MLJBase.package_license(::Type{<:GLM_MODELS}) = "MIT"
	MLJBase.input_scitype(::Type{<:OLS})  = MLJBase.Continuous
	MLJBase.target_scitype(::Type{<:OLS}) = MLJBase.Continuous
    MLJBase.input_scitype(::Type{<:BC})   = MLJBase.Continuous
	MLJBase.target_scitype(::Type{<:BC})  = MLJBase.Finite
else
	MLJBase.input_scitype_union(::Type{<:OLS})  = MLJBase.Continuous
	MLJBase.target_scitype_union(::Type{<:OLS}) = MLJBase.Continuous
    MLJBase.input_scitype_union(::Type{<:BC})   = MLJBase.Continuous
	MLJBase.target_scitype_union(::Type{<:BC})  = MLJBase.Finite
end

end # module
