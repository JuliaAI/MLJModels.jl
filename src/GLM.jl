module GLM_

# -------------------------------------------------------------------
# TODO
# - return feature names in the report
# - handle binomial case properly, needs MLJ API change for weighted
# samples (y/N ~ Be(p) with weights N)
# - handle levels properly (see GLM.jl/issues/240); if feed something
# with levels, the fit will fail.
# - revisit and test Poisson and Negbin regrssion once there's a clear
# example we can test on (requires handling levels which deps upon GLM)
# - test Logit, Probit etc on Binomial once binomial case is handled
# -------------------------------------------------------------------

import MLJBase
import Distributions
using DocStringExtensions

export GLMRegressor,
        OLSRegressor,
        PoissonRegressor,
        LogitRegressor, LogisticRegressor,
        ProbitRegressor,
        CauchitRegressor,
        CloglogRegressor,
        NegativeBinomialRegressor

import ..GLM

####
#### HELPER FUNCTIONS
####

"""
$SIGNATURES

Augment the matrix X with a column of ones if the intercept is to be fitted.
"""
augment_X(m::MLJBase.Model, X::Matrix)::Matrix =
    m.fit_intercept ? hcat(X, ones(Int, size(X, 1), 1)) : X

## TODO: add feature importance curve to report using `features`
"""
$SIGNATURES

Report based on the fitresult of a GLM model.
"""
glm_report(fitresult) = ( deviance     = GLM.deviance(fitresult),
                          dof_residual = GLM.dof_residual(fitresult),
                          stderror     = GLM.stderror(fitresult),
                          vcov         = GLM.vcov(fitresult) )

####
#### REGRESSION TYPES
####

"""
GLMRegressor

Generalized Linear model corresponding to:

``y ∼ D(θ),``
``g(θ) = Xβ``

where

* `y` is the observed response vector
* `D(θ)` is a parametric distribution (typically in the exponential family)
* `g` is the link function
* `X` is the design matrix (possibly with a column of one if an intercept is to be fitted)
* `β` is the vector of coefficients
"""
mutable struct GLMRegressor <: MLJBase.Probabilistic
    distr::Distributions.Distribution
    link::GLM.Link
    fit_intercept::Bool
    allowrankdeficient::Bool # OLS only
end

"""
$SIGNATURES

Model for continuous regression with

``y ∼ N(μ, σ²)``

where `μ=Xβ` and `N` denotes a normal distribution.
See also [`GLMRegressor`](@ref).
"""
OLSRegressor(; fit_intercept=true, allowrankdeficient=false) =
    GLMRegressor(Distributions.Normal(), GLM.IdentityLink(), fit_intercept, allowrankdeficient)

"""
$SIGNATURES

Model for count regression with

``y ∼ Poi(λ)``

where `log(λ) = Xβ` and `Poi` denotes a Poisson distribution.
See also [`GLMRegressor`](@ref).
"""
PoissonRegressor(; fit_intercept=true) =
    GLMRegressor(Distributions.Poisson(), GLM.LogLink(), fit_intercept, false)

"""
$SIGNATURES

Model for count regression with

``y ∼ NB(λ)``

where `λ = Xβ` and `NB` denotes a Negative Binomial distribution.
See also [`GLMRegressor`](@ref).
"""
NegativeBinomialRegressor(; fit_intercept=true, r=1.) =
    GLMRegressor(Distributions.NegativeBinomial(r), GLM.LogLink(), fit_intercept, false)

"""
$SIGNATURES

Model for bernoulli (binary) and binomial regression with logit link:

``y ∼ Be(p)`` or ``y ∼ Bin(n, p)``

where `logit(p) = Xβ`.
See also [`GLMRegressor`](@ref), [`ProbitRegressor`](@ref), [`CauchitRegressor`](@ref),
[`CloglogRegressor`](@ref).
"""
LogitRegressor(; fit_intercept=true, distr=Distributions.Bernoulli()) =
    GLMRegressor(distr, GLM.LogitLink(), fit_intercept, false)
LogisticRegressor = LogitRegressor

"""
$SIGNATURES

Model for bernoulli (binary) and binomial regression with probit link:

``y ∼ Be(p)`` or ``y ∼ Bin(n, p)``

where `probit(p) = Xβ`.
See also [`GLMRegressor`](@ref), [`LogitRegressor`](@ref), [`CauchitRegressor`](@ref),
[`CloglogRegressor`](@ref).
"""
ProbitRegressor(; fit_intercept=true, distr=Distributions.Bernoulli()) =
    GLMRegressor(distr, GLM.ProbitLink(), fit_intercept, false)

"""
$SIGNATURES

Model for bernoulli (binary) and binomial regression with cauchit link:

``y ∼ Be(p)`` or ``y ∼ Bin(n, p)``

where `cauchit(p) = Xβ`.
See also [`GLMRegressor`](@ref), [`ProbitRegressor`](@ref), [`LogitRegressor`](@ref),
[`CloglogRegressor`](@ref).
"""
CauchitRegressor(; fit_intercept=true, distr=Distributions.Bernoulli()) =
    GLMRegressor(distr, GLM.CauchitLink(), fit_intercept, false)

"""
$SIGNATURES

Model for bernoulli (binary) and binomial regression with complentary log log link:

``y ∼ Be(p)`` or ``y ∼ Bin(n, p)``

where `cloglog(p) = Xβ`.
See also [`GLMRegressor`](@ref), [`ProbitRegressor`](@ref), [`LogitRegressor`](@ref),
[`CauchitRegressor`](@ref).
"""
CloglogRegressor(; fit_intercept=true, distr=Distributions.Bernoulli()) =
    GLMRegressor(distr, GLM.CloglogLink(), fit_intercept, false)

####
#### FIT
####

function MLJBase.fit(model::GLMRegressor, verbosity::Int, X, y)
    features  = MLJBase.schema(X).names
    Xmatrix   = augment_X(model, MLJBase.matrix(X))
    fitresult = GLM.glm(Xmatrix, y, model.distr, model.link)
    report    = glm_report(fitresult)
    cache     = nothing
    return fitresult, cache, report
end

function MLJBase.fitted_params(model::GLMRegressor, fitresult)
    coefs = GLM.coef(fitresult)
    return (coef      = coefs[1:end-Int(model.fit_intercept)],
            intercept = ifelse(model.fit_intercept, coefs[end],
            nothing))
end

####
#### PREDICT
####

function MLJBase.predict_mean(model::GLMRegressor, fitresult, Xnew)
    Xmatrix = augment_X(model, MLJBase.matrix(Xnew))
    return GLM.predict(fitresult, Xmatrix)
end

function MLJBase.predict(model::GLMRegressor, fitresult, Xnew)
    Xmatrix = augment_X(model, MLJBase.matrix(Xnew))

    # Ordinary least squares
    if isa(model.distr, Distributions.Normal) && isa(model.link, GLM.IdentityLink)
        μ = GLM.predict(fitresult, Xmatrix)
        σ̂ = GLM.dispersion(fitresult, false)
        return GLM.Normal.(μ, σ̂)

    # Poisson regression
    elseif isa(model.distr, Distributions.Poisson) && isa(model.link, GLM.LogLink)
        λ = GLM.predict(fitresult, Xmatrix)
        return GLM.Poisson.(λ)

    elseif isa(model.distr, Union{Distributions.Bernoulli,Distributions.Binomial}) &&
           isa(model.link, Union{GLM.LogitLink,GLM.ProbitLink,GLM.CauchitLink,GLM.CloglogLink})
        π = GLM.predict(fitresult, Xmatrix)
        if isa(model.distr, Distributions.Bernoulli)
            return GLM.Bernoulli.(π)
        else
            error("Binomial regression not yet supported")
        end
    end
end

####
#### METADATA
####

# shared metadata
MLJBase.package_name(::Type{<:GLMRegressor})  = "GLM"
MLJBase.package_uuid(::Type{<:GLMRegressor})  = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MLJBase.package_url(::Type{<:GLMRegressor})   = "https://github.com/JuliaStats/GLM.jl"
MLJBase.is_pure_julia(::Type{<:GLMRegressor}) = true

MLJBase.load_path(::Type{<:GLMRegressor})            = "MLJModels.GLM_.GLMRegressor"
MLJBase.input_scitype_union(::Type{<:GLMRegressor})   = Union{MLJBase.Continuous, MLJBase.Count}
MLJBase.target_scitype_union(::Type{<:GLMRegressor})  = Union{MLJBase.Continuous, MLJBase.Count}
MLJBase.input_is_multivariate(::Type{<:GLMRegressor}) = true

end # module
