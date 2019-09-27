module Constant

using ..MLJBase, ..Tables
using Distributions

export ConstantRegressor, ConstantClassifier,
        DeterministicConstantRegressor, DeterministicConstantClassifier

## THE CONSTANT REGRESSOR

"""
    ConstantRegressor(; distribution_type=Distributions.Normal)

A regressor that, for any new input pattern, predicts the univariate
probability distribution best fitting the training target data. Use
`predict_mean` to predict the mean value instead.

"""
struct ConstantRegressor{D} <: MLJBase.Probabilistic
    distribution_type::Type{D}
end
function ConstantRegressor(; distribution_type=Distributions.Normal)
    model = ConstantRegressor(distribution_type)
    message = clean!(model)
    isempty(message) || @warn message
    return model
end

function clean!(model::ConstantRegressor)
    message = ""
    MLJBase.isdistribution(model.distribution_type) ||
        error("$model.distribution_type is not a valid distribution_type.")
    return message
end

function MLJBase.fit(model::ConstantRegressor{D}, verbosity::Int, X, y) where D
    fitresult = Distributions.fit(D, y)

    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end

MLJBase.fitted_params(::ConstantRegressor, fitresult) = (target_distribution=fitresult,)

MLJBase.predict(model::ConstantRegressor, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))
# MLJBase.predict_mean(model::ConstantRegressor, fitresult, Xnew) =
#     fill(Distributions.mean(fitresult), nrows(Xnew))

# metadata:
MLJBase.load_path(::Type{<:ConstantRegressor}) = "MLJModels.ConstantRegressor"
MLJBase.package_name(::Type{<:ConstantRegressor}) = "MLJModels"
MLJBase.package_uuid(::Type{<:ConstantRegressor}) =
    "d491faf4-2d78-11e9-2867-c94bc002c0b7"
MLJBase.package_url(::Type{<:ConstantRegressor}) = "https://github.com/alan-turing-institute/MLJModels.jl"
MLJBase.is_pure_julia(::Type{<:ConstantRegressor}) = true
MLJBase.input_scitype(::Type{<:ConstantRegressor}) = MLJBase.Table(MLJBase.Scientific) # anything goes
MLJBase.target_scitype(::Type{<:ConstantRegressor}) = AbstractVector{MLJBase.Continuous}


## THE CONSTANT DETERMINISTIC REGRESSOR (FOR TESTING)

struct DeterministicConstantRegressor <: MLJBase.Deterministic end

function MLJBase.fit(model::DeterministicConstantRegressor, verbosity::Int, X, y)
    fitresult = mean(y)
    cache = nothing
    report = NamedTuple
    return fitresult, cache, report
end

MLJBase.predict(model::DeterministicConstantRegressor, fitresult, Xnew) = fill(fitresult, nrows(Xnew))

# metadata:
MLJBase.load_path(::Type{<:DeterministicConstantRegressor}) = "MLJModels.DeterministicConstantRegressor"
MLJBase.package_name(::Type{<:DeterministicConstantRegressor}) = MLJBase.package_name(ConstantRegressor)
MLJBase.package_uuid(::Type{<:DeterministicConstantRegressor}) = MLJBase.package_url(ConstantRegressor)
MLJBase.package_url(::Type{<:DeterministicConstantRegressor}) = MLJBase.package_url(ConstantRegressor)
MLJBase.is_pure_julia(::Type{<:DeterministicConstantRegressor}) = true
MLJBase.input_scitype(::Type{<:DeterministicConstantRegressor}) = MLJBase.Table(MLJBase.Scientific) # anything goes
MLJBase.target_scitype(::Type{<:DeterministicConstantRegressor}) = AbstractVector{MLJBase.Continuous}


## THE CONSTANT CLASSIFIER

"""
    ConstantClassifier()

A classifier that, for any new input pattern, `predict`s the
`UnivariateFinite` probability distribution `d` best fitting the
training target data. So, `pdf(d, level)` is the proportion of levels
in the training data coinciding with `level`. Use `predict_mode` to
obtain the training target mode instead.

"""
struct ConstantClassifier <: MLJBase.Probabilistic end

function MLJBase.fit(model::ConstantClassifier,
                 verbosity::Int, X, y)

    fitresult = Distributions.fit(MLJBase.UnivariateFinite, y)

    cache = nothing
    report = NamedTuple

    return fitresult, cache, report

end

MLJBase.fitted_params(::ConstantClassifier, fitresult) = (target_distribution=fitresult,)

function MLJBase.predict(model::ConstantClassifier, fitresult, Xnew)
    return fill(fitresult, nrows(Xnew))
end

# metadata:
MLJBase.load_path(::Type{<:ConstantClassifier}) = "MLJModels.ConstantClassifier"
MLJBase.package_name(::Type{<:ConstantClassifier}) = MLJBase.package_name(ConstantRegressor)
MLJBase.package_uuid(::Type{<:ConstantClassifier}) = MLJBase.package_uuid(ConstantRegressor)
MLJBase.package_url(::Type{<:ConstantClassifier}) = MLJBase.package_url(ConstantRegressor)
MLJBase.is_pure_julia(::Type{<:ConstantClassifier}) = true
MLJBase.input_scitype(::Type{<:ConstantClassifier}) = MLJBase.Table(MLJBase.Scientific) # anything goes
MLJBase.target_scitype(::Type{<:ConstantClassifier}) = AbstractVector{<:MLJBase.Finite}


## DETERMINISTIC CONSTANT CLASSIFIER (FOR TESTING)

struct DeterministicConstantClassifier <: MLJBase.Deterministic end

function MLJBase.fit(model::DeterministicConstantClassifier,
                 verbosity::Int, X, y)

    # dump missing target values and make into a regular array:

    fitresult = mode(skipmissing(y)|>collect) # a CategoricalValue or CategoricalString

    cache = nothing
    report = NamedTuple()

    return fitresult, cache, report

end

function MLJBase.predict(model::DeterministicConstantClassifier, fitresult, Xnew)
    n = nrows(Xnew)
    yhat = fill(fitresult, n)
    return yhat
end

# metadata:
MLJBase.load_path(::Type{<:DeterministicConstantClassifier}) = "MLJModels.DeterministicConstantClassifier"
MLJBase.package_name(::Type{<:DeterministicConstantClassifier}) = MLJBase.package_name(ConstantRegressor)
MLJBase.package_uuid(::Type{<:DeterministicConstantClassifier}) = MLJBase.package_uuid(ConstantRegressor)
MLJBase.package_url(::Type{<:DeterministicConstantClassifier}) = MLJBase.package_url(ConstantRegressor)
MLJBase.is_pure_julia(::Type{<:DeterministicConstantClassifier}) = true
MLJBase.input_scitype(::Type{<:DeterministicConstantClassifier}) = MLJBase.Table(MLJBase.Scientific) # anything goes
MLJBase.target_scitype(::Type{<:DeterministicConstantClassifier}) = AbstractVector{<:MLJBase.Finite}

end

using MLJModels.Constant
