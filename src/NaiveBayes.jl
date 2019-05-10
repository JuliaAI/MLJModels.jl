module NaiveBayes_

export GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier

import MLJBase
using CategoricalArrays
import ..NaiveBayes

mutable struct GaussianNBClassifier <: MLJBase.Probabilistic
end

function MLJBase.fit(model::GaussianNBClassifier, verbosity::Int
                , X
                , y)

    Xmatrix = MLJBase.matrix(X)' |> collect
    p = size(Xmatrix, 1)

    yplain = identity.(y) # y as plain Vector
    classes_seen = unique(yplain)

    # initiates dictionaries keyed on classes_seen:
    res = NaiveBayes.GaussianNB(classes_seen, p) 

    fitresult = NaiveBayes.fit(res, Xmatrix, yplain)

    report = NamedTuple{}()
    
    return fitresult, nothing, report
    
end

function MLJBase.fitted_params(model::GaussianNBClassifier, fitresult)
    res = fitresult[1]
    return (c_counts=res.c_counts,
            c_stats=res.c_stats,
            gaussians=res.gaussians,
            n_obs=res.n_obs)
end
    
function MLJBase.predict(model::GaussianNBClassifier, fitresult, Xnew)

    Xmatrix = MLJBase.matrix(Xnew)' |> collect
    n = size(Xmatrix, 2)

    classes_observed, logprobs = NaiveBayes.predict_logprobs(fitresult, Xmatrix)

    # Note that NaiveBayes does not normalize the probabilities.

    probs = exp.(logprobs)
    col_sums = sum(probs, dims=1)
    probs = probs ./ col_sums

    # UnivariateFinite constructor automatically adds unobserved
    # classes with zero probability:
    return [MLJBase.UnivariateFinite(classes_observed, probs[:,i])
            for i in 1:n]
    
end

# MultinomialNBClassifier

mutable struct MultinomialNBClassifier <: MLJBase.Probabilistic
    alpha::Int
end

function MultinomialNBClassifier(; alpha=1)
    m = MultinomialNBClassifier(alpha)
    return m
end

function MLJBase.fit(model::MultinomialNBClassifier, verbosity::Int
                , X
                , y)

    Xmatrix = MLJBase.matrix(X) |> permutedims
    p = size(Xmatrix, 1)
    yplain = identity.(y)
    classes_observed = unique(yplain)

    res = NaiveBayes.MultinomialNB(classes_observed, p ,alpha= model.alpha)
    fitresult = NaiveBayes.fit(res, Xmatrix, yplain)

    report = NamedTuple()
    
    return fitresult, nothing, report
end

function MLJBase.fitted_params(model::MultinomialNBClassifier, fitresult)
    res = fitresult[1]
    return (c_counts=res.c_counts,
            x_counts=res.x_counts,
            x_totals=res.x_totals,
            n_obs=res.n_obs)
end
    
function MLJBase.predict(model::MultinomialNBClassifier, fitresult, Xnew)

    Xmatrix = MLJBase.matrix(Xnew) |> collect |> permutedims
    n = size(Xmatrix, 2)

    # Note that NaiveBayes.predict_logprobs returns probabilities that
    # are not normalized.

    classes_observed, logprobs = NaiveBayes.predict_logprobs(fitresult, Int.(Xmatrix))

    probs = exp.(logprobs)
    col_sums = sum(probs, dims=1)
    probs = probs ./ col_sums

    return [MLJBase.UnivariateFinite(classes_observed, probs[:,i]) for i in 1:n]
end

# metadata:
MLJBase.load_path(::Type{<:GaussianNBClassifier}) = "MLJModels.NaiveBayes_.GaussianNBClassifier"
MLJBase.package_name(::Type{<:GaussianNBClassifier}) = "NaiveBayes"
MLJBase.package_uuid(::Type{<:GaussianNBClassifier}) = "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MLJBase.package_url(::Type{<:GaussianNBClassifier}) = "https://github.com/dfdx/NaiveBayes.jl"
MLJBase.is_pure_julia(::Type{<:GaussianNBClassifier}) = true
MLJBase.input_scitype_union(::Type{<:GaussianNBClassifier}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:GaussianNBClassifier}) = MLJBase.Finite
MLJBase.input_is_multivariate(::Type{<:GaussianNBClassifier}) = true

MLJBase.load_path(::Type{<:MultinomialNBClassifier}) = "MLJModels.NaiveBayes_.MultinomialNBClassifier"
MLJBase.package_name(::Type{<:MultinomialNBClassifier}) = "NaiveBayes"
MLJBase.package_uuid(::Type{<:MultinomialNBClassifier}) = "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MLJBase.package_url(::Type{<:MultinomialNBClassifier}) = "https://github.com/dfdx/NaiveBayes.jl"
MLJBase.is_pure_julia(::Type{<:MultinomialNBClassifier}) = true
MLJBase.input_scitype_union(::Type{<:MultinomialNBClassifier}) = MLJBase.Count
MLJBase.target_scitype_union(::Type{<:MultinomialNBClassifier}) = MLJBase.Finite
MLJBase.input_is_multivariate(::Type{<:MultinomialNBClassifier}) = true

end     #module
