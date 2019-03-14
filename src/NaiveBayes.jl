module NaiveBayes_

export GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier

import MLJBase
using CategoricalArrays
import ..NaiveBayes

mutable struct GaussianNBClassifier <: MLJBase.Probabilistic{Any}
end

function MLJBase.fit(model::GaussianNBClassifier, verbosity::Int
                , X
                , Y)

    Xmatrix = MLJBase.matrix(X)' |> collect
    p = size(Xmatrix, 1)

    levels_observed = unique(Y)
    levels_all = levels(Y)

    decoder = MLJBase.CategoricalDecoder(Y)
    y = MLJBase.transform(decoder, Y)

    res = NaiveBayes.GaussianNB(levels_observed, p)
    res = NaiveBayes.fit(res, Xmatrix, y)

    fitresult = (res, levels_all)

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

    res, levels_all = fitresult

    Xmatrix = MLJBase.matrix(Xnew)' |> collect
    n = size(Xmatrix, 2)

    levels_observed, logprobs = NaiveBayes.predict_logprobs(res, Xmatrix)

    # Note that NaiveBayes.predict_logprobs returns the
    # levels_observed in possibly different order to the
    # levels_observed passed to NaiveBayes.fit. And the probabilities
    # are not normalized.

    # re-order levels_all, so observed come first:
    levels_unobserved = filter(levels_all) do L
        !(L in levels_observed)
    end
    levels_all = vcat(levels_observed, levels_unobserved)

    probs = exp.(logprobs)
    col_sums = sum(probs, dims=1)
    probs = probs ./ col_sums

    z = zeros(length(levels_all) - length(levels_observed)) 

    return [MLJBase.UnivariateNominal(levels_all, vcat(probs[:,i], z)) for i in 1:n]
    
end

# MultinomialNBClassifier

mutable struct MultinomialNBClassifier <: MLJBase.Probabilistic{Any} where T
    alpha::Int
end

function MultinomialNBClassifier(; alpha=1)
    m = MultinomialNBClassifier(alpha)
    return m
end

function MLJBase.fit(model::MultinomialNBClassifier, verbosity::Int
                , X
                , Y)

    Xmatrix = MLJBase.matrix(X) |> permutedims
    p = size(Xmatrix, 1)
    levels_observed = unique(Y)
    levels_all = MLJBase.levels(Y)

    decoder = MLJBase.CategoricalDecoder(Y)
    y = MLJBase.transform(decoder, Y)

    res = NaiveBayes.MultinomialNB(levels_all, p ,alpha= model.alpha)
    res = NaiveBayes.fit(res, Xmatrix, y)

    fitresult = (res, levels_all)

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
    res, levels_all = fitresult
    Xmatrix = MLJBase.matrix(Xnew) |> collect |> permutedims
    n = size(Xmatrix, 2)

    # Note that NaiveBayes.predict_logprobs returns the
    # levels_observed in possibly different order to the
    # levels_observed passed to NaiveBayes.fit. And the probabilities
    # are not normalized.

    levels_observed ,logprobs = NaiveBayes.predict_logprobs(res, Int.(Xmatrix))
    levels_unobserved = filter(levels_all) do L
        !(L in levels_observed)
    end
    levels_all = vcat(levels_observed, levels_unobserved)

    probs = exp.(logprobs)
    col_sums = sum(probs, dims=1)
    probs = probs ./ col_sums

    z = zeros(length(levels_all) - length(levels_observed)) 

    return [MLJBase.UnivariateNominal(levels_all, vcat(probs[:,i], z)) for i in 1:n]
end

# metadata:
MLJBase.load_path(::Type{<:GaussianNBClassifier}) = "MLJModels.NaiveBayes_.GaussianNBClassifier"
MLJBase.package_name(::Type{<:GaussianNBClassifier}) = "NaiveBayes"
MLJBase.package_uuid(::Type{<:GaussianNBClassifier}) = "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MLJBase.package_url(::Type{<:GaussianNBClassifier}) = "https://github.com/dfdx/NaiveBayes.jl"
MLJBase.is_pure_julia(::Type{<:GaussianNBClassifier}) = true
MLJBase.input_scitypes(::Type{<:GaussianNBClassifier}) = MLJBase.Continuous
MLJBase.target_scitype(::Type{<:GaussianNBClassifier}) = Union{MLJBase.Multiclass,MLJBase.FiniteOrderedFactor}
MLJBase.input_is_multivariate(::Type{<:GaussianNBClassifier}) = true

MLJBase.load_path(::Type{<:MultinomialNBClassifier}) = "MLJModels.NaiveBayes_.MultinomialNBClassifier"
MLJBase.package_name(::Type{<:MultinomialNBClassifier}) = "NaiveBayes"
MLJBase.package_uuid(::Type{<:MultinomialNBClassifier}) = "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MLJBase.package_url(::Type{<:MultinomialNBClassifier}) = "https://github.com/dfdx/NaiveBayes.jl"
MLJBase.is_pure_julia(::Type{<:MultinomialNBClassifier}) = true
MLJBase.input_scitypes(::Type{<:MultinomialNBClassifier}) = MLJBase.Count
MLJBase.target_scitype(::Type{<:MultinomialNBClassifier}) = Union{MLJBase.Multiclass,MLJBase.FiniteOrderedFactor}
MLJBase.input_is_multivariate(::Type{<:MultinomialNBClassifier}) = true

end     #module
