module NaiveBayes_

export GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier

import MLJModelInterface
import MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor,
                          Multiclass

const MMI = MLJModelInterface

import ..NaiveBayes

mutable struct GaussianNBClassifier <: MMI.Probabilistic
end

function MMI.fit(model::GaussianNBClassifier, verbosity::Int
                , X
                , y)

    Xmatrix = MMI.matrix(X)' |> collect
    p = size(Xmatrix, 1)

    yplain = Any[y...] # y as Vector
    classes_seen = unique(yplain)

    # initiates dictionaries keyed on classes_seen:
    res = NaiveBayes.GaussianNB(classes_seen, p)

    fitresult = NaiveBayes.fit(res, Xmatrix, yplain)

    report = NamedTuple{}()

    return fitresult, nothing, report

end

function MMI.fitted_params(model::GaussianNBClassifier, fitresult)
    res = fitresult[1]
    return (c_counts=res.c_counts,
            c_stats=res.c_stats,
            gaussians=res.gaussians,
            n_obs=res.n_obs)
end

function MMI.predict(model::GaussianNBClassifier, fitresult, Xnew)

    Xmatrix = MMI.matrix(Xnew)' |> collect
    n = size(Xmatrix, 2)

    classes_observed, logprobs = NaiveBayes.predict_logprobs(fitresult, Xmatrix)
    # Note that NaiveBayes does not normalize the probabilities.

    probs = exp.(logprobs)
    col_sums = sum(probs, dims=1)
    probs = probs ./ col_sums

    # UnivariateFinite constructor automatically adds unobserved
    # classes with zero probability:
    return [MMI.UnivariateFinite([classes_observed...], probs[:,i])
            for i in 1:n]

end

# MultinomialNBClassifier

mutable struct MultinomialNBClassifier <: MMI.Probabilistic
    alpha::Int
end

function MultinomialNBClassifier(; alpha=1)
    m = MultinomialNBClassifier(alpha)
    return m
end

function MMI.fit(model::MultinomialNBClassifier, verbosity::Int
                , X
                , y)

    Xmatrix = MMI.matrix(X) |> permutedims
    p = size(Xmatrix, 1)
    yplain = Any[y...] # ordinary Vector
    classes_observed = unique(yplain)

    res = NaiveBayes.MultinomialNB(classes_observed, p ,alpha= model.alpha)
    fitresult = NaiveBayes.fit(res, Xmatrix, yplain)

    report = NamedTuple()

    return fitresult, nothing, report
end

function MMI.fitted_params(model::MultinomialNBClassifier, fitresult)
    res = fitresult[1]
    return (c_counts=res.c_counts,
            x_counts=res.x_counts,
            x_totals=res.x_totals,
            n_obs=res.n_obs)
end

function MMI.predict(model::MultinomialNBClassifier, fitresult, Xnew)

    Xmatrix = MMI.matrix(Xnew) |> collect |> permutedims
    n = size(Xmatrix, 2)

    # Note that NaiveBayes.predict_logprobs returns probabilities that
    # are not normalized.

    classes_observed, logprobs = NaiveBayes.predict_logprobs(fitresult, Int.(Xmatrix))

    probs = exp.(logprobs)
    col_sums = sum(probs, dims=1)
    probs = probs ./ col_sums

    return [MMI.UnivariateFinite([classes_observed...],
                                     probs[:,i]) for i in 1:n]
end

# metadata:
MMI.load_path(::Type{<:GaussianNBClassifier}) = "MLJModels.NaiveBayes_.GaussianNBClassifier"
MMI.package_name(::Type{<:GaussianNBClassifier}) = "NaiveBayes"
MMI.package_uuid(::Type{<:GaussianNBClassifier}) = "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MMI.package_url(::Type{<:GaussianNBClassifier}) = "https://github.com/dfdx/NaiveBayes.jl"
MMI.is_pure_julia(::Type{<:GaussianNBClassifier}) = true
MMI.input_scitype(::Type{<:GaussianNBClassifier}) = Table(Continuous)
MMI.target_scitype(::Type{<:GaussianNBClassifier}) = AbstractVector{<:Finite}

MMI.load_path(::Type{<:MultinomialNBClassifier}) = "MLJModels.NaiveBayes_.MultinomialNBClassifier"
MMI.package_name(::Type{<:MultinomialNBClassifier}) = "NaiveBayes"
MMI.package_uuid(::Type{<:MultinomialNBClassifier}) = "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MMI.package_url(::Type{<:MultinomialNBClassifier}) = "https://github.com/dfdx/NaiveBayes.jl"
MMI.is_pure_julia(::Type{<:MultinomialNBClassifier}) = true
MMI.input_scitype(::Type{<:MultinomialNBClassifier}) = Table(Count)
MMI.target_scitype(::Type{<:MultinomialNBClassifier}) = AbstractVector{<:Finite}

end     #module
