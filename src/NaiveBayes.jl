module NaiveBayes_

export GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier

import MLJBase
using CategoricalArrays
import ..NaiveBayes
#import NaiveBayes

mutable struct GaussianNBClassifier{T} <: MLJBase.Probabilistic{Any} where T
    target_type::Type{T}
end

GaussianNBClassifier(; target_type=Int) = GaussianNBClassifier{target_type}(target_type)

function MLJBase.fit(model::GaussianNBClassifier, verbosity::Int
                , X
                , Y)

    Xmatrix = MLJBase.matrix(X)' |> collect
    p = size(Xmatrix, 1)

    levels_observed = unique(Y)
    levels_all = levels(Y)
    levels_unobserved = filter(levels_all) do L
        !(L in levels_observed)
    end
    levels_all = vcat(levels_observed, levels_unobserved)

    decoder = MLJBase.CategoricalDecoder(Y)
    y = MLJBase.transform(decoder, Y)

    res = NaiveBayes.GaussianNB(levels_observed, p)
    NaiveBayes.fit(res, Xmatrix, y)

    fitresult = (res, levels_all)
    report = Dict{Symbol,Any}()
    report[:c_counts] = res.c_counts
    report[:c_stats] = res.c_stats
    report[:gaussians] = res.gaussians
    report[:n_obs] = res.n_obs
    
    return fitresult, nothing, report
    
end

function MLJBase.predict(model::GaussianNBClassifier, fitresult, Xnew)

    res, levels_all = fitresult

    Xmatrix = MLJBase.matrix(Xnew)' |> collect
    n = size(Xmatrix, 2)

    levels_observed, logprobs = NaiveBayes.predict_logprobs(res, Xmatrix)

    probs = exp.(logprobs)
    col_sums = sum(probs, dims=1)
    probs = probs ./ col_sums

    z = zeros(length(levels_all) - length(levels_observed)) 

    return [MLJBase.UnivariateNominal(levels_all, vcat(probs[:,i], z)) for i in 1:n]
end

# MultinomialNBClassifier

mutable struct MultinomialNBClassifier{T} <: MLJBase.Probabilistic{Any} where T
    target_type::Type{T}
    alpha::Int
end

function MultinomialNBClassifier(;alpha=1)
    m = MultinomialNBClassifier(Int, alpha)
    return m
end

function MLJBase.fit(model::MultinomialNBClassifier, verbosity::Int
                , X
                , Y)

    Xmatrix = MLJBase.matrix(X)' |> collect
    p = size(Xmatrix, 1)
    levels_observed = unique(Y)
    levels_all = MLJBase.levels(Y)
    levels_unobserved = filter(levels_all) do L
        !(L in levels_observed)
    end
    levels_all = vcat(levels_observed, levels_unobserved)

    res = NaiveBayes.MultinomialNB(levels_all, p ,alpha= model.alpha)

    NaiveBayes.fit(res, Xmatrix, Y)

    fitresult = (res, levels_all)

    report = Dict{Symbol,Any}()
    report[:c_counts] = res.c_counts
    report[:x_counts] = res.x_counts
    report[:x_totals] = res.x_totals
    report[:n_obs] = res.n_obs
    
    return fitresult, nothing, report
end

function MLJBase.predict(model::MultinomialNBClassifier, fitresult, Xnew)
    res, levels_all = fitresult
    Xmatrix = MLJBase.matrix(Xnew)' |> collect
    n = size(Xmatrix, 2)
    levels_observed ,logprobs = NaiveBayes.predict_logprobs(res, Int.(Xmatrix))
    probs = exp.(logprobs)
    col_sums = sum(probs, dims=1)
    probs = probs ./ col_sums

    z = zeros(length(levels_all) - length(levels_observed)) 

    return [MLJBase.UnivariateNominal(levels_all, vcat(probs[:,i], z)) for i in 1:n]
end
    
# metadata:
MLJBase.load_path(::Type{<:GaussianNBClassifier}) = "MLJModels.NaiveBayes_.GaussianNBClassifier"
MLJBase.package_name(::Type{<:GaussianNBClassifier}) = "GaussianNB"
MLJBase.package_uuid(::Type{<:GaussianNBClassifier}) = "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MLJBase.package_url(::Type{<:GaussianNBClassifier}) = "https://github.com/dfdx/NaiveBayes.jl"
MLJBase.is_pure_julia(::Type{<:GaussianNBClassifier}) = true
MLJBase.input_scitypes(::Type{<:GaussianNBClassifier}) = MLJBase.Continuous
MLJBase.target_scitype(::Type{<:GaussianNBClassifier}) = MLJBase.Multiclass
MLJBase.input_is_multivariate(::Type{<:GaussianNBClassifier}) = true

MLJBase.load_path(::Type{<:MultinomialNBClassifier}) = "MLJModels.NaiveBayes_.MultinomialNBClassifier"
MLJBase.package_name(::Type{<:MultinomialNBClassifier}) = "MultinomialNB"
MLJBase.package_uuid(::Type{<:MultinomialNBClassifier}) = "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MLJBase.package_url(::Type{<:MultinomialNBClassifier}) = "https://github.com/dfdx/NaiveBayes.jl"
MLJBase.is_pure_julia(::Type{<:MultinomialNBClassifier}) = true
MLJBase.input_scitypes(::Type{<:MultinomialNBClassifier}) = MLJBase.Continuous
MLJBase.target_scitype(::Type{<:MultinomialNBClassifier}) = MLJBase.Multiclass
MLJBase.input_is_multivariate(::Type{<:MultinomialNBClassifier}) = true

end     #module