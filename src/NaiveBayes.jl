module NaiveBayes_

export GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier

import MLJBase
using CategoricalArrays
import ..NaiveBayes

mutable struct GaussianNBClassifier{T} <: MLJBase.Probabilistic{Any} where T
    target_type::Type{T}
    c_counts
    c_stats
    gaussians
    n_obs::Int
end

function GaussianNBClassifier(
        ; c_counts
        , c_stats
        , gaussians
        , n_obs)

    m = GaussianNBClassifier(Int, c_counts, c_stats,
        gaussians, n_obs)

    return m
end

function GaussianNBClassifier(
    ; classes
    , n)
    m = NaiveBayes.GaussianNB(classes, n)
    model = GaussianNBClassifier( Int,
        m.c_counts
        ,m.c_stats
        ,m.gaussians
        ,m.n_obs)

    return model
end

function MLJBase.fit(model::GaussianNBClassifier
                , X
                , Y)

    Xmatrix = MLJBase.matrix(X)'
    res = NaiveBayes.GaussianNB(model.c_counts, model.c_stats,
    model.gaussians, model.n_obs)

    res = NaiveBayes.fit(res, Float64.(Xmatrix), Y)
    return res, nothing, nothing
end

function MLJBase.predict(model::GaussianNBClassifier, fitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)'
    return NaiveBayes.predict_logprobs(fitresult, Float64.(Xmatrix)) 
end

mutable struct MultinomialNBClassifier{T} <: MLJBase.Probabilistic{Any} where T
    target_type::Type{T}
    c_counts
    x_counts
    x_totals
    n_obs::Int
end

function MultinomialNBClassifier(
        ; c_counts
        , x_counts
        , x_totals
        , n_obs)

    m = GaussianNBClassifier(Int, c_counts, x_counts,
        x_totals, n_obs)

    return m
end

function GaussianNBClassifier(
    ; classes
    , n)
    m = NaiveBayes.MultinomialNB(classes, n)
    model = MultinomialNBNBClassifier( Int,
        m.c_counts
        ,m.x_counts
        ,m.x_totals
        ,m.n_obs)

    return model
end

function MLJBase.fit(model::MultinomialNBClassifier
                , X
                , Y)

    Xmatrix = MLJBase.matrix(X)'
    res = NaiveBayes.MultinomialNB(model.c_counts, model.x_counts,
    model.x_totals, model.n_obs)

    res = NaiveBayes.fit(res, Float64.(Xmatrix), Y)
    return res, nothing, nothing
end

function MLJBase.predict(model::MultinomialNBClassifier, fitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)'
    return NaiveBayes.predict_logprobs(fitresult, Float64.(Xmatrix)) 
end
    
# metadata:
MLJBase.load_path(::Type{<:GaussianNBClassifier}) = "MLJModels.NaiveBayes_.GaussianNBClassifier" 
MLJBase.package_name(::Type{<:GaussianNBClassifier}) = "GaussianNB"
MLJBase.package_uuid(::Type{<:GaussianNBClassifier}) = "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MLJBase.package_url(::Type{<:GaussianNBClassifier}) = "https://github.com/dfdx/NaiveBayes.jl"
MLJBase.is_pure_julia(::Type{<:GaussianNBClassifier}) = :yes
MLJBase.input_kinds(::Type{<:GaussianNBClassifier}) = [:continuous, ]
MLJBase.output_kind(::Type{<:GaussianNBClassifier}) = :multiclass
MLJBase.output_quantity(::Type{<:GaussianNBClassifier}) = :univariate

MLJBase.load_path(::Type{<:MultinomialNBClassifier}) = "MLJModels.NaiveBayes_.MultinomialNBClassifier" 
MLJBase.package_name(::Type{<:MultinomialNBClassifier}) = "MultinomialNB"
MLJBase.package_uuid(::Type{<:MultinomialNBClassifier}) = "9bbee03b-0db5-5f46-924f-b5c9c21b8c60"
MLJBase.package_url(::Type{<:MultinomialNBClassifier}) = "https://github.com/dfdx/NaiveBayes.jl"
MLJBase.is_pure_julia(::Type{<:MultinomialNBClassifier}) = :yes
MLJBase.input_kinds(::Type{<:MultinomialNBClassifier}) = [:continuous, ]
MLJBase.output_kind(::Type{<:MultinomialNBClassifier}) = :multiclass
MLJBase.output_quantity(::Type{<:MultinomialNBClassifier}) = :univariate

end     #module