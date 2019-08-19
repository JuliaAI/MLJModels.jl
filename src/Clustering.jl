# NOTE: there's a `kmeans!` function that updates centers, maybe a candidate
# for the `update` machinery. Same for `kmedoids!`
# NOTE: if the prediction is done on the original array, just the assignment
# should be returned, unclear what's the best way of doing this.

module Clustering_

export KMeans
export KMedoids

import MLJBase

import ..Clustering # strange sytax for lazy-loading

using Distances
using LinearAlgebra: norm

const C = Clustering

# ----------------------------------

mutable struct KMeans{M<:SemiMetric} <: MLJBase.Unsupervised
    k::Int
    metric::M
end

mutable struct KMedoids{M<:SemiMetric} <: MLJBase.Unsupervised
    k::Int
    metric::M
end

const CM = Union{<:KMeans, <:KMedoids}

function MLJBase.clean!(model::CM)
    warning = ""
    if model.k < 2
        warning *= "Need k >= 2. Resetting k=2.\n"
        model.k = 2
    end
    return warning
end

####
#### KMEANS: constructor, fit, transform and predict
####

function KMeans(; k=3, metric=SqEuclidean())
    model = KMeans(k, metric)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.fit(model::KMeans
                   , verbosity::Int
                   , X)

    Xarray = MLJBase.matrix(X)

    # NOTE see https://github.com/JuliaStats/Clustering.jl/issues/136
    # this has been updated but only on #master, in the future replace permutedims
    # with transpose (lazy)
    result    = C.kmeans(permutedims(Xarray), model.k; distance=model.metric)
    fitresult = result.centers # centers (p x k)
    cache     = nothing
    report    = (assignments=result.assignments,) # size n

    return fitresult, cache, report
end

MLJBase.fitted_params(::KMeans, fitresult) = (centers=fitresult,)

function MLJBase.transform(model::KMeans
                         , fitresult
                         , X)

    Xarray = MLJBase.matrix(X)
    (n, p), k = size(Xarray), model.k
    # pairwise distance from samples to centers
    X̃ = pairwise(model.metric, transpose(Xarray), fitresult, dims=2)
    return MLJBase.table(X̃, prototype=X)
end

####
#### KMEDOIDS: constructor, fit and predict
#### NOTE there is no transform in the sense of kmeans
####

function KMedoids(; k=3, metric=SqEuclidean())
    model = KMedoids(k, metric)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.fit(model::KMedoids
                   , verbosity::Int
                   , X)

    Xarray = MLJBase.matrix(X)
    # cost matrix: all the pairwise distances
    Carray = pairwise(model.metric, transpose(Xarray), dims=2) # n x n

    result    = C.kmedoids(Carray, model.k)
    fitresult = permutedims(view(Xarray, result.medoids, :)) # medoids (p x k)
    cache     = nothing
    report    = (assignments=result.assignments,) # size n

    return fitresult, cache, report
end

MLJBase.fitted_params(::KMedoids, fitresult) = (medoids=fitresult,)


function MLJBase.transform(model::KMedoids
                         , fitresult
                         , X)

    Xarray = MLJBase.matrix(X)
    (n, p), k = size(Xarray), model.k
    # pairwise distance from samples to medoids
    X̃ = pairwise(model.metric, transpose(Xarray), fitresult, dims=2)
    return MLJBase.table(X̃, prototype=X)
end

####
#### Predict methods
####

function MLJBase.predict(model::Union{KMeans,KMedoids}, fitresult, Xnew)

    Xarray = MLJBase.matrix(Xnew)
    (n, p), k = size(Xarray), model.k

    pred = zeros(Int, n)
    @inbounds for i ∈ 1:n
        minv = Inf
        for j ∈ 1:k
            curv    = evaluate(model.metric,
                                  view(Xarray, i, :), view(fitresult, :, j))
            P       = curv < minv
            pred[i] =    j * P + pred[i] * !P # if P is true --> j
            minv    = curv * P +    minv * !P # if P is true --> curvalue
        end
    end
    return pred
end

####
#### METADATA
####

MLJBase.package_url(::Type{<:CM}) = "https://github.com/JuliaStats/Clustering.jl"
MLJBase.package_name(::Type{<:CM}) = "Clustering"
MLJBase.package_uuid(::Type{<:CM}) = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
MLJBase.is_pure_julia(::Type{<:CM}) = true

MLJBase.load_path(::Type{<:KMeans}) = "MLJModels.Clustering_.KMeans" # lazy-loaded from MLJ

MLJBase.load_path(::Type{<:KMedoids}) = "MLJModels.Clustering_.KMedoids" # lazy-loaded from MLJ
MLJBase.package_url(::Type{<:KMedoids}) = MLJBase.package_url(KMeans)
MLJBase.package_name(::Type{<:KMedoids}) = MLJBase.package_name(KMeans)
MLJBase.package_uuid(::Type{<:KMedoids}) = MLJBase.package_uuid(KMeans)
MLJBase.is_pure_julia(::Type{<:KMedoids}) = true

using Pkg
if Pkg.installed()["MLJBase"] > v"0.3"
    MLJBase.package_license(::Type{<:CM})  = "MIT"
    MLJBase.input_scitype(::Type{<:KMeans}) = MLJBase.Continuous
    MLJBase.output_scitype(::Type{<:KMeans}) = MLJBase.Continuous
    MLJBase.input_scitype(::Type{<:KMedoids}) = MLJBase.Continuous
    MLJBase.output_scitype(::Type{<:KMedoids}) = MLJBase.Continuous
else
    MLJBase.input_scitype_union(::Type{<:KMeans}) = MLJBase.Continuous
    MLJBase.output_scitype_union(::Type{<:KMeans}) = MLJBase.Continuous
    MLJBase.input_scitype_union(::Type{<:KMedoids}) = MLJBase.Continuous
    MLJBase.output_scitype_union(::Type{<:KMedoids}) = MLJBase.Continuous
end

end # module
