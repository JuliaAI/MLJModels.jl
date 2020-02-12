# NOTE: there's a `kmeans!` function that updates centers, maybe a candidate
# for the `update` machinery. Same for `kmedoids!`
# NOTE: if the prediction is done on the original array, just the assignment
# should be returned, unclear what's the best way of doing this.

module Clustering_

import MLJModelInterface
import MLJModelInterface: @mlj_model, metadata_pkg, metadata_model,
                          Table, Continuous, Count, Finite, OrderedFactor,
                          Multiclass

const MMI = MLJModelInterface

import ..Clustering # strange sytax for lazy-loading

using Distances
using LinearAlgebra: norm
using CategoricalArrays

const C = Clustering

const KMeansDescription =
    """
    K-Means algorithm: find K centroids corresponding to K clusters in the data.
    """

const KMedoidsDescription =
    """
    K-Medoids algorithm: find K centroids corresponding to K clusters in the data.
    Unlike K-Means, the centroids are found among data points themselves."
    """

const KMFields =
    """
    ## Keywords

    * `k=3`     : number of centroids
    * `metric`  : distance metric to use
    """

"""
KMeans(; kwargs...)

$KMeansDescription

$KMFields

See also the [package documentation](http://juliastats.github.io/Clustering.jl/latest/kmeans.html).
"""
@mlj_model mutable struct KMeans <: MMI.Unsupervised
    k::Int = 3::(_ ≥ 2)
    metric::SemiMetric = SqEuclidean()
end

"""
KMedoids(; kwargs...)

$KMedoidsDescription

$KMFields

See also the [package documentation](http://juliastats.github.io/Clustering.jl/latest/kmedoids.html).
"""
@mlj_model mutable struct KMedoids <: MMI.Unsupervised
    k::Int = 3::(_ ≥ 2)
    metric::SemiMetric = SqEuclidean()
end

function MMI.fit(model::KMeans
                   , verbosity::Int
                   , X)
    # NOTE: using transpose here to get a LinearAlgebra.Transpose object which Kmeans can handle
    Xarray = transpose(MMI.matrix(X))

    result    = C.kmeans(Xarray, model.k; distance=model.metric)
    cluster_labels = MMI.categorical(1:model.k)
    fitresult = (result.centers, cluster_labels) # centers (p x k)
    cache     = nothing
    report    = (assignments=result.assignments, # size n
                 cluster_labels=cluster_labels)

    return fitresult, cache, report
end

MMI.fitted_params(::KMeans, fitresult) = (centers=fitresult[1],)

function MMI.transform(model::KMeans
                         , fitresult
                         , X)
    # pairwise distance from samples to centers
    X̃ = pairwise(model.metric, transpose(MMI.matrix(X)),
                 fitresult[1], dims=2)
    return MMI.table(X̃, prototype=X)
end

function MMI.fit(model::KMedoids
                   , verbosity::Int
                   , X)

    # NOTE: using transpose=true will materialize the transpose (~ permutedims), KMedoids
    # does not yet accept LinearAlgebra.Transpose
    Xarray = MMI.matrix(X, transpose=true)
    # cost matrix: all the pairwise distances
    Carray    = pairwise(model.metric, Xarray, dims=2) # n x n
    result    = C.kmedoids(Carray, model.k)
    cluster_labels = MMI.categorical(1:model.k)
    fitresult = (view(Xarray, :, result.medoids), cluster_labels) # medoids
    cache     = nothing
    report    = (assignments=result.assignments, # size n
                 cluster_labels=cluster_labels)

    return fitresult, cache, report
end

MMI.fitted_params(::KMedoids, fitresult) = (medoids=fitresult[1],)

function MMI.transform(model::KMedoids
                         , fitresult
                         , X)
    # pairwise distance from samples to medoids
                 X̃ = pairwise(model.metric, MMI.matrix(X, transpose=true),
                              fitresult[1], dims=2)
    return MMI.table(X̃, prototype=X)
end

####
#### Predict methods
####

function MMI.predict(model::Union{KMeans,KMedoids}, fitresult, Xnew)

    locations, cluster_labels = fitresult

    Xarray = MMI.matrix(Xnew)
    (n, p), k = size(Xarray), model.k

    pred = zeros(Int, n)
    @inbounds for i ∈ 1:n
        minv = Inf
        for j ∈ 1:k
            curv    = evaluate(model.metric,
                                  view(Xarray, i, :), view(locations, :, j))
            P       = curv < minv
            pred[i] =    j * P + pred[i] * !P # if P is true --> j
            minv    = curv * P +    minv * !P # if P is true --> curvalue
        end
    end
    return cluster_labels[pred]
end

####
#### METADATA
####

metadata_pkg.((KMeans, KMedoids),
    name="Clustering",
    uuid="aaaa29a8-35af-508c-8bc3-b662a17a0fe5",
    url="https://github.com/JuliaStats/Clustering.jl",
    julia=true,
    license="MIT",
    is_wrapper=false
    )

metadata_model(KMeans,
    input   = Table(Continuous),
    output  = Table(Continuous),
    weights = false,
    descr   = KMeansDescription
    )

metadata_model(KMedoids,
    input   = Table(Continuous),
    output  = Table(Continuous),
    weights = false,
    descr   = KMedoidsDescription
    )

end # module
