# NOTE: there's a `kmeans!` function that updates centers, maybe a candidate
# for the `update` machinery. Same for `kmedoids!`
# NOTE: if the prediction is done on the original array, just the assignment
# should be returned, unclear what's the best way of doing this.

module Clustering_

import MLJBase
import MLJBase: @mlj_model, metadata_pkg, metadata_model
using ScientificTypes

import ..Clustering # strange sytax for lazy-loading

using Distances
using LinearAlgebra: norm

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
@mlj_model mutable struct KMeans <: MLJBase.Unsupervised
    k::Int = 3::(_ ≥ 2)
    metric::SemiMetric = SqEuclidean()
end

"""
KMedoids(; kwargs...)

$KMedoidsDescription

$KMFields

See also the [package documentation](http://juliastats.github.io/Clustering.jl/latest/kmedoids.html).
"""
@mlj_model mutable struct KMedoids <: MLJBase.Unsupervised
    k::Int = 3::(_ ≥ 2)
    metric::SemiMetric = SqEuclidean()
end

function MLJBase.fit(model::KMeans
                   , verbosity::Int
                   , X)
    # NOTE: using transpose here to get a LinearAlgebra.Transpose object which Kmeans can handle
    Xarray = transpose(MLJBase.matrix(X))

    result    = C.kmeans(Xarray, model.k; distance=model.metric)
    fitresult = result.centers # centers (p x k)
    cache     = nothing
    report    = (assignments=result.assignments,) # size n

    return fitresult, cache, report
end

MLJBase.fitted_params(::KMeans, fitresult) = (centers=fitresult,)

function MLJBase.transform(model::KMeans
                         , fitresult
                         , X)
    # pairwise distance from samples to centers
    X̃ = pairwise(model.metric, transpose(MLJBase.matrix(X)), fitresult, dims=2)
    return MLJBase.table(X̃, prototype=X)
end

function MLJBase.fit(model::KMedoids
                   , verbosity::Int
                   , X)

    # NOTE: using transpose=true will materialize the transpose (~ permutedims), KMedoids
    # does not yet accept LinearAlgebra.Transpose
    Xarray = MLJBase.matrix(X, transpose=true)
    # cost matrix: all the pairwise distances
    Carray    = pairwise(model.metric, Xarray, dims=2) # n x n
    result    = C.kmedoids(Carray, model.k)
    fitresult = view(Xarray, :, result.medoids) # medoids
    cache     = nothing
    report    = (assignments=result.assignments,) # size n

    return fitresult, cache, report
end

MLJBase.fitted_params(::KMedoids, fitresult) = (medoids=fitresult,)

function MLJBase.transform(model::KMedoids
                         , fitresult
                         , X)
    # pairwise distance from samples to medoids
    X̃ = pairwise(model.metric, MLJBase.matrix(X, transpose=true), fitresult, dims=2)
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

import ..metadata_pkg, ..metadata_model

metadata_pkg.((KMeans, KMedoids),
    name="Clustering",
    uuid="aaaa29a8-35af-508c-8bc3-b662a17a0fe5",
    url="https://github.com/JuliaStats/Clustering.jl",
    julia=true,
    license="MIT",
    is_wrapper=false
    )

metadata_model(KMeans,
    input=MLJBase.Table(MLJBase.Continuous),
    output=MLJBase.Table(MLJBase.Continuous),
    weights=false,
    descr=KMeansDescription
    )

metadata_model(KMedoids,
    input=MLJBase.Table(MLJBase.Continuous),
    output=MLJBase.Table(MLJBase.Continuous),
    weights=false,
    descr=KMedoidsDescription
    )

end # module
