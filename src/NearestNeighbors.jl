module NearestNeighbors_

import MLJModelInterface
import MLJModelInterface: @mlj_model, metadata_model, metadata_pkg,
                          Table, Continuous, Count, Finite, OrderedFactor,
                          Multiclass

const MMI = MLJModelInterface

using Distances

import ..NearestNeighbors

const NN = NearestNeighbors

const KNNRegressorDescription =
    """
    K-Nearest Neighbors regressor: predicts the response associated with a new point
    by taking an average of the response of the K-nearest points.
    """

const KNNClassifierDescription =
    """
    K-Nearest Neighbors classifier: predicts the class associated with a new point
    by taking a vote over the classes of the K-nearest points.
    """

const KNNFields =
    """
    ## Keywords

    * `K=5`                 : number of neighbors
    * `algorithm=:kdtree`   : one of `(:kdtree, :brutetree, :balltree)`
    * `metric=Euclidean()`  : a `Metric` object for the distance between points
    * `leafsize=10`         : at what number of points to stop splitting the tree
    * `reorder=true`        : if true puts points close in distance close in memory
    * `weights=:uniform`    : one of `(:uniform, :distance)` if `:uniform` all neighbors are
                              considered as equally important, if `:distance`, closer neighbors
                              are proportionally more important.

    See also the [package documentation](https://github.com/KristofferC/NearestNeighbors.jl).
    """

"""
KNNRegressoor(;kwargs...)

$KNNRegressorDescription

$KNNFields
"""
@mlj_model mutable struct KNNRegressor <: MMI.Deterministic
    K::Int            = 5::(_ > 0)
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :brutetree, :balltree))
    metric::Metric    = Euclidean()
    leafsize::Int     = 10::(_ ≥ 0)
    reorder::Bool     = true
    weights::Symbol   = :uniform::(_ in (:uniform, :distance))
end

"""
KNNRegressor(;kwargs...)

$KNNClassifierDescription

$KNNFields
"""
@mlj_model mutable struct KNNClassifier <: MMI.Probabilistic
    K::Int            = 5::(_ > 0)
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :brutetree, :balltree))
    metric::Metric    = Euclidean()
    leafsize::Int     = 10::(_ ≥ 0)
    reorder::Bool     = true
    weights::Symbol   = :uniform::(_ in (:uniform, :distance))
end

const KNN = Union{KNNRegressor, KNNClassifier}

function MMI.fit(m::KNN, verbosity::Int, X, y, w=nothing)
    Xmatrix = MMI.matrix(X, transpose=true) # NOTE: copies the data
    if m.algorithm == :kdtree
        tree = NN.KDTree(Xmatrix; leafsize=m.leafsize, reorder=m.reorder)
    elseif m.algorithm == :balltree
        tree = NN.BallTree(Xmatrix; leafsize=m.leafsize, reorder=m.reorder)
    elseif m.algorithm == :brutetree
        tree = NN.BruteTree(Xmatrix; leafsize=m.leafsize, reorder=m.reorder)
    end
    report = NamedTuple{}()
    return (tree, y, w), nothing, report
end

MMI.fitted_params(model::KNN, (tree, _)) = (tree=tree,)

function MMI.predict(m::KNNClassifier, (tree, y, w), X)
    Xmatrix = MMI.matrix(X, transpose=true) # NOTE: copies the data
    # for each entry, get the K closest training point + their distance
    idxs, dists = NN.knn(tree, Xmatrix, m.K)

    classes     = MMI.classes(y[1])
    probas      = zeros(length(idxs), length(classes))

    w_ = ones(m.K)

    # go over each test record, and for each go over the k nearest entries
    for i in eachindex(idxs)
        idxs_  = idxs[i]
        dists_ = dists[i]
        labels = y[idxs_]
        if w !== nothing
            w_ = w[idxs_]
        end
#        probas .*= 0.0 # (adb) seems to be redundant
        if m.weights == :uniform
            for (k, label) in enumerate(labels)
                probas[i, classes .== label] .+= 1.0 / m.K * w_[k]
            end
        else
            for (k, label) in enumerate(labels)
                probas[i, classes .== label] .+= 1.0 / dists_[k] * w_[k]
            end
        end
    end
    # normalize probas along rows:
    probas ./= sum(probas, dims=2)
    return  MMI.UnivariateFinite(classes, probas)

end

function MMI.predict(m::KNNRegressor, (tree, y, w), X)
    Xmatrix     = MMI.matrix(X, transpose=true) # NOTE: copies the data
    idxs, dists = NN.knn(tree, Xmatrix, m.K)
    preds       = similar(y, length(idxs))

    w_ = ones(m.K)

    for i in eachindex(idxs)
        idxs_  = idxs[i]
        dists_ = dists[i]
        values = y[idxs_]
        if w !== nothing
            w_ = w[idxs_]
        end
        if m.weights == :uniform
            preds[i] = sum(values .* w_) / sum(w_)
        else
            preds[i] = sum(values .* w_ .* (1.0 .- dists_ ./ sum(dists_))) / (sum(w_) - 1)
        end
    end
    return preds
end

# ====

metadata_pkg.((KNNRegressor, KNNClassifier),
    name       = "NearestNeighbors",
    uuid       = "b8a86587-4115-5ab1-83bc-aa920d37bbce",
    url        = "https://github.com/KristofferC/NearestNeighbors.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false
    )

metadata_model(KNNRegressor,
    input   = Table(Continuous),
    target  = Union{AbstractVector{Continuous}, AbstractVector{<:AbstractArray{Continuous}}},
    weights = true,
    descr   = KNNRegressorDescription
    )

metadata_model(KNNClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = true,
    descr   = KNNClassifierDescription
    )

end # module
