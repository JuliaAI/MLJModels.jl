module NearestNeighbors_

import MLJBase
using Parameters
using Distances

import ..NearestNeighbors

const NN = NearestNeighbors

export KNNRegressor, KNNClassifier

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

    * `k=5`                 : number of neighbors
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
@with_kw mutable struct KNNRegressor <: MLJBase.Deterministic
    k::Int            = 5           # > 0
    algorithm::Symbol = :kdtree     # (:kdtree, :brutetree, :balltree)
    metric::Metric    = Euclidean() #
    leafsize::Int     = 10          # > 0
    reorder::Bool     = true
    weights::Symbol   = :uniform    # (:uniform, :distance)
end

"""
KNNRegressoor(;kwargs...)

$KNNClassifierDescription

$KNNFields
"""
@with_kw mutable struct KNNClassifier <: MLJBase.Probabilistic
    k::Int            = 5           # > 0
    algorithm::Symbol = :kdtree     # (:kdtree, :brutetree, :balltree)
    metric::Metric    = Euclidean() #
    leafsize::Int     = 10          # > 0
    reorder::Bool     = true
    weights::Symbol   = :uniform    # (:uniform, :distance)
end

const KNN = Union{KNNRegressor, KNNClassifier}

function MLJBase.clean!(m::KNN)
    warning = ""
    if m.k < 1
        warning *= "Number of neighbors 'k' needs to be larger than 0. Setting to 1.\n"
        m.k = 1
    end
    if m.leafsize < 0
        warning *= "Leaf size should be ≥ 0. Setting to 10.\n"
        m.leafsize = 10
    end
    if m.algorithm ∉ (:kdtree, :brutetree, :balltree)
        warning *= "The tree algorithm should be ':kdtree', ':brutetree' or ':balltree'." *
                   "Setting to ':kdtree'.\n"
        m.algorithm = :kdtree
    end
    if m.algorithm == :kdtree && !isa(m.metric ∉ (Euclidean, Chebyshev, Minkowski, Citiblock))
        warning *= "KDTree only supports axis-aligned metrics. Setting to 'Euclidean'.\n"
        m.metric = Euclidean()
    end
    if m.weights ∉ (:uniform, :distance)
        warning *= "Weighing should be ':uniform' or ':distance'. Setting to ':uniform'.\n"
        m.weights = :distance
    end
    return warning
end

function MLJBase.fit(m::KNN, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X, transpose=true) # NOTE: copies the data
    if m.algorithm == :kdtree
        tree = NN.KDTree(Xmatrix; leafsize=m.leafsize, reorder=m.reorder)
    elseif m.algorithm == :balltree
        tree = NN.BallTree(Xmatrix; leafsize=m.leafsize, reorder=m.reorder)
    elseif m.algorithm == :brutetree
        tree = NN.BruteTree(Xmatrix; leafsize=m.leafsize, reorder=m.reorder)
    end
    report = NamedTuple{}()
    return (tree, y), nothing, report
end

MLJBase.fitted_params(model::KNN, (tree, _)) = (tree=tree,)

function MLJBase.predict(m::KNNClassifier, (tree, y), X)
    Xmatrix     = MLJBase.matrix(X, transpose=true) # NOTE: copies the data
    idxs, dists = NN.knn(tree, Xmatrix, m.k)
    preds       = Vector{MLJBase.UnivariateFinite}(undef, length(idxs))
    classes     = MLJBase.classes(y[1])
    probas      = zeros(length(classes))
    for i in eachindex(idxs)
        idxs_    = idxs[i]
        dists_   = dists[i]
        labels   = y[idxs_]
        probas .*= 0.0
        if m.weights == :uniform
            for label in labels
                probas[classes .== label] .+= 1.0 / m.k
            end
        else
            for (i, label) in enumerate(labels)
                probas[classes .== label] .+= 1.0 / dists_[i]
            end
            # normalize so that sum to 1
            probas ./= sum(probas)
        end
        preds[i] = MLJBase.UnivariateFinite(classes, probas)
    end
    return preds
end

function MLJBase.predict(m::KNNRegressor, (tree, y), X)
    Xmatrix     = MLJBase.matrix(X, transpose=true) # NOTE: copies the data
    idxs, dists = NN.knn(tree, Xmatrix, m.k)
    preds       = zeros(length(idxs))
    for i in eachindex(idxs)
        idxs_  = idxs[i]
        dists_ = dists[i]
        values = y[idxs_]
        if m.weights == :uniform
            preds[i] = sum(values) / m.k
        else
            preds[i] = sum(values .* (1.0 .- dists_ ./ sum(dists_))) / (m.k-1)
        end
    end
    return preds
end

# ====

import ..metadata_pkg, ..metadata_mod

metadata_pkg.((KNNRegressor, KNNClassifier),
    name="NearestNeighbors",
    uuid="b8a86587-4115-5ab1-83bc-aa920d37bbce",
    url="https://github.com/KristofferC/NearestNeighbors.jl",
    julia=true,
    license="MIT",
    is_wrapper=false
    )

metadata_model(KNNRegressor,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{MLJBase.Continuous},
    weights=false,
    descr=KNNRegressorDescription
    )

metadata_model(KNNClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr=KNNClassifierDescription
    )

end # module
