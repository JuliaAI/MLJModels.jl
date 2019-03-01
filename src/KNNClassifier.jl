module KNNClassifier

export KNNClassifier
# metrics from distances
export  Euclidean,
        Cityblock,
        Minkowski,
        Chebyshev,
        Hamming

import MLJBase

using LinearAlgebra

#> needed for all classifiers:
using CategoricalArrays
using NearestNeighbors
using Distances
using StatsBase: mode # for choosing in knn 

# to be extended:

struct KNNFitResult{T, F}
    fitted :: T
    y :: F
end


# have to rethink -- dispatch on singletons
abstract type NeighbourTree end
struct KD <: NeighbourTree end
struct Ball <: NeighbourTree end
struct Brute <: NeighbourTree end


mutable struct KNNClassifier{M <: Metric, T <: NeighbourTree, F} 
    target_type :: Type{F}
    K :: Int                # number of local target values averaged
    metric :: M
    tree :: T  # BallTree, KDTree or BruteTree
    leafsize :: Int   # when to stop doing new leaves
end


function KNNClassifier(;
                        target_type = Int,
                        K = 5,
                        metric = Euclidean(),
                        tree = KD(),
                        leafsize = 10
                        )
    # have to implement this here due to the fact
    # that KD tree can be only with this type of metric
    # cannot change the type of tree after it is constructed
    metricType = typeof(metric)
    message = ""
    if  (typeof(tree) == KD) && 
        !((metricType <: Euclidean) || (metricType <: Chebyshev) ||
        (metricType <: Minkowski) || (metricType <: Cityblock))
        tree = Ball()
        message *= "KDTree supports only these types of metric :"
        message *= "Euclidean, Chebyshev, Minkowski, Cityblock. "
        message *= "Tree was set to BallTree"
    end


    model = KNNClassifier(
        target_type,
        K,
        metric,
        tree,
        leafsize
    )
    message *= MLJBase.clean!(model)
    isempty(message) || @warn message 

    return model
end


    
function MLJBase.clean!(model::KNNClassifier)
    message = ""
    if model.K <= 0
        model.K = 1
        message *= "K cannot be negative. K set to 1. "
    end
    if model.leafsize <= 0
        model.leafsize = 10
        message *= "Leafsize cannot be less than 1. Leafsize was set to 10."
    end
    return message
end


function create_tree(model :: KNNClassifier{M, KD, F} , X) where M where F
    return KDTree(X, model.metric; leafsize = model.leafsize)
end


function create_tree(model :: KNNClassifier{M, Ball, F} , X) where M where F
    return BallTree(X, model.metric; leafsize = model.leafsize)
end


function create_tree(model :: KNNClassifier{M, Brute, F} , X) where M where F
    return BruteTree(X, model.metric; leafsize = model.leafsize)
end


function MLJBase.fit(model::KNNClassifier{M, T, F}
                     , verbosity::Int
                     , X :: S
                     , y) where M where F where T where S

    Xraw = MLJBase.matrix(X)
    # bad place : right now KDtree and others accept 
    # only Array{A, 2}, no Adjoint allowed
    Xmatrix = Matrix(Xraw')
    tree = create_tree(model, Xmatrix)
    fitresult = KNNFitResult(tree, y)
    cache = nothing
    report = nothing
    return fitresult, cache, report 
end


function MLJBase.predict(model::KNNClassifier, fitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    Xmatrix = Matrix(Xnew')
    neighbors, distances = knn(fitresult.fitted, Xmatrix, model.K)
    # choose the most popular 
    nearest_points = mode.(neighbors)
    yres = fitresult.y[nearest_points]
    return yres
end


    
# metadata for KNNClassifier
# TO DO !!!!
MLJBase.load_path(::Type{<:KNNClassifier}) = "MLJ.KNNClassifier"
MLJBase.package_name(::Type{<:KNNClassifier}) = "MLJ"
MLJBase.package_uuid(::Type{<:KNNClassifier}) = ""
MLJBase.is_pure_julia(::Type{<:KNNClassifier}) = :yes
MLJBase.input_kinds(::Type{<:KNNClassifier}) = [:continuous]
MLJBase.output_kind(::Type{<:KNNClassifier}) = :continuous
MLJBase.output_quantity(::Type{<:KNNClassifier}) = :univariate


end # module

using .KNN

