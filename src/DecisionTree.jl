#> This code implements the MLJ model interface for models in the
#> DecisionTree.jl package. It is annotated so that it may serve as a
#> template for other regressors of Deterministic type and classifiers
#> of Probabilistic type. The annotations, which begin with "#>",
#> should be removed (but copy this file first!). See also the model
#> interface specification at "doc/adding_new_models.md".

#> Note that all models need to "register" their location by setting
#> `load_path(<:ModelType)` appropriately.

module DecisionTree_

#> export the new models you're going to define (and nothing else):
export DecisionTreeClassifier, DecisionTreeRegressor

import MLJBase

#> needed for classifiers:
using CategoricalArrays

#> import package:
import ..DecisionTree # strange syntax b/s we are lazy-loading


## CLASSIFIER

"""
    DecisionTreeClassifer(; kwargs...)

A variation on the CART decision tree classifier from
    [https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md](https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md).

Instead of predicting the mode class at each leaf, a UnivariateFinite
distribution is fit to the leaf training classes, with smoothing
controlled by an additional hyperparameter `pdf_smoothing`: If `n` is
the number of classes, then each class probability is replaced by
`pdf_smoothing/n`, if it falls below that value, and the resulting
vector of probabilities is renormalized.

For post-fit pruning, set `post-prune=true` and set
`min_purity_threshold` appropriately. Other hyperparameters as per
package documentation cited above.


"""
mutable struct DecisionTreeClassifier <: MLJBase.Probabilistic
    pruning_purity::Float64
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    min_purity_increase::Float64
    n_subfeatures::Int
    display_depth::Int
    post_prune::Bool
    merge_purity_threshold::Float64
    pdf_smoothing::Float64
end

# keywork constructor:
#> all arguments are kwargs with a default value
function DecisionTreeClassifier(
    ; pruning_purity=1.0
    , max_depth=-1
    , min_samples_leaf=1
    , min_samples_split=2
    , min_purity_increase=0.0
    , n_subfeatures=0
    , display_depth=5
    , post_prune=false
    , merge_purity_threshold=0.9
    , pdf_smoothing=0.05)

    model = DecisionTreeClassifier(
        pruning_purity
        , max_depth
        , min_samples_leaf
        , min_samples_split
        , min_purity_increase
        , n_subfeatures
        , display_depth
        , post_prune
        , merge_purity_threshold
        , pdf_smoothing)

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

#> The following optional method (the fallback does nothing, returns
#> empty warning) is called by the constructor above but also by the
#> fit methods below:
function MLJBase.clean!(model::DecisionTreeClassifier)
    warning = ""
    if  model.pruning_purity > 1
        warning *= "Need pruning_purity < 1. Resetting pruning_purity=1.0.\n"
        model.pruning_purity = 1.0
    end
    if model.min_samples_split < 2
        warning *= "Need min_samples_split >= 2. Resetting min_samples_slit=2.\n"
        model.min_samples_split = 2
    end
    if model.pdf_smoothing < 0 || model.pdf_smoothing > 1
        warning *= "Need pdf_smoothing in range [0, 1]. Resetting to 0.05\n"
        model.pdf_smoothing = 0.05
    end
    return warning
end


#> A required `fit` method returns `fitresult, cache, report`. (Return
#> `cache=nothing` unless you are overloading `update`)
function MLJBase.fit(model::DecisionTreeClassifier
             , verbosity::Int   #> must be here (and typed!!) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)

    yplain = identity.(y) # y as plain not abstact vector
    classes_seen = unique(yplain)

    tree = DecisionTree.build_tree(yplain,
                                   Xmatrix,
                                   model.n_subfeatures,
                                   model.max_depth,
                                   model.min_samples_leaf,
                                   model.min_samples_split,
                                   model.min_purity_increase)
    if model.post_prune
        tree = DecisionTree.prune_tree(tree, model.merge_purity_threshold)
    end

    verbosity < 3 || DecisionTree.print_tree(tree, model.display_depth)

    fitresult = (tree, classes_seen)

    #> return package-specific statistics (eg, feature rankings,
    #> internal estimates of generalization error) in `report`, which
    #> should be a named tuple with the same type every call (can have
    #> empty values):

    cache = nothing
    report = NamedTuple{}()

    return fitresult, cache, report

end

MLJBase.fitted_params(::DecisionTreeClassifier, fitresult) = fitresult[1]

function smooth(prob_vector, smoothing)
    threshold = smoothing/length(prob_vector)
    smoothed_vector = map(prob_vector) do p
        p < threshold ? threshold : p
    end
    smoothed_vector = smoothed_vector/sum(smoothed_vector)
    return smoothed_vector
end

function MLJBase.predict(model::DecisionTreeClassifier
                     , fitresult
                     , Xnew)
    Xmatrix = MLJBase.matrix(Xnew)

    tree, classes_seen = fitresult

    y_probabilities = DecisionTree.apply_tree_proba(tree, Xmatrix, classes_seen)
    return [MLJBase.UnivariateFinite(classes_seen,
                                     smooth(y_probabilities[i,:],
                                            model.pdf_smoothing))
            for i in 1:size(y_probabilities, 1)]
end


## REGRESSOR

"""
    DecisionTreeRegressor(; kwargs...)

CART decision tree classifier from
[https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md](https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md). Predictions
are Deterministic.

For post-fit pruning, set `post-prune=true` and set
`pruning_purity_threshold` appropriately. Other hyperparameters as per
package documentation cited above.

"""

mutable struct DecisionTreeRegressor <: MLJBase.Deterministic
    pruning_purity_threshold::Float64
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    min_purity_increase::Float64
    n_subfeatures::Int
    post_prune::Bool
end

# constructor:
#> all arguments are kwargs with a default value
function DecisionTreeRegressor(;
    pruning_purity_threshold=0.0
    , max_depth=-1
    , min_samples_leaf=5
    , min_samples_split=2
    , min_purity_increase=0.0
    , n_subfeatures=0
    , post_prune=false)

    model = DecisionTreeRegressor(
       pruning_purity_threshold
       , max_depth
       , min_samples_leaf
       , min_samples_split
       , min_purity_increase
       , n_subfeatures
       , post_prune)

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

function MLJBase.clean!(model::DecisionTreeRegressor)
    warning = ""
    if model.min_samples_split < 2
        warning *= "Need min_samples_split >= 2. Resetting min_samples_slit=2.\n"
        model.min_samples_split = 2
    end
    return warning
end

function MLJBase.fit(model::DecisionTreeRegressor
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)

    fitresult = DecisionTree.build_tree(float.(y)
				   , Xmatrix
				   , model.n_subfeatures
				   , model.max_depth
				   , model.min_samples_leaf
				   , model.min_samples_split
				   , model.min_purity_increase)

    if model.post_prune
        fitresult = DecisionTree.prune_tree(fitresult,
                                            model.pruning_purity_threshold)
    end
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

MLJBase.fitted_params(::DecisionTreeRegressor, fitresult) = fitresult

function MLJBase.predict(model::DecisionTreeRegressor
                     , fitresult
                     , Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return DecisionTree.apply_tree(fitresult,Xmatrix)
end


## METADATA

const DTTypes = Union{DecisionTreeClassifier, DecisionTreeRegressor}

MLJBase.package_name(::Type{<:DTTypes}) = "DecisionTree"
MLJBase.package_uuid(::Type{<:DTTypes}) = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
MLJBase.package_url(::Type{<:DTTypes}) = "https://github.com/bensadeghi/DecisionTree.jl"
MLJBase.is_pure_julia(::Type{<:DTTypes}) = true

MLJBase.load_path(::Type{<:DecisionTreeClassifier}) = "MLJModels.DecisionTree_.DecisionTreeClassifier"
MLJBase.load_path(::Type{<:DecisionTreeRegressor}) = "MLJModels.DecisionTree_.DecisionTreeRegressor"

MLJBase.input_scitype(::Type{<:DecisionTreeClassifier}) = MLJBase.Continuous
MLJBase.input_scitype(::Type{<:DecisionTreeRegressor}) = MLJBase.Continuous

MLJBase.target_scitype(::Type{<:DecisionTreeClassifier}) = MLJBase.Finite
MLJBase.target_scitype(::Type{<:DecisionTreeRegressor}) = MLJBase.Continuous

end # module
