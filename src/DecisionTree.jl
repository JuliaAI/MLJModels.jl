module DecisionTree_

import MLJBase
import MLJBase: @mlj_model, metadata_pkg, metadata_model
using ScientificTypes

import ..DecisionTree

struct TreePrinter{T}
    tree::T
end
(c::TreePrinter)(depth) = DecisionTree.print_tree(c.tree, depth)
(c::TreePrinter)() = DecisionTree.print_tree(c.tree, 5)

Base.show(stream::IO, c::TreePrinter) =
    print(stream, "TreePrinter object (call with display depth)")


const DTC_DESCR = "CART decision tree classifier."

const DTR_DESCR = "CART decision tree regressor."


"""
DecisionTreeClassifer(; kwargs...)

$DTC_DESCR

Inputs are tables with ordinal columns. That is, the element scitype
of each column can be `Continuous`, `Count` or `OrderedFactor`.
Predictions are Probabilistic.

Instead of predicting the mode class at each leaf, a UnivariateFinite
distribution is fit to the leaf training classes, with smoothing
controlled by an additional hyperparameter `pdf_smoothing`: If `n` is
the number of observed classes, then each class probability is
replaced by `pdf_smoothing/n`, if it falls below that ratio, and the
resulting vector of probabilities is renormalized. Smoothing is only
applied to classes actually observed in training. Unseen classes
retain zero-probability predictions.

To visualize the fitted tree in the REPL, set `verbosity=2` when
fitting, or call `report(mach).print_tree(display_depth)` where `mach`
is the fitted machine, and `display_depth` the desired
depth. Interpretting the results will require a knowledge of the
internal integer encodings of classes, which are given in
`fitted_params(mach)` (which also stores the raw learned tree object
from the DecisionTree.jl algorithm).

## Hyperparameters

- `max_depth=-1`:          max depth of the decision tree (-1=any)
- `min_samples_leaf=1`:    max number of samples each leaf needs to have
- `min_samples_split=2`:   min number of samples needed for a split
- `min_purity_increase=0`: min purity needed for a split
- `n_subfeatures=0`:       number of features to select at random (0=all)
- `post_prune=false`:      set to `true` for post-fit pruning
- `merge_purity_threshold=1.0`:  (post-pruning) merge leaves having `>=thresh`
                           combined purity
- `pdf_smoothing=0.0`:     threshold for smoothing the scores
- `display_depth=5`:       max depth to show when displaying the tree
"""
@mlj_model mutable struct DecisionTreeClassifier <: MLJBase.Probabilistic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = 0::(_ ≥ 0)
    post_prune::Bool             = false
    merge_purity_threshold::Float64 = 1.0::(_ ≤ 1)
    pdf_smoothing::Float64       = 0.0::(0 ≤ _ ≤ 1)
    display_depth::Int           = 5::(_ ≥ 1)
end

function MLJBase.fit(model::DecisionTreeClassifier, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    yplain  = MLJBase.int(y)
    classes_seen  = filter(in(unique(y)), MLJBase.classes(y[1]))
    integers_seen = MLJBase.int(classes_seen)

    tree = DecisionTree.build_tree(yplain, Xmatrix,
                                   model.n_subfeatures,
                                   model.max_depth,
                                   model.min_samples_leaf,
                                   model.min_samples_split,
                                   model.min_purity_increase)
    if model.post_prune
        tree = DecisionTree.prune_tree(tree, model.merge_purity_threshold)
    end
    verbosity < 2 || DecisionTree.print_tree(tree, model.display_depth)

    fitresult = (tree, classes_seen, integers_seen)

    cache  = nothing
    report = (classes_seen=classes_seen,
              print_tree=TreePrinter(tree))

    return fitresult, cache, report
end

function get_encoding(classes_seen)
    a_cat_element = classes_seen[1]
    return Dict(c => MLJBase.int(c) for c in MLJBase.classes(a_cat_element))
end

MLJBase.fitted_params(::DecisionTreeClassifier, fitresult) =
    (tree_or_leaf=fitresult[1], encoding=get_encoding(fitresult[2]))

function smooth(scores, smoothing)
    iszero(smoothing) && return scores
    threshold = smoothing / size(scores, 2)
    # clip low values
    scores[scores .< threshold] .= threshold
    # normalize
    return scores ./ sum(scores, dims=2)
end

function MLJBase.predict(model::DecisionTreeClassifier, fitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    tree, classes_seen, integers_seen = fitresult
    # retrieve the predicted scores
    scores = DecisionTree.apply_tree_proba(tree, Xmatrix, integers_seen)
    # smooth if required
    sm_scores = smooth(scores, model.pdf_smoothing)
    # return vector of UF
    return [MLJBase.UnivariateFinite(classes_seen, sm_scores[i, :])
                    for i in 1:size(sm_scores, 1)]
end


"""
DecisionTreeRegressor(; kwargs...)

$DTC_DESCR

Inputs are tables with ordinal columns. That is, the element scitype
of each column can be `Continuous`, `Count` or `OrderedFactor`. Predictions
are Deterministic.

## Hyperparameters

- `max_depth=-1`:          max depth of the decision tree (-1=any)
- `min_samples_leaf=1`:    max number of samples each leaf needs to have
- `min_samples_split=2`:   min number of samples needed for a split
- `min_purity_increase=0`: min purity needed for a split
- `n_subfeatures=0`:       number of features to select at random (0=all)
- `post_prune=false`:      set to `true` for post-fit pruning
- `merge_purity_threshold=1.0`: (post-pruning) merge leaves having `>=thresh`
                           combined purity
"""
@mlj_model mutable struct DecisionTreeRegressor <: MLJBase.Deterministic
    max_depth::Int				 = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int		 = 5::(_ ≥ 0)
    min_samples_split::Int		 = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int			 = 0::(_ ≥ 0)
    post_prune::Bool			 = false
    merge_purity_threshold::Float64 = 1.0::(0 ≤ _ ≤ 1)
end

function MLJBase.fit(model::DecisionTreeRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    tree = DecisionTree.build_tree(float(y), Xmatrix,
                                        model.n_subfeatures,
                                        model.max_depth,
                                        model.min_samples_leaf,
                                        model.min_samples_split,
                                        model.min_purity_increase)

    if model.post_prune
        tree = DecisionTree.prune_tree(tree, model.merge_purity_threshold)
    end
    cache  = nothing
    report = nothing
    return tree, cache, report
end

MLJBase.fitted_params(::DecisionTreeRegressor, tree) = (tree=tree,)

function MLJBase.predict(model::DecisionTreeRegressor, tree, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return DecisionTree.apply_tree(tree, Xmatrix)
end

# ===

metadata_pkg.((DecisionTreeClassifier, DecisionTreeRegressor),
              name="DecisionTree",
              uuid="7806a523-6efd-50cb-b5f6-3fa6f1930dbb",
              url="https://github.com/bensadeghi/DecisionTree.jl",
              julia=true,
              license="MIT",
              is_wrapper=false)

metadata_model(DecisionTreeClassifier,
               input=MLJBase.Table(Continuous, Count, OrderedFactor),
               target=AbstractVector{<:MLJBase.Finite},
               weights=false,
               descr=DTC_DESCR)

metadata_model(DecisionTreeRegressor,
               input=MLJBase.Table(Continuous, Count, OrderedFactor),
               target=AbstractVector{MLJBase.Continuous},
               weights=false,
               descr=DTR_DESCR)

end # module
