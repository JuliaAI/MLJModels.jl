module DecisionTree_

import MLJModelInterface
import MLJModelInterface: @mlj_model, metadata_pkg, metadata_model,
                          Table, Continuous, Count, Finite, OrderedFactor,
                          Multiclass

const MMI = MLJModelInterface

import ..DecisionTree

const DT = DecisionTree

struct TreePrinter{T}
    tree::T
end
(c::TreePrinter)(depth) = DT.print_tree(c.tree, depth)
(c::TreePrinter)() = DT.print_tree(c.tree, 5)

Base.show(stream::IO, c::TreePrinter) =
    print(stream, "TreePrinter object (call with display depth)")


const DTC_DESCR = "CART decision tree classifier."
const DTR_DESCR = "CART decision tree regressor."
const RFC_DESCR = "Random forest classifier."
const RFR_DESCR = "Random forest regressor."
const ABS_DESCR = "Ada-boosted stump classifier."


"""
DecisionTreeClassifer(; kwargs...)

$DTC_DESCR

Inputs are tables with ordinal columns. That is, the element scitype
of each column can be `Continuous`, `Count` or `OrderedFactor`.
Predictions are probabilistic, but uncalibrated.

Instead of predicting the mode class at each leaf, a `UnivariateFinite`
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

- `n_subfeatures=0`: number of features to select at random (0 for all,
  -1 for square root of number of features)

- `post_prune=false`:      set to `true` for post-fit pruning

- `merge_purity_threshold=1.0`:  (post-pruning) merge leaves having `>=thresh`
                           combined purity

- `pdf_smoothing=0.0`:     threshold for smoothing the predicted scores

- `display_depth=5`:       max depth to show when displaying the tree

"""
@mlj_model mutable struct DecisionTreeClassifier <: MMI.Probabilistic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = 0::(_ ≥ -1)
    post_prune::Bool             = false
    merge_purity_threshold::Float64 = 1.0::(_ ≤ 1)
    pdf_smoothing::Float64       = 0.0::(0 ≤ _ ≤ 1)
    display_depth::Int           = 5::(_ ≥ 1)
end

function MMI.fit(m::DecisionTreeClassifier, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X)
    yplain  = MMI.int(y)

    classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
    integers_seen = MMI.int(classes_seen)

    tree = DT.build_tree(yplain, Xmatrix,
                         m.n_subfeatures,
                         m.max_depth,
                         m.min_samples_leaf,
                         m.min_samples_split,
                         m.min_purity_increase)
    if m.post_prune
        tree = DT.prune_tree(tree, m.merge_purity_threshold)
    end
    verbosity < 2 || DT.print_tree(tree, m.display_depth)

    fitresult = (tree, classes_seen, integers_seen)

    cache  = nothing
    report = (classes_seen=classes_seen,
              print_tree=TreePrinter(tree))

    return fitresult, cache, report
end

function get_encoding(classes_seen)
    a_cat_element = classes_seen[1]
    return Dict(c => MMI.int(c) for c in MMI.classes(a_cat_element))
end

MMI.fitted_params(::DecisionTreeClassifier, fitresult) =
    (tree=fitresult[1], encoding=get_encoding(fitresult[2]))

function smooth(scores, smoothing)
    iszero(smoothing) && return scores
    threshold = smoothing / size(scores, 2)
    # clip low values
    scores[scores .< threshold] .= threshold
    # normalize
    return scores ./ sum(scores, dims=2)
end

function MMI.predict(m::DecisionTreeClassifier, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    tree, classes_seen, integers_seen = fitresult
    # retrieve the predicted scores
    scores = DT.apply_tree_proba(tree, Xmatrix, integers_seen)
    # smooth if required
    sm_scores = smooth(scores, m.pdf_smoothing)
    # return vector of UF
    return MMI.UnivariateFinite(classes_seen, sm_scores)
end

"""
RandomForestClassifer(; kwargs...)

$RFC_DESCR

## Hyperparameters

- `max_depth=-1`:          max depth of the decision tree (-1=any)

- `min_samples_leaf=1`:    max number of samples each leaf needs to have

- `min_samples_split=2`:   min number of samples needed for a split

- `min_purity_increase=0`: min purity needed for a split

- `n_subfeatures=-1`: number of features to select at random (0 for all,
  -1 for square root of number of features)

- `n_trees=10`:            number of trees to train

- `sampling_fraction=0.7`  fraction of samples to train each tree on

- `pdf_smoothing=0.0`:     threshold for smoothing the predicted scores

"""
@mlj_model mutable struct RandomForestClassifier <: MMI.Probabilistic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = (-)(1)::(_ ≥ -1)
    n_trees::Int                 = 10::(_ ≥ 2)
    sampling_fraction::Float64   = 0.7::(0 < _ ≤ 1)
    pdf_smoothing::Float64       = 0.0::(0 ≤ _ ≤ 1)
end

function MMI.fit(m::RandomForestClassifier, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X)
    yplain  = MMI.int(y)

    classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
    integers_seen = MMI.int(classes_seen)

    forest = DT.build_forest(yplain, Xmatrix,
                             m.n_subfeatures,
                             m.n_trees,
                             m.sampling_fraction,
                             m.max_depth,
                             m.min_samples_leaf,
                             m.min_samples_split,
                             m.min_purity_increase)
    cache  = nothing
    report = NamedTuple()
    return (forest, classes_seen, integers_seen), cache, report
end

MMI.fitted_params(::RandomForestClassifier, (forest,_)) = (forest=forest,)

function MMI.predict(m::RandomForestClassifier, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    forest, classes_seen, integers_seen = fitresult
    scores = DT.apply_forest_proba(forest, Xmatrix, integers_seen)
    sm_scores = smooth(scores, m.pdf_smoothing)
    return MMI.UnivariateFinite(classes_seen, sm_scores)
end

"""
AdaBoostStumpClassifer(; kwargs...)

$RFC_DESCR

## Hyperparameters

- `n_iter=10`:   number of iterations of AdaBoost
- `pdf_smoothing=0.0`: threshold for smoothing the predicted scores
"""
@mlj_model mutable struct AdaBoostStumpClassifier <: MMI.Probabilistic
    n_iter::Int            = 10::(_ ≥ 1)
    pdf_smoothing::Float64 = 0.0::(0 ≤ _ ≤ 1)
end

function MMI.fit(m::AdaBoostStumpClassifier, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X)
    yplain  = MMI.int(y)

    classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
    integers_seen = MMI.int(classes_seen)

    stumps, coefs = DT.build_adaboost_stumps(yplain, Xmatrix,
                                             m.n_iter)
    cache  = nothing
    report = NamedTuple()
    return (stumps, coefs, classes_seen, integers_seen), cache, report
end

MMI.fitted_params(::AdaBoostStumpClassifier, (stumps,coefs,_)) =
    (stumps=stumps,coefs=coefs)

function MMI.predict(m::AdaBoostStumpClassifier, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    stumps, coefs, classes_seen, integers_seen = fitresult
    scores = DT.apply_adaboost_stumps_proba(stumps, coefs,
                                            Xmatrix, integers_seen)
    sm_scores = smooth(scores, m.pdf_smoothing)
    return MMI.UnivariateFinite(classes_seen, sm_scores)
end

## REGRESSION

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

- `n_subfeatures=0`: number of features to select at random (0 for all,
  -1 for square root of number of features)

- `post_prune=false`:      set to `true` for post-fit pruning

- `merge_purity_threshold=1.0`: (post-pruning) merge leaves having `>=thresh`
                           combined purity
"""
@mlj_model mutable struct DecisionTreeRegressor <: MMI.Deterministic
    max_depth::Int                               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int                = 5::(_ ≥ 0)
    min_samples_split::Int               = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int                   = 0::(_ ≥ -1)
    post_prune::Bool                     = false
    merge_purity_threshold::Float64 = 1.0::(0 ≤ _ ≤ 1)
end

function MMI.fit(m::DecisionTreeRegressor, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X)
    tree    = DT.build_tree(float(y), Xmatrix,
                            m.n_subfeatures,
                            m.max_depth,
                            m.min_samples_leaf,
                            m.min_samples_split,
                            m.min_purity_increase)

    if m.post_prune
        tree = DT.prune_tree(tree, m.merge_purity_threshold)
    end
    cache  = nothing
    report = nothing
    return tree, cache, report
end

MMI.fitted_params(::DecisionTreeRegressor, tree) = (tree=tree,)

function MMI.predict(::DecisionTreeRegressor, tree, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    return DT.apply_tree(tree, Xmatrix)
end

"""
RandomForestRegressor(; kwargs...)

$RFC_DESCR

## Hyperparameters

- `max_depth=-1`:          max depth of the decision tree (-1=any)

- `min_samples_leaf=1`:    max number of samples each leaf needs to have

- `min_samples_split=2`:   min number of samples needed for a split

- `min_purity_increase=0`: min purity needed for a split

- `n_subfeatures=-1`: number of features to select at random (0 for all,
  -1 for square root of number of features)

- `n_trees=10`:            number of trees to train

- `sampling_fraction=0.7`  fraction of samples to train each tree on

- `pdf_smoothing=0.0`:     threshold for smoothing the predicted scores

"""
@mlj_model mutable struct RandomForestRegressor <: MMI.Deterministic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = (-)(1)::(_ ≥ -1)
    n_trees::Int                 = 10::(_ ≥ 2)
    sampling_fraction::Float64   = 0.7::(0 < _ ≤ 1)
    pdf_smoothing::Float64       = 0.0::(0 ≤ _ ≤ 1)
end

function MMI.fit(m::RandomForestRegressor, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X)
    forest  = DT.build_forest(float(y), Xmatrix,
                              m.n_subfeatures,
                              m.n_trees,
                              m.sampling_fraction,
                              m.max_depth,
                              m.min_samples_leaf,
                              m.min_samples_split,
                              m.min_purity_increase)
    cache  = nothing
    report = NamedTuple()
    return forest, cache, report
end

MMI.fitted_params(::RandomForestRegressor, forest) = (forest=forest,)

function MMI.predict(::RandomForestRegressor, forest, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    return DT.apply_forest(forest, Xmatrix)
end

# ===

metadata_pkg.(
    (DecisionTreeClassifier, DecisionTreeRegressor,
     RandomForestClassifier, RandomForestRegressor,
     AdaBoostStumpClassifier),
    name       = "DecisionTree",
    uuid       = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb",
    url        = "https://github.com/bensadeghi/DecisionTree.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false)

metadata_model(DecisionTreeClassifier,
    input   = Table(Continuous, Count, OrderedFactor),
    target  = AbstractVector{<:Finite},
    weights = false,
    descr   = DTC_DESCR)

metadata_model(RandomForestClassifier,
    input   = Table(Continuous, Count, OrderedFactor),
    target  = AbstractVector{<:Finite},
    weights = false,
    descr   = RFC_DESCR)

metadata_model(AdaBoostStumpClassifier,
    input   = Table(Continuous, Count, OrderedFactor),
    target  = AbstractVector{<:Finite},
    weights = false,
    descr   = ABS_DESCR)

metadata_model(DecisionTreeRegressor,
    input   = Table(Continuous, Count, OrderedFactor),
    target  = AbstractVector{Continuous},
    weights = false,
    descr   = DTR_DESCR)

metadata_model(RandomForestRegressor,
    input   = Table(Continuous, Count, OrderedFactor),
    target  = AbstractVector{Continuous},
    weights = false,
    descr   = RFR_DESCR)

end # module
