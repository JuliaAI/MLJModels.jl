module XGBoost_

export XGBoostRegressor, XGBoostClassifier, XGBoostCount

import MLJModelInterface
import MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor,
                          Multiclass

const MMI = MLJModelInterface

import Tables: schema

import ..XGBoost

# helper for feature importances:
# XGBoost used "f
function named_importance(fi::XGBoost.FeatureImportance, features)

    new_fname = features[parse(Int, fi.fname[2:end]) + 1] |> string
    return XGBoost.FeatureImportance(new_fname, fi.gain, fi.cover, fi.freq)
end


# TODO: Why do we need this?
generate_seed() = mod(round(Int, time()*1e8), 10000)

# helper to preprocess hyper-parameters:
function kwargs(model, silent, seed, objective)

    kwargs = (booster = model.booster
              , silent = silent
              , disable_default_eval_metric = model.disable_default_eval_metric
              , eta = model.eta
              , gamma = model.gamma
              , max_depth = model.max_depth
              , min_child_weight = model.min_child_weight
              , max_delta_step = model.max_delta_step
              , subsample = model.subsample
              , colsample_bytree = model.colsample_bytree
              , colsample_bylevel = model.colsample_bylevel
              , lambda = model.lambda
              , alpha = model.alpha
              , tree_method = model.tree_method
              , sketch_eps = model.sketch_eps
              , scale_pos_weight = model.scale_pos_weight
              , refresh_leaf = model.refresh_leaf
              , process_type = model.process_type
              , grow_policy = model.grow_policy
              , max_leaves = model.max_leaves
              , max_bin = model.max_bin
              , predictor = model.predictor
              , sample_type = model.sample_type
              , normalize_type = model.normalize_type
              , rate_drop = model.rate_drop
              , one_drop = model.one_drop
              , skip_drop = model.skip_drop
              , feature_selector = model.feature_selector
              , top_k = model.top_k
              , tweedie_variance_power = model.tweedie_variance_power
              , objective = objective
              , base_score = model.base_score
              , eval_metric=model.eval_metric
              , seed = seed)

    if model.updater != "auto"
        return merge(kwargs, (updater=model.updater,))
    else
        return kwargs
    end
end


## REGRESSOR

mutable struct XGBoostRegressor <:MMI.Deterministic
    num_round::Int
    booster::String
    disable_default_eval_metric::Int
    eta::Float64
    gamma::Float64
    max_depth::Int
    min_child_weight::Float64
    max_delta_step::Float64
    subsample::Float64
    colsample_bytree::Float64
    colsample_bylevel::Float64
    lambda::Float64
    alpha::Float64
    tree_method::String
    sketch_eps::Float64
    scale_pos_weight::Float64
    updater::String
    refresh_leaf::Union{Int,Bool}
    process_type::String
    grow_policy::String
    max_leaves::Int
    max_bin::Int
    predictor::String
    sample_type::String
    normalize_type::String
    rate_drop::Float64
    one_drop
    skip_drop::Float64
    feature_selector::String
    top_k::Int
    tweedie_variance_power::Float64
    objective
    base_score::Float64
    eval_metric
    seed::Int
end

"""
    XGBoostRegressor(; objective="linear", seed=0, kwargs...)

The XGBoost model for univariate targets with `Continuous` element
scitype. Gives deterministic (point) predictions. For possible values
for `objective` and `kwargs`, see
[https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html).

For a time-dependent random seed, use `seed=-1`.

See also: XGBoostCount, XGBoostClassifier

"""
function XGBoostRegressor(
    ;num_round=100
    ,booster="gbtree"
    ,disable_default_eval_metric=0
    ,eta=0.3
    ,gamma=0
    ,max_depth=6
    ,min_child_weight=1
    ,max_delta_step=0
    ,subsample=1
    ,colsample_bytree=1
    ,colsample_bylevel=1
    ,lambda=1
    ,alpha=0
    ,tree_method="auto"
    ,sketch_eps=0.03
    ,scale_pos_weight=1
    ,updater="auto"
    ,refresh_leaf=1
    ,process_type="default"
    ,grow_policy="depthwise"
    ,max_leaves=0
    ,max_bin=256
    ,predictor="cpu_predictor" #> gpu version not currently working with Julia, maybe remove completely?
    ,sample_type="uniform"
    ,normalize_type="tree"
    ,rate_drop=0.0
    ,one_drop=0
    ,skip_drop=0.0
    ,feature_selector="cyclic"
    ,top_k=0
    ,tweedie_variance_power=1.5
    ,objective="reg:squarederror"
    ,base_score=0.5
    ,eval_metric="rmse"
    ,seed=0)

    model = XGBoostRegressor(
    num_round
    ,booster
    ,disable_default_eval_metric
    ,eta
    ,gamma
    ,max_depth
    ,min_child_weight
    ,max_delta_step
    ,subsample
    ,colsample_bytree
    ,colsample_bylevel
    ,lambda
    ,alpha
    ,tree_method
    ,sketch_eps
    ,scale_pos_weight
    ,updater
    ,refresh_leaf
    ,process_type
    ,grow_policy
    ,max_leaves
    ,max_bin
    ,predictor #> gpu version not currently working with Julia, maybe remove completely?
    ,sample_type
    ,normalize_type
    ,rate_drop
    ,one_drop
    ,skip_drop
    ,feature_selector
    ,top_k
    ,tweedie_variance_power
    ,objective
    ,base_score
    ,eval_metric
    ,seed)

     message = MMI.clean!(model)
     isempty(message) || @warn message

    return model
end

function MMI.clean!(model::XGBoostRegressor)
    warning = ""
    if model.objective == "count:poisson"
        warning *= "Your `objective` suggests prediction of a "*
        "`Count` variable.\n You may want to consider XGBoostCount instead. "
    elseif model.objective in ["reg:logistic", "binary:logistic",
                               "binary:logitraw", "binary:hinge",
                               "multi:softmax", "multi:softprob"]
        warning *="Your `objective` suggests prediction of a "*
        "`Finite` variable.\n You may want to consider XGBoostClassifier "*
        "instead. "
    end
    return warning
end

function MMI.fit(model::XGBoostRegressor
             , verbosity::Int     #> must be here even if unsupported in pkg
             , X
             , y)

             silent =
                 verbosity > 0 ?  false : true
    Xmatrix = MMI.matrix(X)
    dm = XGBoost.DMatrix(Xmatrix,label=y)

    objective =
        model.objective in ["linear", "gamma", "tweedie"] ?
            "reg:"*model.objective : model.objective

    seed =
        model.seed == -1 ? generate_seed() : model.seed


    fitresult = XGBoost.xgboost(dm, model.num_round;
                                kwargs(model, silent, seed, objective)...)

    features = schema(X).names
    importances = [named_importance(fi, features) for
                   fi in XGBoost.importance(fitresult)]
    cache = nothing

    report = (feature_importances=importances, )

    return fitresult, cache, report

end


function MMI.predict(model::XGBoostRegressor
        , fitresult
        , Xnew)
    Xmatrix = MMI.matrix(Xnew)
    return XGBoost.predict(fitresult, Xmatrix)
end


## COUNT REGRESSOR

mutable struct XGBoostCount <:MMI.Deterministic
    num_round::Int
    booster::String
    disable_default_eval_metric::Int
    eta::Float64
    gamma::Float64
    max_depth::Int
    min_child_weight::Float64
    max_delta_step::Float64
    subsample::Float64
    colsample_bytree::Float64
    colsample_bylevel::Float64
    lambda::Float64
    alpha::Float64
    tree_method::String
    sketch_eps::Float64
    scale_pos_weight::Float64
    updater::String
    refresh_leaf::Union{Int,Bool}
    process_type::String
    grow_policy::String
    max_leaves::Int
    max_bin::Int
    predictor::String
    sample_type::String
    normalize_type::String
    rate_drop::Float64
    one_drop
    skip_drop::Float64
    feature_selector::String
    top_k::Int
    tweedie_variance_power::Float64
    objective
    base_score::Float64
    eval_metric
    seed::Int
end


"""
    XGBoostCount(; seed=0, kwargs...)

The XGBoost model for targets with `Count` scitype. Gives
deterministic (point) predictions. For admissible `kwargs`, see
[https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html).

For a time-dependent random seed, use `seed=-1`.

See also: XGBoostRegressor, XGBoostClassifier

"""
function XGBoostCount(
    ;num_round=100
    ,booster="gbtree"
    ,disable_default_eval_metric=0
    ,eta=0.3
    ,gamma=0
    ,max_depth=6
    ,min_child_weight=1
    ,max_delta_step=0
    ,subsample=1
    ,colsample_bytree=1
    ,colsample_bylevel=1
    ,lambda=1
    ,alpha=0
    ,tree_method="auto"
    ,sketch_eps=0.03
    ,scale_pos_weight=1
    ,updater="auto"
    ,refresh_leaf=1
    ,process_type="default"
    ,grow_policy="depthwise"
    ,max_leaves=0
    ,max_bin=256
    ,predictor="cpu_predictor" #> gpu version not currently working with Julia, maybe remove completely?
    ,sample_type="uniform"
    ,normalize_type="tree"
    ,rate_drop=0.0
    ,one_drop=0
    ,skip_drop=0.0
    ,feature_selector="cyclic"
    ,top_k=0
    ,tweedie_variance_power=1.5
    ,objective="count:poisson"
    ,base_score=0.5
    ,eval_metric="rmse"
    ,seed=0)

    model = XGBoostCount(
    num_round
    ,booster
    ,disable_default_eval_metric
    ,eta
    ,gamma
    ,max_depth
    ,min_child_weight
    ,max_delta_step
    ,subsample
    ,colsample_bytree
    ,colsample_bylevel
    ,lambda
    ,alpha
    ,tree_method
    ,sketch_eps
    ,scale_pos_weight
    ,updater
    ,refresh_leaf
    ,process_type
    ,grow_policy
    ,max_leaves
    ,max_bin
    ,predictor #> gpu version not currently working with Julia, maybe remove completely?
    ,sample_type
    ,normalize_type
    ,rate_drop
    ,one_drop
    ,skip_drop
    ,feature_selector
    ,top_k
    ,tweedie_variance_power
    ,objective
    ,base_score
    ,eval_metric
    ,seed)

     message = MMI.clean!(model)
     isempty(message) || @warn message

    return model
end

function MMI.clean!(model::XGBoostCount)
    warning = ""
    if(!(model.objective in ["count:poisson"]))
        warning *= "Changing objective to \"poisson\", "*
                       "the only supported value. "
        model.objective="poisson"
    end
    return warning
end

function MMI.fit(model::XGBoostCount
             , verbosity::Int     #> must be here even if unsupported in pkg
             , X
             , y)

    silent = verbosity > 0 ?  false : true

    Xmatrix = MMI.matrix(X)
    dm = XGBoost.DMatrix(Xmatrix,label=y)

    seed =
        model.seed == -1 ? generate_seed() : model.seed

    fitresult = XGBoost.xgboost(dm, model.num_round;
                                kwargs(model, silent, seed, "count:poisson")...)
    features = schema(X).names
    importances = [named_importance(fi, features) for
                   fi in XGBoost.importance(fitresult)]

    cache = nothing
    report = (feature_importances=importances, )

    return fitresult, cache, report

end

function MMI.predict(model::XGBoostCount
        , fitresult
        , Xnew)
    Xmatrix = MMI.matrix(Xnew)
    return XGBoost.predict(fitresult, Xmatrix)
end


## CLASSIFIER

mutable struct XGBoostClassifier <:MMI.Probabilistic
    num_round::Int
    booster::String
    disable_default_eval_metric::Int
    eta::Float64
    gamma::Float64
    max_depth::Int
    min_child_weight::Float64
    max_delta_step::Float64
    subsample::Float64
    colsample_bytree::Float64
    colsample_bylevel::Float64
    lambda::Float64
    alpha::Float64
    tree_method::String
    sketch_eps::Float64
    scale_pos_weight::Float64
    updater::String
    refresh_leaf::Union{Int,Bool}
    process_type::String
    grow_policy::String
    max_leaves::Int
    max_bin::Int
    predictor::String
    sample_type::String
    normalize_type::String
    rate_drop::Float64
    one_drop
    skip_drop::Float64
    feature_selector::String
    top_k::Int
    tweedie_variance_power::Float64
    objective
    base_score::Float64
    eval_metric
    seed::Int
end

"""
    XGBoostClassifier(; seed=0, kwargs...)

The XGBoost model for targets with `Finite` scitype (which includes
`Binary=Finite{2}`). Gives probabilistic predictions. For admissible
`kwargs`, see
[https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html).

For a time-dependent random seed, use `seed=-1`.

See also: XGBoostCount, XGBoostRegressor

"""
function XGBoostClassifier(
    ;num_round=100
    ,booster="gbtree"
    ,disable_default_eval_metric=0
    ,eta=0.3
    ,gamma=0
    ,max_depth=6
    ,min_child_weight=1
    ,max_delta_step=0
    ,subsample=1
    ,colsample_bytree=1
    ,colsample_bylevel=1
    ,lambda=1
    ,alpha=0
    ,tree_method="auto"
    ,sketch_eps=0.03
    ,scale_pos_weight=1
    ,updater="auto"
    ,refresh_leaf=1
    ,process_type="default"
    ,grow_policy="depthwise"
    ,max_leaves=0
    ,max_bin=256
    ,predictor="cpu_predictor" #> gpu version not currently working with Julia, maybe remove completely?
    ,sample_type="uniform"
    ,normalize_type="tree"
    ,rate_drop=0.0
    ,one_drop=0
    ,skip_drop=0.0
    ,feature_selector="cyclic"
    ,top_k=0
    ,tweedie_variance_power=1.5
    ,objective="automatic"
    ,base_score=0.5
    ,eval_metric="mlogloss"
    ,seed=0)

    model = XGBoostClassifier(
    num_round
    ,booster
    ,disable_default_eval_metric
    ,eta
    ,gamma
    ,max_depth
    ,min_child_weight
    ,max_delta_step
    ,subsample
    ,colsample_bytree
    ,colsample_bylevel
    ,lambda
    ,alpha
    ,tree_method
    ,sketch_eps
    ,scale_pos_weight
    ,updater
    ,refresh_leaf
    ,process_type
    ,grow_policy
    ,max_leaves
    ,max_bin
    ,predictor #> gpu version not currently working with Julia, maybe remove completely?
    ,sample_type
    ,normalize_type
    ,rate_drop
    ,one_drop
    ,skip_drop
    ,feature_selector
    ,top_k
    ,tweedie_variance_power
    ,objective
    ,base_score
    ,eval_metric
    ,seed)

     message = MMI.clean!(model)           #> future proof by including these
     isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end


function MMI.clean!(model::XGBoostClassifier)
    warning = ""
    if(!(model.objective =="automatic"))
            warning *="Changing objective to \"automatic\", the only supported value. "
            model.objective="automatic"
    end
    return warning
end

function MMI.fit(model::XGBoostClassifier
                     , verbosity::Int     #> must be here even if unsupported in pkg
                     , X
                     , y)
    Xmatrix = MMI.matrix(X)

    a_target_element = y[1] # a CategoricalValue or CategoricalString
    num_class = length(MMI.classes(a_target_element))

    eval_metric = model.eval_metric
    if num_class == 2 && eval_metric == "mlogloss"
        eval_metric = "logloss"
    end
    if num_class > 2 && eval_metric == "logloss"
        eval_metric = "mlogloss"
    end

    y_plain = MMI.int(y) .- 1 # integer relabeling should start at 0

    # An idiosynchrony of xgboost is that num_class=1 for binary case.
    if(num_class==2)
        objective="binary:logistic"
        y_plain = convert(Array{Bool}, y_plain)
        num_class = 1
    else
        objective="multi:softprob"
    end

    silent =
        verbosity > 0 ?  false : true

    seed =
        model.seed == -1 ? generate_seed() : model.seed


    result = XGBoost.xgboost(Xmatrix, label=y_plain, model.num_round;
                             num_class=num_class,
                             kwargs(model, silent, seed, objective)...)

    fitresult = (result, a_target_element)

    features = schema(X).names

    importances = [named_importance(fi, features) for
                   fi in XGBoost.importance(result)]

    cache = nothing
    report = (feature_importances=importances, )

    return fitresult, cache, report

end


function MMI.predict(model::XGBoostClassifier
        , fitresult
        , Xnew)

    result, a_target_element = fitresult
    decode = MMI.decoder(a_target_element)
    classes = MMI.classes(a_target_element)

    Xmatrix = MMI.matrix(Xnew)
    XGBpredictions = XGBoost.predict(result, Xmatrix)

    nlevels = length(classes)
    npatterns = MMI.nrows(Xnew)

    if nlevels == 2
        true_class_probabilities = reshape(XGBpredictions, 1, npatterns)
        false_class_probabilities = 1 .- true_class_probabilities
        XGBpredictions = vcat(false_class_probabilities, true_class_probabilities)
    end

    prediction_probabilities = reshape(XGBpredictions, nlevels, npatterns)

    # note we use adjoint of above:
    predictions = MMI.UnivariateFinite(classes, prediction_probabilities')

    return predictions
end

## METADATA

XGTypes=Union{XGBoostRegressor,XGBoostCount,XGBoostClassifier}

MMI.package_name(::Type{<:XGTypes}) = "XGBoost"
MMI.package_uuid(::Type{<:XGTypes}) = "009559a3-9522-5dbb-924b-0b6ed2b22bb9"
MMI.package_url(::Type{<:XGTypes}) = "https://github.com/dmlc/XGBoost.jl"
MMI.is_pure_julia(::Type{<:XGTypes}) = false

MMI.load_path(::Type{<:XGBoostRegressor}) = "MLJModels.XGBoost_.XGBoostRegressor"
MMI.input_scitype(::Type{<:XGBoostRegressor}) = Table(Continuous)
MMI.target_scitype(::Type{<:XGBoostRegressor}) = AbstractVector{Continuous}
MMI.docstring(::Type{<:XGBoostRegressor}) =
    "The XGBoost gradient boosting method, for use with "*
    "`Continuous` univariate targets. "

MMI.load_path(::Type{<:XGBoostCount}) = "MLJModels.XGBoost_.XGBoostCount"
MMI.input_scitype(::Type{<:XGBoostCount}) = Table(Continuous)
MMI.target_scitype(::Type{<:XGBoostCount}) = AbstractVector{Count}
MMI.docstring(::Type{<:XGBoostCount}) =
    "The XGBoost gradient boosting method, for use with "*
    "`Count` univariate targets, using a Poisson objective function. "

MMI.load_path(::Type{<:XGBoostClassifier}) = "MLJModels.XGBoost_.XGBoostClassifier"
MMI.input_scitype(::Type{<:XGBoostClassifier}) = Table(Continuous)
MMI.target_scitype(::Type{<:XGBoostClassifier}) = AbstractVector{<:Finite}
MMI.docstring(::Type{<:XGBoostClassifier}) =
    "The XGBoost gradient boosting method, for use with "*
    "`Finite` univariate targets (`Multiclass`, "*
    "`OrderedFactor` and `Binary=Finite{2}`)."

end
