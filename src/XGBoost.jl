module XGBoost_

#> export the new models you're going to define (and nothing else):
export XGBoostRegressor, XGBoostClassifier, XGBoostCount

#> for all Supervised models:
import MLJBase

#> for all classifiers:
using CategoricalArrays

#> import package:
import ..XGBoost

mutable struct XGBoostRegressor{Any} <:MLJBase.Deterministic{Any}
    num_round::Integer
    booster::String
    disable_default_eval_metric::Real
    eta::Real
    gamma::Real
    max_depth::Real
    min_child_weight::Real
    max_delta_step::Real
    subsample::Real
    colsample_bytree::Real
    colsample_bylevel::Real
    lambda::Real
    alpha::Real
    tree_method::String
    sketch_eps::Real
    scale_pos_weight::Real
    updater::String
    refresh_leaf::Union{Int,Bool}
    process_type::String
    grow_policy::String
    max_leaves::Int
    max_bin::Int
    predictor::String
    sample_type::String
    normalize_type::String
    rate_drop::Real
    one_drop
    skip_drop::Real
    feature_selector::String
    top_k::Real
    tweedie_variance_power::Real
    objective
    base_score::Real
    eval_metric
    seed::Integer
    watchlist
end


"""
# constructor:
A full list of the kwargs accepted, and their value ranges, consult
https://xgboost.readthedocs.io/en/latest/parameter.html.
"""
function XGBoostRegressor(
    ;num_round=1
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
    ,updater="grow_colmaker"
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
    ,objective="reg:linear"
    ,base_score=0.5
    ,eval_metric="rmse"
    ,seed=0
    ,watchlist=[])

    model = XGBoostRegressor{Any}(
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
    ,seed
    ,watchlist)

     message = MLJBase.clean!(model)           #> future proof by including these
     isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end


function MLJBase.clean!(model::XGBoostRegressor)
    warning = ""
    if(!(model.objective in ["reg:linear","reg:gamma","reg:tweedie"]))
            warning *="\n objective function is not valid and has been changed to reg:linear"
            model.objective="reg:linear"
    end
    return warning
end


#> The following optional method (the fallback does nothing, returns
#> empty warning) is called by the constructor above but also by the
#> fit methods below:

#> A required `fit` method returns `fitresult, cache, report`. (Return
#> `cache=nothing` unless you are overloading `update`)

function MLJBase.fit(model::XGBoostRegressor
             , verbosity::Int     #> must be here even if unsupported in pkg
             , X
             , y)

             silent =
                 verbosity > 0 ?  0 : 1
    Xmatrix = MLJBase.matrix(X)
    dm = XGBoost.DMatrix(Xmatrix,label=y)

    fitresult = XGBoost.xgboost(dm
                               , model.num_round
                               , booster = model.booster
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
                               , updater = model.updater
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
                               , objective = model.objective
                               , base_score = model.base_score
                               , eval_metric=model.eval_metric
                               , seed = model.seed
                               , watchlist=model.watchlist)

    #> return package-specific statistics (eg, feature rankings,
    #> internal estimates of generalization error) in `report`, which
    #> should be `nothing` or a dictionary keyed on symbols.

    cache = nothing
    report = nothing

    return fitresult, cache, report

end


function MLJBase.predict(model::XGBoostRegressor
        , fitresult
        , Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return XGBoost.predict(fitresult, Xmatrix)
end







mutable struct XGBoostCount{Any} <:MLJBase.Deterministic{Any}
    num_round::Integer
    booster::String
    disable_default_eval_metric::Real
    eta::Real
    gamma::Real
    max_depth::Real
    min_child_weight::Real
    max_delta_step::Real
    subsample::Real
    colsample_bytree::Real
    colsample_bylevel::Real
    lambda::Real
    alpha::Real
    tree_method::String
    sketch_eps::Real
    scale_pos_weight::Real
    updater::String
    refresh_leaf::Union{Int,Bool}
    process_type::String
    grow_policy::String
    max_leaves::Int
    max_bin::Int
    predictor::String
    sample_type::String
    normalize_type::String
    rate_drop::Real
    one_drop
    skip_drop::Real
    feature_selector::String
    top_k::Real
    tweedie_variance_power::Real
    objective
    base_score::Real
    eval_metric
    seed::Integer
    watchlist
end


"""
# constructor:
A full list of the kwargs accepted, and their value ranges, consult
https://xgboost.readthedocs.io/en/latest/parameter.html.
"""
function XGBoostCount(
    ;num_round=1
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
    ,updater="grow_colmaker"
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
    ,seed=0
    ,watchlist=[])

    model = XGBoostCount{Any}(
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
    ,seed
    ,watchlist)

     message = MLJBase.clean!(model)           #> future proof by including these
     isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end


function MLJBase.clean!(model::XGBoostCount)
    warning = ""
    if(!(model.objective in ["count:poisson"]))
            warning *="\n objective function is not valid and has been changed to count:poisson"
            model.objective="count:poisson"
    end
    return warning
end



function MLJBase.fit(model::XGBoostCount
             , verbosity::Int     #> must be here even if unsupported in pkg
             , X
             , y)

             silent =
                 verbosity > 0 ?  0 : 1

    Xmatrix = MLJBase.matrix(X)
    dm = XGBoost.DMatrix(Xmatrix,label=y)

    fitresult = XGBoost.xgboost(dm
                               , model.num_round
                               , booster = model.booster
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
                               , updater = model.updater
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
                               , objective = model.objective
                               , base_score = model.base_score
                               , eval_metric=model.eval_metric
                               , seed = model.seed
                               , watchlist=model.watchlist)

    #> return package-specific statistics (eg, feature rankings,
    #> internal estimates of generalization error) in `report`, which
    #> should be `nothing` or a dictionary keyed on symbols.

    cache = nothing
    report = nothing

    return fitresult, cache, report

end

function MLJBase.predict(model::XGBoostCount
        , fitresult
        , Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return XGBoost.predict(fitresult, Xmatrix)
end



mutable struct XGBoostClassifier{Any} <:MLJBase.Probabilistic{Any}
    num_round::Integer
    booster::String
    disable_default_eval_metric::Real
    eta::Real
    gamma::Real
    max_depth::Real
    min_child_weight::Real
    max_delta_step::Real
    subsample::Real
    colsample_bytree::Real
    colsample_bylevel::Real
    lambda::Real
    alpha::Real
    tree_method::String
    sketch_eps::Real
    scale_pos_weight::Real
    updater::String
    refresh_leaf::Union{Int,Bool}
    process_type::String
    grow_policy::String
    max_leaves::Int
    max_bin::Int
    predictor::String
    sample_type::String
    normalize_type::String
    rate_drop::Real
    one_drop
    skip_drop::Real
    feature_selector::String
    top_k::Real
    tweedie_variance_power::Real
    objective
    base_score::Real
    eval_metric
    seed::Integer
    watchlist
end


"""
# constructor:
A full list of the kwargs accepted, and their value ranges, consult
https://xgboost.readthedocs.io/en/latest/parameter.html.
"""
function XGBoostClassifier(
    ;num_round=1
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
    ,updater="grow_colmaker"
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
    ,objective=":automatic"
    ,base_score=0.5
    ,eval_metric="rmse"
    ,seed=0
    ,watchlist=[])

    model = XGBoostClassifier{Any}(
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
    ,seed
    ,watchlist)

     message = MLJBase.clean!(model)           #> future proof by including these
     isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end


function MLJBase.clean!(model::XGBoostClassifier)
    warning = ""
    if(!(model.objective ==":automatic"))
            warning *="\n objective function is more suited to :automatic"
            model.objective=":automatic"
    end
    return warning
end






#> The following optional method (the fallback does nothing, returns
#> empty warning) is called by the constructor above but also by the
#> fit methods below:

#> A required `fit` method returns `fitresult, cache, report`. (Return
#> `cache=nothing` unless you are overloading `update`)

function MLJBase.fit(model::XGBoostClassifier
             , verbosity::Int     #> must be here even if unsupported in pkg
             , X
             , y)
             Xmatrix = MLJBase.matrix(X)
             classes = levels(y) # *all* levels in pool of y, not just observed ones
             num_class = length(classes)
             decoder = MLJBase.CategoricalDecoder(y, Int)
             y_plain = MLJBase.transform(decoder, y)
             if(num_class==2)
                 model.objective="binary:logistic"
             else
                 model.objective="multi:softprob"
             end
             @show(num_class)
             @show(model.objective)
             silent =
                 verbosity > 0 ?  1 : 1
                 #classifier case currently doesn't accept different silent, check

    result = XGBoost.xgboost(Xmatrix, label=y_plain
                               , model.num_round
                               , booster = model.booster
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
                               , updater = model.updater
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
                               , objective = model.objective
                               , base_score = model.base_score
                               , eval_metric=model.eval_metric
                               , seed = model.seed
                               , watchlist=model.watchlist
                               , num_class=num_class)

    fitresult = (result, decoder)

    #> return package-specific statistics (eg, feature rankings,
    #> internal estimates of generalization error) in `report`, which
    #> should be `nothing` or a dictionary keyed on symbols.

    cache = nothing
    report = nothing

    return fitresult, cache, report

end


function MLJBase.predict(model::XGBoostClassifier
        , fitresult
        , Xnew)

    result,decoder = fitresult
    Xmatrix = MLJBase.matrix(Xnew)
    prediction = XGBoost.predict(result, Xmatrix)
    cols = length((decoder.pool).levels)
    rows = Int(length(prediction)/cols)
    preds = reshape(prediction,cols,rows)
    #return MLJBase.UnivariateNominal(decoder,prediction)
    result=[]
    for i=1:rows
        single = preds[:,i]
        d = MLJBase.UnivariateNominal(levels(decoder), single)
        result=[result;d]
    end
    return result
end





XGTypes=Union{XGBoostRegressor,XGBoostCount,XGBoostClassifier}


MLJBase.package_name(::Type{<:XGTypes}) = "XGBoost"
MLJBase.package_uuid(::Type{<:XGTypes}) = "009559a3-9522-5dbb-924b-0b6ed2b22bb9"
MLJBase.package_url(::Type{<:XGTypes}) = "https://github.com/dmlc/XGBoost.jl"
MLJBase.is_pure_julia(::Type{<:XGTypes}) = false

MLJBase.load_path(::Type{<:XGBoostRegressor}) = "MLJModels.XGBoost_.XGBoostRegressor"
MLJBase.input_scitypes(::Type{<:XGBoostRegressor}) = MLJBase.Continuous
MLJBase.target_scitype(::Type{<:XGBoostRegressor}) = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:XGBoostRegressor}) = true


MLJBase.load_path(::Type{<:XGBoostCount}) = "MLJModels.XGBoost_.XGBoostCount"
MLJBase.input_scitypes(::Type{<:XGBoostCount}) = MLJBase.Continuous
MLJBase.target_scitype(::Type{<:XGBoostCount}) = MLJBase.Count
MLJBase.input_is_multivariate(::Type{<:XGBoostCount}) = true


MLJBase.load_path(::Type{<:XGBoostClassifier}) = "MLJModels.XGBoost_.XGBoostCount"
MLJBase.input_scitypes(::Type{<:XGBoostClassifier}) = MLJBase.Continuous
MLJBase.target_scitype(::Type{<:XGBoostClassifier}) = Union{MLJBase.Multiclass,MLJBase.FiniteOrderedFactor}
MLJBase.input_is_multivariate(::Type{<:XGBoostClassifier}) = true


end
