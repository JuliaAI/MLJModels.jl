# | model                  | build  | fitted_params | report | metadata | tests 1 | tests 2 |
# | ---------------------- | ------ | ------------- | ------ | -------- | ------- | ------- |
# | AdaboostClassif        | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | AdaboostReg            | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | BaggingClassif         | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | BaggingReg             | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | ExtraTreesClassif      | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | ExtraTreesReg          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | GDBClassif             | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | GDBReg                 | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | RFClassif              | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | RFReg                  | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | VotingClassif          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | VotingReg              | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | HGBClassif             | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | HGBReg                 | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | RANSACClassif          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | RANSACReg              | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |

AdaBoostRegressor_ = SKEN.AdaBoostRegressor
@sk_model mutable struct AdaBoostRegressor <: MLJBase.Deterministic
    base_estimator::Any    = nothing
    n_estimators::Int      = 50::(_ > 0)
    learning_rate::Float64 = 1.0::(_ > 0)
    loss::String           = "linear"::(_ in ("linear","square","exponential"))
    random_state::Any      = nothing
end
MLJBase.fitted_params(model::AdaBoostRegressor, (f, _, _)) = (
    estimators           = f.estimators_,
    estimator_weights    = f.estimator_weights_,
    estimator_errors     = f.estimator_errors_,
    feature_importances_ = f.feature_importances_
    )

# ----------------------------------------------------------------------------
AdaBoostClassifier_ = SKEN.AdaBoostClassifier
@sk_model mutable struct AdaBoostClassifier <: MLJBase.Probabilistic
    base_estimator::Any    = nothing
    n_estimators::Int      = 50::(_ > 0)
    learning_rate::Float64 = 1.0::(_ > 0)
    algorithm::String      = "SAMME.R"::(_ in ("SAMME", "SAMME.R"))
    random_state::Any      = nothing
end
MLJBase.fitted_params(m::AdaBoostClassifier, (f, _, _)) = (
    estimators           = f.estimators_,
    estimator_weights    = f.estimator_weights_,
    estimator_errors     = f.estimator_errors_,
    classes              = f.classes_,
    n_classes            = f.n_classes_
    )
metadata_model(AdaBoostClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Adaboost ensemble classifier."
    )

# ============================================================================
BaggingRegressor_ = SKEN.BaggingRegressor
@sk_model mutable struct BaggingRegressor <: MLJBase.Deterministic
    base_estimator::Any = nothing
    n_estimators::Int   = 10::(_>0)
    max_samples::Union{Int,Float64}  = 1.0::(_>0)
    max_features::Union{Int,Float64} = 1.0::(_>0)
    bootstrap::Bool     = true
    bootstrap_features::Bool = false
    oob_score::Bool     = false
    warm_start::Bool    = false
    n_jobs::Option{Int} = nothing
    random_state::Any   = nothing
    verbose::Int        = 0
end
MLJBase.fitted_params(model::BaggingRegressor, (f, _, _)) = (
    estimators           = f.estimators_,
    estimators_samples   = f.estimators_samples_,
    estimators_features  = f.estimators_features_,
    oob_score            = model.oob_score ? f.oob_score_ : nothing,
    oob_prediction       = model.oob_score ? f.oob_prediction_ : nothing
    )

# ----------------------------------------------------------------------------
BaggingClassifier_ = SKEN.BaggingClassifier
@sk_model mutable struct BaggingClassifier <: MLJBase.Probabilistic
    base_estimator::Any = nothing
    n_estimators::Int   = 10::(_>0)
    max_samples::Union{Int,Float64}  = 1.0::(_>0)
    max_features::Union{Int,Float64} = 1.0::(_>0)
    bootstrap::Bool     = true
    bootstrap_features::Bool = false
    oob_score::Bool     = false
    warm_start::Bool    = false
    n_jobs::Option{Int} = nothing
    random_state::Any   = nothing
    verbose::Int        = 0
end
MLJBase.fitted_params(m::BaggingClassifier, (f, _, _)) = (
    base_estimator       = f.base_estimator_,
    estimators           = f.estimators_,
    estimators_samples   = f.estimators_samples_,
    estimators_features  = f.estimators_features_,
    classes              = f.classes_,
    n_classes            = f.n_classes_,
    oob_score            = m.oob_score ? f.oob_score_ : nothing,
    oob_decision_function = m.oob_score ? f.oob_decision_function_ : nothing
    )
metadata_model(BaggingClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Bagging ensemble classifier."
    )

# ============================================================================
GradientBoostingRegressor_ = SKEN.GradientBoostingRegressor
@sk_model mutable struct GradientBoostingRegressor <: MLJBase.Deterministic
    loss::String           = "ls"::(_ in ("ls","lad","huber","quantile"))
    learning_rate::Float64 = 0.1::(_>0)
    n_estimators::Int      = 100::(_>0)
    subsample::Float64     = 1.0::(_>0)
    criterion::String      = "friedman_mse"::(_ in ("mse","mae","friedman_mse"))
    min_samples_split::Union{Int,Float64} = 2::(_>0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_>0)
    min_weight_fraction_leaf::Float64     = 0.0::(_≥0)
    max_depth::Int         = 3::(_>0)
    min_impurity_decrease::Float64 = 0.0::(_≥0)
#    min_impurity_split::Float64    = 1e-7::(_>0) # deprecated in favour of min_decrease
    init::Any             = nothing
    random_state::Any     = nothing
    max_features::Union{Int,Float64,String,Nothing} = nothing::(_===nothing || (isa(_,String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    alpha::Float64        = 0.9::(_>0)
    verbose::Int          = 0
    max_leaf_nodes::Option{Int} = nothing::(_===nothing || _>0)
    warm_start::Bool      = false
    presort::Union{Bool,String}  = "auto"::(isa(_, Bool) || _ == "auto")
    validation_fraction::Float64 = 0.1::(_>0)
    n_iter_no_change::Option{Int} = nothing
    tol::Float64          = 1e-4::(_>0)
end
MLJBase.fitted_params(m::GradientBoostingRegressor, (f, _, _)) = (
    feature_importances = f.feature_importances_,
    train_score         = f.train_score_,
    loss                = f.loss_,
    init                = f.init_,
    estimators          = f.estimators_,
    oob_improvement     = m.subsample < 1 ? f.oob_improvement_ : nothing
    )

# ----------------------------------------------------------------------------
GradientBoostingClassifier_ = SKEN.GradientBoostingClassifier
@sk_model mutable struct GradientBoostingClassifier <: MLJBase.Probabilistic
    loss::String           = "deviance"::(_ in ("deviance","exponential"))
    learning_rate::Float64 = 0.1::(_>0)
    n_estimators::Int      = 100::(_>0)
    subsample::Float64     = 1.0::(_>0)
    criterion::String      = "friedman_mse"::(_ in ("mse","mae","friedman_mse"))
    min_samples_split::Union{Int,Float64} = 2::(_>0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_>0)
    min_weight_fraction_leaf::Float64     = 0.0::(_≥0)
    max_depth::Int         = 3::(_>0)
    min_impurity_decrease::Float64 = 0.0::(_≥0)
#    min_impurity_split::Float64    = 1e-7::(_>0) # deprecated in favour of min_decrease
    init::Any             = nothing
    random_state::Any     = nothing
    max_features::Union{Int,Float64,String,Nothing} = nothing::(_===nothing || (isa(_,String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    verbose::Int          = 0
    max_leaf_nodes::Option{Int}    = nothing::(_===nothing || _>0)
    warm_start::Bool      = false
    presort::Union{Bool,String}    = "auto"::(isa(_, Bool) || _ == "auto")
    validation_fraction::Float64   = 0.1::(_>0)
    n_iter_no_change::Option{Int}  = nothing
    tol::Float64          = 1e-4::(_>0)
end
MLJBase.fitted_params(m::GradientBoostingClassifier, (f, _, _)) = (
    n_estimators        = f.n_estimators_,
    feature_importances = f.feature_importances_,
    train_score         = f.train_score_,
    loss                = f.loss_,
    init                = f.init_,
    estimators          = f.estimators_,
    oob_improvement     = m.subsample < 1 ? f.oob_improvement_ : nothing
    )
metadata_model(GradientBoostingClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Gradient boosting ensemble classifier."
    )

# ============================================================================
# HistGradientBoostingRegressor_ = SKEN.HistGradientBoostingRegressor
# @sk_model mutable struct HistGradientBoostingRegressor <: MLJBase.Deterministic
#     loss::String           = "least_squares"::(_ == "least_squares") # currently the only option
#     learning_rate::Float64 = 0.1::(_>0)
#     max_iter::Int          = 100::(_>0)
#     max_leaf_nodes::Int    = 31::(_>0)
#     max_depth::Option{Int} = nothing::(_===nothing || _>0)
#     min_samples_leaf::Int  = 20::(_>0)
#     l2_regularization::Float64 = 0.0::(_≥0)
#     max_bins::Int          = 256::(_>0)
#     scoring::Any           = nothing
#     validation_fraction::Union{Int,Float64,Nothing} = 0.1::(_===nothing || _>0)
#     n_iter_no_change::Union{Int, Nothing}           = nothing::(_===nothing || _>0)
#     tol::Float64           = 1e-7::(_>0)
#     random_state::Any      = nothing
# end
# MLJBase.fitted_params(model::HistGradientBoostingRegressor, (f, _, _)) = (
#     n_trees_per_iteration = f.n_trees_per_iteration_,
#     train_score           = f.train_score_,
#     validation_score      = f.validation_score_
#     )

# ============================================================================
RandomForestRegressor_ = SKEN.RandomForestRegressor
@sk_model mutable struct RandomForestRegressor <: MLJBase.Deterministic
    n_estimators::Int   = 100::(_ > 0)
    criterion::String   = "mse"::(_ in ("mae", "mse"))
    max_depth::Option{Int}                = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64} = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_ > 0)
    min_weight_fraction_leaf::Float64     = 0.0::(_ ≥ 0)
    max_features::Union{Int,Float64,String,Nothing} = "auto"::(_ === nothing || (isa(_, String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    max_leaf_nodes::Option{Int}           = nothing::(_ === nothing || _ > 0)
    min_impurity_decrease::Float64        = 0.0::(_ ≥ 0)
#    min_impurity_split::Float64    = 1e-7::(_>0) # deprecated in favour of min_decrease
    bootstrap::Bool     = true
    oob_score::Bool     = false
    n_jobs::Option{Int} = nothing
    random_state::Any   = nothing
    verbose::Int        = 0
    warm_start::Bool    = false
end
MLJBase.fitted_params(model::RandomForestRegressor, (f, _, _)) = (
    estimators     = f.estimators_,
    feature_importances = f.feature_importances_,
    n_features     = f.n_features_,
    n_outputs      = f.n_outputs_,
    oob_score      = model.oob_score ? f.oob_score_ : nothing,
    oob_prediction = model.oob_score ? f.oob_prediction_ : nothing
    )

# ----------------------------------------------------------------------------
RandomForestClassifier_ = SKEN.RandomForestClassifier
@sk_model mutable struct RandomForestClassifier <: MLJBase.Probabilistic
    n_estimators::Int   = 100::(_ > 0)
    criterion::String   = "gini"::(_ in ("gini","entropy"))
    max_depth::Option{Int}                = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64} = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_ > 0)
    min_weight_fraction_leaf::Float64     = 0.0::(_ ≥ 0)
    max_features::Union{Int,Float64,String,Nothing} = "auto"::(_ === nothing || (isa(_, String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    max_leaf_nodes::Option{Int}           = nothing::(_ === nothing || _ > 0)
    min_impurity_decrease::Float64        = 0.0::(_ ≥ 0)
#    min_impurity_split::Float64    = 1e-7::(_>0) # deprecated in favour of min_decrease
    bootstrap::Bool     = true
    oob_score::Bool     = false
    n_jobs::Option{Int} = nothing
    random_state::Any   = nothing
    verbose::Int        = 0
    warm_start::Bool    = false
    class_weight::Any   = nothing
end
MLJBase.fitted_params(m::RandomForestClassifier, (f, _, _)) = (
    estimators          = f.estimators_,
    classes             = f.classes_,
    n_classes           = f.n_classes_,
    n_features          = f.n_features_,
    n_outputs           = f.n_outputs_,
    feature_importances = f.feature_importances_,
    oob_score            = m.oob_score ? f.oob_score_ : nothing,
    oob_decision_function = m.oob_score ? f.oob_decision_function_ : nothing
    )
metadata_model(RandomForestClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Random forest classifier."
    )

# ============================================================================
# ExtraTreeRegressor_ = SKEN.ExtraTreeRegressor
# @sk_model mutable struct ExtraTreeRegressor <: MLJBase.Deterministic
#     n_estimators::Int   = 100::(_>0)
#     criterion::String   = "mse"::(_ in ("mae","mse"))
#     max_depth::Union{Int,Nothing}          = nothing::(_===nothing || _>0)
#     min_samples_split::Union{Int,Float64}  = 2::(_>0)
#     min_samples_leaf::Union{Int, Float64}  = 1::(_>0)
#     min_weight_fraction_leaf::Float64      = 0.0::(_≥0)
#     max_features::Union{Int,Float64,String,Nothing} = "auto"::(_===nothing || (isa(_,String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
#     max_leaf_nodes::Union{Int,Nothing}     = nothing::(_===nothing || _>0)
#     min_impurity_decrease::Float64         = 0.0::(_≥0)
#     min_impurity_split::Float64            = 1e-7::(_≥0)
#     bootstrap::Bool     = true
#     oob_score::Bool     = false
#     n_jobs::Option{Int} = nothing
#     random_state::Any   = nothing
#     verbose::Int        = 0
#     warm_start::Bool    = false
# end
# MLJBase.fitted_params(model::ExtraTreeRegressor, (f, _, _)) = (
#     estimators     = f.estimators_,
#     feature_importances = f.feature_importances_,
#     n_features     = f.n_features_,
#     n_outputs      = f.n_outputs_,
#     oob_score      = f.oob_score_,
#     oob_prediction = f.oob_prediction_
#     )

const ENSEMBLE_REG = Union{Type{<:AdaBoostRegressor}, Type{<:BaggingRegressor}, Type{<:GradientBoostingRegressor}, Type{<:RandomForestRegressor}}

MLJBase.input_scitype(::ENSEMBLE_REG)  = MLJBase.Table(MLJBase.Continuous)
MLJBase.target_scitype(::ENSEMBLE_REG) = AbstractVector{MLJBase.Continuous}
