# | model                  | build  | fitted_params | report | metadata | tests 1 | tests 2 |
# | ---------------------- | ------ | ------------- | ------ | -------- | ------- | ------- |
# | AdaboostClassif        | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | AdaboostReg            | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | BaggingClassif         | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | BaggingReg             | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | ExtraTreesClassif      | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | ExtraTreesReg          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | GDBClassif             | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | GDBReg                 | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | RFClassif              | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | RFReg                  | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | VotingClassif          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | VotingReg              | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | HGBClassif             | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | HGBReg                 | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | RANSACClassif          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | RANSACReg              | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |

# other models to investigate: IsolationForest, RandomTreesEmbedding

export AdaBoostRegressor,          # AdaBoostClassifier,
       BaggingRegressor,           # BaggingClassifier,
                                   # ExtraTreeRegressor, ExtraTreeClassifier,
       GradientBoostingRegressor,  # GradientBoostingClassifier,
       RandomForestRegressor       # RandomForestClassifier,
                                   # VotingRegressor, VotingClassifier,
                                   # HistGradientBoostingRegressor, HistGradientBoostingClassifier


# ==============================================================================
AdaBoostRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).AdaBoostRegressor
@sk_model mutable struct AdaBoostRegressor <: MLJBase.Deterministic
    base_estimator::Any    = nothing
    n_estimators::Int      = 50::(arg>0)
    learning_rate::Float64 = 1.0::(arg>0)
    loss::String           = "linear"::(arg in ("linear","square","exponential"))
    random_state::Any      = nothing
end
MLJBase.fitted_params(model::AdaBoostRegressor, (fitresult, _)) = (
    estimators           = fitresult.estimators_,
    estimator_weights    = fitresult.estimator_weights_,
    estimator_errors     = fitresult.estimator_errors_,
    feature_importances_ = fitresult.feature_importances_
    )

# ==============================================================================
BaggingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).BaggingRegressor
@sk_model mutable struct BaggingRegressor <: MLJBase.Deterministic
    base_estimator::Any = nothing
    n_estimators::Int   = 10::(arg>0)
    max_samples::Union{Int,Float64}  = 1.0::(arg>0)
    max_features::Union{Int,Float64} = 1.0::(arg>0)
    bootstrap::Bool     = true
    bootstrap_features::Bool = false
    oob_score::Bool     = false
    warm_start::Bool    = false
    n_jobs::Union{Nothing,Int} = nothing
    random_state::Any   = nothing
    verbose::Int        = 0
end
MLJBase.fitted_params(model::BaggingRegressor, (fitresult, _)) = (
    estimators           = fitresult.estimators_,
    estimators_samples   = fitresult.estimators_samples_,
    estimators_features  = fitresult.estimators_features_,
    oob_score            = model.oob_score ? fitresult.oob_score_ : nothing,
    oob_prediction       = model.oob_score ? fitresult.oob_prediction_ : nothing
    )

# ==============================================================================
GradientBoostingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).GradientBoostingRegressor
@sk_model mutable struct GradientBoostingRegressor <: MLJBase.Deterministic
    loss::String           = "ls"::(arg in ("ls","lad","huber","quantile"))
    learning_rate::Float64 = 0.1::(arg>0)
    n_estimators::Int      = 100::(arg>0)
    subsample::Float64     = 1.0::(arg>0)
    criterion::String      = "friedman_mse"::(arg in ("mse","mae","friedman_mse"))
    min_samples_split::Union{Int,Float64} = 2::(arg>0)
    min_samples_leaf::Union{Int,Float64}  = 1::(arg>0)
    min_weight_fraction_leaf::Float64     = 0.0::(arg≥0)
    max_depth::Int         = 3::(arg>0)
    min_impurity_decrease::Float64 = 0.0::(arg≥0)
#    min_impurity_split::Float64    = 1e-7::(arg>0) # deprecated in favour of min_decrease
    init::Any             = nothing
    random_state::Any     = nothing
    max_features::Union{Int,Float64,String,Nothing} = nothing::(arg===nothing || (isa(arg,String) && (arg in ("auto","sqrt","log2"))) || arg > 0)
    alpha::Float64        = 0.9::(arg>0)
    verbose::Int          = 0
    max_leaf_nodes::Union{Nothing,Int} = nothing::(arg===nothing || arg>0)
    warm_start::Bool      = false
    presort::Union{Bool,String}  = "auto"::(isa(arg, Bool) || arg == "auto")
    validation_fraction::Float64 = 0.1::(arg>0)
    n_iter_no_change::Union{Nothing,Int} = nothing
    tol::Float64          = 1e-4::(arg>0)
end
MLJBase.fitted_params(model::GradientBoostingRegressor, (fitresult, _)) = (
    feature_importances = fitresult.feature_importances_,
#    oob_improvement     = fitresult.oob_improvement_, # not found ?
    train_score         = fitresult.train_score_,
    loss                = fitresult.loss_,
    init                = fitresult.init_,
    estimators          = fitresult.estimators_
    )

# ==============================================================================
# HistGradientBoostingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).HistGradientBoostingRegressor
# @sk_model mutable struct HistGradientBoostingRegressor <: MLJBase.Deterministic
#     loss::String           = "least_squares"::(arg == "least_squares") # currently the only option
#     learning_rate::Float64 = 0.1::(arg>0)
#     max_iter::Int          = 100::(arg>0)
#     max_leaf_nodes::Int    = 31::(arg>0)
#     max_depth::Union{Nothing,Int} = nothing::(arg===nothing || arg>0)
#     min_samples_leaf::Int  = 20::(arg>0)
#     l2_regularization::Float64 = 0.0::(arg≥0)
#     max_bins::Int          = 256::(arg>0)
#     scoring::Any           = nothing
#     validation_fraction::Union{Int,Float64,Nothing} = 0.1::(arg===nothing || arg>0)
#     n_iter_no_change::Union{Int, Nothing}           = nothing::(arg===nothing || arg>0)
#     tol::Float64           = 1e-7::(arg>0)
#     random_state::Any      = nothing
# end
# MLJBase.fitted_params(model::HistGradientBoostingRegressor, (fitresult, _)) = (
#     n_trees_per_iteration = fitresult.n_trees_per_iteration_,
#     train_score           = fitresult.train_score_,
#     validation_score      = fitresult.validation_score_
#     )

# ==============================================================================
RandomForestRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).RandomForestRegressor
@sk_model mutable struct RandomForestRegressor <: MLJBase.Deterministic
    n_estimators::Int   = 100::(arg>0)
    criterion::String   = "mse"::(arg in ("mae","mse"))
    max_depth::Union{Int,Nothing}          = nothing::(arg===nothing || arg>0)
    min_samples_split::Union{Int,Float64}  = 2::(arg>0)
    min_samples_leaf::Union{Int, Float64}  = 1::(arg>0)
    min_weight_fraction_leaf::Float64      = 0.0::(arg≥0)
    max_features::Union{Int,Float64,String,Nothing} = "auto"::(arg===nothing || (isa(arg,String) && (arg in ("auto","sqrt","log2"))) || arg > 0)
    max_leaf_nodes::Union{Int,Nothing}     = nothing::(arg===nothing || arg>0)
    min_impurity_decrease::Float64         = 0.0::(arg≥0)
#    min_impurity_split::Float64    = 1e-7::(arg>0) # deprecated in favour of min_decrease
    bootstrap::Bool     = true
    oob_score::Bool     = false
    n_jobs::Union{Nothing,Int} = nothing
    random_state::Any   = nothing
    verbose::Int        = 0
    warm_start::Bool    = false
end
MLJBase.fitted_params(model::RandomForestRegressor, (fitresult, _)) = (
    estimators     = fitresult.estimators_,
    feature_importances = fitresult.feature_importances_,
    n_features     = fitresult.n_features_,
    n_outputs      = fitresult.n_outputs_,
    oob_score      = model.oob_score ? fitresult.oob_score_ : nothing,
    oob_prediction = model.oob_score ? fitresult.oob_prediction_ : nothing
    )

# ==============================================================================
# ExtraTreeRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).ExtraTreeRegressor
# @sk_model mutable struct ExtraTreeRegressor <: MLJBase.Deterministic
#     n_estimators::Int   = 100::(arg>0)
#     criterion::String   = "mse"::(arg in ("mae","mse"))
#     max_depth::Union{Int,Nothing}          = nothing::(arg===nothing || arg>0)
#     min_samples_split::Union{Int,Float64}  = 2::(arg>0)
#     min_samples_leaf::Union{Int, Float64}  = 1::(arg>0)
#     min_weight_fraction_leaf::Float64      = 0.0::(arg≥0)
#     max_features::Union{Int,Float64,String,Nothing} = "auto"::(arg===nothing || (isa(arg,String) && (arg in ("auto","sqrt","log2"))) || arg > 0)
#     max_leaf_nodes::Union{Int,Nothing}     = nothing::(arg===nothing || arg>0)
#     min_impurity_decrease::Float64         = 0.0::(arg≥0)
#     min_impurity_split::Float64            = 1e-7::(arg≥0)
#     bootstrap::Bool     = true
#     oob_score::Bool     = false
#     n_jobs::Union{Nothing,Int} = nothing
#     random_state::Any   = nothing
#     verbose::Int        = 0
#     warm_start::Bool    = false
# end
# MLJBase.fitted_params(model::ExtraTreeRegressor, (fitresult, _)) = (
#     estimators     = fitresult.estimators_,
#     feature_importances = fitresult.feature_importances_,
#     n_features     = fitresult.n_features_,
#     n_outputs      = fitresult.n_outputs_,
#     oob_score      = fitresult.oob_score_,
#     oob_prediction = fitresult.oob_prediction_
#     )

const ENSEMBLE_REG = Union{Type{<:AdaBoostRegressor}, Type{<:BaggingRegressor}, Type{<:GradientBoostingRegressor}, Type{<:RandomForestRegressor}}

MLJBase.input_scitype(::ENSEMBLE_REG)  = MLJBase.Table(MLJBase.Continuous)
MLJBase.target_scitype(::ENSEMBLE_REG) = AbstractVector{MLJBase.Continuous}
