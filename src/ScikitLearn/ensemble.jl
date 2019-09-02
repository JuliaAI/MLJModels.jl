# | model                  | build  | fitted_params | report | metadata | tests 1 | tests 2 |
# | ---------------------- | ------ | ------------- | ------ | -------- | ------- | ------- |
# | AdaboostClassif        | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | AdaboostReg            | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | BaggingClassif         | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | BaggingReg             | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | ExtraTreesClassif      | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | ExtraTreesReg          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | GDBClassif             | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | GDBReg                 | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | RFClassif              | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | RFReg                  | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | VotingClassif          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | VotingReg              | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | HGBClassif             | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | HGBReg                 | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |

export AdaBoostClassifier, AdaBoostRegressor,
       BaggingClassifier, BaggingRegressor,
       ExtraTreesClassifier, ExtraTreesRegressor,
       GradientBoostingClassifier, GradientBoostingRegressor,
       RandomForestClassifier, RandomForestRegressor,
       VotingClassifier, VotingRegressor
       HistGradientBoostingClassifier, HistGradientBoostingRegressor

# other models to investigate
#
# IsolationForest, RandomTreesEmbedding,


AdaBoostRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).AdaBoostRegressor
@sk_model mutable struct AdaBoostRegressor <: MLJBase.Deterministic
    base_estimator::Any    = nothing
    n_estimators::Int      = 50
    learning_rate::Float64 = 1.0
    loss::Any              = "linear"::(arg in ("linear","square","exponential"))
    random_state::Union{Nothing,Int} = nothing
end

BaggingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).BaggingRegressor
@sk_model mutable struct BaggingRegressor <: MLJBase.Deterministic
    base_estimator::Any = nothing
    n_estimators::Int   = 10
    max_samples::Union{Int, Float64}  = 1.0
    max_features::Union{Int, Float64} = 1.0
    bootstrap::Bool     = true
    bootstrap_features::Bool = false
    oob_score::Bool     = false
    warm_start::Bool    = false
    n_jobs::Union{Nothing,Int} = nothing
    random_state::Any   = nothing
    verbose::Int        = 0
end

GradientBoostingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).GradientBoostingRegressor
@sk_model mutable struct GradientBoostingRegressor <: MLJBase.Deterministic
    loss::String = "ls"::(arg in ("ls","lad","huber","quantile"))
    learning_rate::Float64 = 0.1
    n_estimators::Int      = 100
    subsample::Float64     = 1.0
    criterion::String      = "friedman_mse"
    min_samples_split::Union{Int,Float64} = 2
    min_samples_leaf::Union{Int,Float64}  = 1
    min_weight_fraction_leaf::Float64     = 0.0
    max_depth::Int         = 3
    min_impurity_decrease::Float64 = 0.0
    min_impurity_split::Float64    = 1e-7
    init::Union{Any, Any} = nothing
    random_state::Any     = nothing
    max_features::Any     = nothing
    alpha::Float64        = 0.9
    verbose::Int          = 0
    max_leaf_nodes::Union{Int, Any} = nothing
    warm_start::Bool      = false
    presort::Union{Bool, Any}       = "auto"
    validation_fraction::Float64    = 0.1
    n_iter_no_change::Union{Nothing,Int} = nothing
    tol::Float64          = 1e-4::(arg>0)
end

# HistGradientBoostingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).HistGradientBoostingRegressor
# @sk_model mutable struct HistGradientBoostingRegressor <: MLJBase.Deterministic
#     loss::Any = "least_squares"
#     learning_rate::Float64 = 0.1
#     max_iter::Int = 100
#     max_leaf_nodes::Int = 31
#     max_depth::Union{Int, Nothing} = nothing
#     min_samples_leaf::Int = 20
#     l2_regularization::Float64 = 0.0
#     max_bins::Int = 256
#     scoring::Any = nothing
#     validation_fraction::Union{Int, Float64, Nothing} = 0.1
#     n_iter_no_change::Union{Int, Nothing} = nothing
#     tol::Union{Float64, Any} = 1e-7
#     random_state::Any = nothing
# end

RandomForestRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).RandomForestRegressor
@sk_model mutable struct RandomForestRegressor <: MLJBase.Deterministic
    n_estimators::Int   = 10::(arg>0)
    criterion::String   = "mse"
    max_depth::Union{Int, Nothing}          = nothing
    min_samples_split::Union{Int, Float64}  = 2
    min_samples_leaf::Union{Int, Float64}   = 1
    min_weight_fraction_leaf::Float64       = 0.0
    max_features::Union{Int, Float64, String, Nothing} = "auto"
    max_leaf_nodes::Union{Int, Nothing}     = nothing
    min_impurity_decrease::Float64          = 0.0
    min_impurity_split::Float64             = 1e-7
    bootstrap::Bool     = true
    oob_score::Bool     = false
    n_jobs::Union{Nothing,Int} = nothing
    random_state::Any   = nothing
    verbose::Int        = 0
    warm_start::Bool    = false
end

VotingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).VotingRegressor
@sk_model mutable struct VotingRegressor <: MLJBase.Deterministic
    estimators::Any = nothing
    weights::Any    = nothing
    n_jobs::Union{Nothing,Int} = nothing
end
