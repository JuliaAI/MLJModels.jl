AdaBoostRegressor_ = SKEN.AdaBoostRegressor
@sk_reg mutable struct AdaBoostRegressor <: MMI.Deterministic
    base_estimator::Any    = nothing
    n_estimators::Int      = 50::(_ > 0)
    learning_rate::Float64 = 1.0::(_ > 0)
    loss::String           = "linear"::(_ in ("linear","square","exponential"))
    random_state::Any      = nothing
end
MMI.fitted_params(model::AdaBoostRegressor, (f, _, _)) = (
    estimators           = f.estimators_,
    estimator_weights    = f.estimator_weights_,
    estimator_errors     = f.estimator_errors_,
    feature_importances_ = f.feature_importances_
    )

# ----------------------------------------------------------------------------
AdaBoostClassifier_ = SKEN.AdaBoostClassifier
@sk_clf mutable struct AdaBoostClassifier <: MMI.Probabilistic
    base_estimator::Any    = nothing
    n_estimators::Int      = 50::(_ > 0)
    learning_rate::Float64 = 1.0::(_ > 0)
    algorithm::String      = "SAMME.R"::(_ in ("SAMME", "SAMME.R"))
    random_state::Any      = nothing
end
MMI.fitted_params(m::AdaBoostClassifier, (f, _, _)) = (
    estimators        = f.estimators_,
    estimator_weights = f.estimator_weights_,
    estimator_errors  = f.estimator_errors_,
    classes           = f.classes_,
    n_classes         = f.n_classes_
    )
metadata_model(AdaBoostClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    descr   = "Adaboost ensemble classifier."
    )

# ============================================================================
BaggingRegressor_ = SKEN.BaggingRegressor
@sk_reg mutable struct BaggingRegressor <: MMI.Deterministic
    base_estimator::Any      = nothing
    n_estimators::Int        = 10::(_>0)
    max_samples::Union{Int,Float64}  = 1.0::(_>0)
    max_features::Union{Int,Float64} = 1.0::(_>0)
    bootstrap::Bool          = true
    bootstrap_features::Bool = false
    oob_score::Bool          = false
    warm_start::Bool         = false
    n_jobs::Option{Int}      = nothing
    random_state::Any        = nothing
    verbose::Int             = 0
end
MMI.fitted_params(model::BaggingRegressor, (f, _, _)) = (
    estimators          = f.estimators_,
    estimators_samples  = f.estimators_samples_,
    estimators_features = f.estimators_features_,
    oob_score           = model.oob_score ? f.oob_score_ : nothing,
    oob_prediction      = model.oob_score ? f.oob_prediction_ : nothing
    )

# ----------------------------------------------------------------------------
BaggingClassifier_ = SKEN.BaggingClassifier
@sk_clf mutable struct BaggingClassifier <: MMI.Probabilistic
    base_estimator::Any      = nothing
    n_estimators::Int        = 10::(_>0)
    max_samples::Union{Int,Float64}  = 1.0::(_>0)
    max_features::Union{Int,Float64} = 1.0::(_>0)
    bootstrap::Bool          = true
    bootstrap_features::Bool = false
    oob_score::Bool          = false
    warm_start::Bool         = false
    n_jobs::Option{Int}      = nothing
    random_state::Any        = nothing
    verbose::Int             = 0
end
MMI.fitted_params(m::BaggingClassifier, (f, _, _)) = (
    base_estimator        = f.base_estimator_,
    estimators            = f.estimators_,
    estimators_samples    = f.estimators_samples_,
    estimators_features   = f.estimators_features_,
    classes               = f.classes_,
    n_classes             = f.n_classes_,
    oob_score             = m.oob_score ? f.oob_score_ : nothing,
    oob_decision_function = m.oob_score ? f.oob_decision_function_ : nothing
    )
metadata_model(BaggingClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    descr   = "Bagging ensemble classifier."
    )

# ============================================================================
GradientBoostingRegressor_ = SKEN.GradientBoostingRegressor
@sk_reg mutable struct GradientBoostingRegressor <: MMI.Deterministic
    loss::String                    = "ls"::(_ in ("ls","lad","huber","quantile"))
    learning_rate::Float64          = 0.1::(_>0)
    n_estimators::Int               = 100::(_>0)
    subsample::Float64              = 1.0::(_>0)
    criterion::String               = "friedman_mse"::(_ in ("mse","mae","friedman_mse"))
    min_samples_split::Union{Int,Float64} = 2::(_>0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_>0)
    min_weight_fraction_leaf::Float64     = 0.0::(_≥0)
    max_depth::Int                 = 3::(_>0)
    min_impurity_decrease::Float64 = 0.0::(_≥0)
    init::Any                      = nothing
    random_state::Any              = nothing
    max_features::Union{Int,Float64,String,Nothing} = nothing::(_===nothing || (isa(_,String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    alpha::Float64                 = 0.9::(_>0)
    verbose::Int                   = 0
    max_leaf_nodes::Option{Int}    = nothing::(_===nothing || _>0)
    warm_start::Bool               = false
    presort::Union{Bool,String}    = "auto"::(isa(_, Bool) || _ == "auto")
    validation_fraction::Float64   = 0.1::(_>0)
    n_iter_no_change::Option{Int}  = nothing
    tol::Float64                   = 1e-4::(_>0)
end
MMI.fitted_params(m::GradientBoostingRegressor, (f, _, _)) = (
    feature_importances = f.feature_importances_,
    train_score         = f.train_score_,
    loss                = f.loss_,
    init                = f.init_,
    estimators          = f.estimators_,
    oob_improvement     = m.subsample < 1 ? f.oob_improvement_ : nothing
    )

# ----------------------------------------------------------------------------
GradientBoostingClassifier_ = SKEN.GradientBoostingClassifier
@sk_clf mutable struct GradientBoostingClassifier <: MMI.Probabilistic
    loss::String                    = "deviance"::(_ in ("deviance","exponential"))
    learning_rate::Float64          = 0.1::(_>0)
    n_estimators::Int               = 100::(_>0)
    subsample::Float64              = 1.0::(_>0)
    criterion::String               = "friedman_mse"::(_ in ("mse","mae","friedman_mse"))
    min_samples_split::Union{Int,Float64} = 2::(_>0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_>0)
    min_weight_fraction_leaf::Float64     = 0.0::(_≥0)
    max_depth::Int                 = 3::(_>0)
    min_impurity_decrease::Float64 = 0.0::(_≥0)
    init::Any                      = nothing
    random_state::Any              = nothing
    max_features::Union{Int,Float64,String,Nothing} = nothing::(_===nothing || (isa(_,String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    verbose::Int                   = 0
    max_leaf_nodes::Option{Int}    = nothing::(_===nothing || _>0)
    warm_start::Bool               = false
    presort::Union{Bool,String}    = "auto"::(isa(_, Bool) || _ == "auto")
    validation_fraction::Float64   = 0.1::(_>0)
    n_iter_no_change::Option{Int}  = nothing
    tol::Float64                   = 1e-4::(_>0)
end
MMI.fitted_params(m::GradientBoostingClassifier, (f, _, _)) = (
    n_estimators        = f.n_estimators_,
    feature_importances = f.feature_importances_,
    train_score         = f.train_score_,
    loss                = f.loss_,
    init                = f.init_,
    estimators          = f.estimators_,
    oob_improvement     = m.subsample < 1 ? f.oob_improvement_ : nothing
    )
metadata_model(GradientBoostingClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    descr   = "Gradient boosting ensemble classifier."
    )

# ============================================================================
RandomForestRegressor_ = SKEN.RandomForestRegressor
@sk_reg mutable struct RandomForestRegressor <: MMI.Deterministic
    n_estimators::Int              = 100::(_ > 0)
    criterion::String              = "mse"::(_ in ("mae", "mse"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64} = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_ > 0)
    min_weight_fraction_leaf::Float64     = 0.0::(_ ≥ 0)
    max_features::Union{Int,Float64,String,Nothing} = "auto"::(_ === nothing || (isa(_, String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    max_leaf_nodes::Option{Int}    = nothing::(_ === nothing || _ > 0)
    min_impurity_decrease::Float64 = 0.0::(_ ≥ 0)
    bootstrap::Bool                = true
    oob_score::Bool                = false
    n_jobs::Option{Int}            = nothing
    random_state::Any              = nothing
    verbose::Int                   = 0
    warm_start::Bool               = false
end
MMI.fitted_params(model::RandomForestRegressor, (f, _, _)) = (
    estimators          = f.estimators_,
    feature_importances = f.feature_importances_,
    n_features          = f.n_features_,
    n_outputs           = f.n_outputs_,
    oob_score           = model.oob_score ? f.oob_score_ : nothing,
    oob_prediction      = model.oob_score ? f.oob_prediction_ : nothing
    )
metadata_model(RandomForestRegressor,
    input   = Table(Count,Continuous),
    target  = AbstractVector{Continuous},
    weights = false,
    descr   = "Random forest regressor."
    )

# ----------------------------------------------------------------------------
RandomForestClassifier_ = SKEN.RandomForestClassifier
@sk_clf mutable struct RandomForestClassifier <: MMI.Probabilistic
    n_estimators::Int              = 100::(_ > 0)
    criterion::String              = "gini"::(_ in ("gini","entropy"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64} = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}  = 1::(_ > 0)
    min_weight_fraction_leaf::Float64     = 0.0::(_ ≥ 0)
    max_features::Union{Int,Float64,String,Nothing} = "auto"::(_ === nothing || (isa(_, String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    max_leaf_nodes::Option{Int}    = nothing::(_ === nothing || _ > 0)
    min_impurity_decrease::Float64 = 0.0::(_ ≥ 0)
    bootstrap::Bool                = true
    oob_score::Bool                = false
    n_jobs::Option{Int}            = nothing
    random_state::Any              = nothing
    verbose::Int                   = 0
    warm_start::Bool               = false
    class_weight::Any              = nothing
end
MMI.fitted_params(m::RandomForestClassifier, (f, _, _)) = (
    estimators            = f.estimators_,
    classes               = f.classes_,
    n_classes             = f.n_classes_,
    n_features            = f.n_features_,
    n_outputs             = f.n_outputs_,
    feature_importances   = f.feature_importances_,
    oob_score             = m.oob_score ? f.oob_score_ : nothing,
    oob_decision_function = m.oob_score ? f.oob_decision_function_ : nothing
    )
metadata_model(RandomForestClassifier,
    input   = Table(Count,Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    descr   = "Random forest classifier."
    )

const ENSEMBLE_REG = Union{Type{<:AdaBoostRegressor}, Type{<:BaggingRegressor}, Type{<:GradientBoostingRegressor}}

MMI.input_scitype(::ENSEMBLE_REG)  = Table(Continuous)
MMI.target_scitype(::ENSEMBLE_REG) = AbstractVector{Continuous}

# ============================================================================
ExtraTreesRegressor_ = SKEN.ExtraTreesRegressor
@sk_reg mutable struct ExtraTreesRegressor <: MMI.Deterministic
    n_estimators::Int              = 100::(_>0)
    criterion::String              = "mse"::(_ in ("mae", "mse"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64}  = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}   = 1::(_ > 0)
    min_weight_fraction_leaf::Float64      = 0.0::(_ ≥ 0)
    max_features::Union{Int,Float64,String,Nothing} = "auto"::(_ === nothing || (isa(_, String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    max_leaf_nodes::Option{Int}    = nothing::(_ === nothing || _ > 0)
    min_impurity_decrease::Float64 = 0.0::(_ ≥ 0)
    bootstrap::Bool                = true
    oob_score::Bool                = false
    n_jobs::Option{Int}            = nothing
    random_state::Any              = nothing
    verbose::Int                   = 0
    warm_start::Bool               = false
end
MMI.fitted_params(m::ExtraTreesRegressor, (f, _, _)) = (
    estimators          = f.estimators_,
    feature_importances = f.feature_importances_,
    n_features          = f.n_features_,
    n_outputs           = f.n_outputs_,
    oob_score           = m.oob_score ? f.oob_score_ : nothing,
    oob_prediction      = m.oob_score ? f.oob_prediction_ : nothing,
    )
metadata_model(ExtraTreesRegressor,
    input   = Table(Continuous),
    target  = AbstractVector{Continuous},
    weights = false,
    descr   = "Extra trees regressor, fits a number of randomized decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting."
    )

# ----------------------------------------------------------------------------
ExtraTreesClassifier_ = SKEN.ExtraTreesClassifier
@sk_clf mutable struct ExtraTreesClassifier <: MMI.Probabilistic
    n_estimators::Int              = 100::(_>0)
    criterion::String              = "gini"::(_ in ("gini", "entropy"))
    max_depth::Option{Int}         = nothing::(_ === nothing || _ > 0)
    min_samples_split::Union{Int,Float64}  = 2::(_ > 0)
    min_samples_leaf::Union{Int,Float64}   = 1::(_ > 0)
    min_weight_fraction_leaf::Float64      = 0.0::(_ ≥ 0)
    max_features::Union{Int,Float64,String,Nothing} = "auto"::(_ === nothing || (isa(_, String) && (_ in ("auto","sqrt","log2"))) || _ > 0)
    max_leaf_nodes::Option{Int}    = nothing::(_ === nothing || _ > 0)
    min_impurity_decrease::Float64 = 0.0::(_ ≥ 0)
    bootstrap::Bool                = true
    oob_score::Bool                = false
    n_jobs::Option{Int}            = nothing
    random_state::Any              = nothing
    verbose::Int                   = 0
    warm_start::Bool               = false
    class_weight::Any              = nothing
end
MMI.fitted_params(m::ExtraTreesClassifier, (f, _, _)) = (
    estimators            = f.estimators_,
    classes               = f.classes_,
    n_classes             = f.n_classes_,
    feature_importances   = f.feature_importances_,
    n_features            = f.n_features_,
    n_outputs             = f.n_outputs_,
    oob_score             = m.oob_score ? f.oob_score_ : nothing,
    oob_decision_function = m.oob_score ? f.oob_decision_function_ : nothing,
    )
metadata_model(ExtraTreesClassifier,
    input   = Table(Continuous),
    target  = AbstractVector{<:Finite},
    weights = false,
    descr   = "Extra trees classifier, fits a number of randomized decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting."
    )
