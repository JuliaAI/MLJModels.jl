# | model                  | build  | fitted_params | report | metadata | tests 1 | tests 2 |
# | ---------------------- | ------ | ------------- | ------ | -------- | ------- | ------- |
# | BernoulliNB            | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | GaussianNB             | ✓      | ✓             | ✗      | ✓        |  ✓     | ✓       |
# | MultinomialNB          | ✓      | ✓             | ✗      | ✓        |  ✓     | ✓       |
# | ComplementNB           | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | KNeighborsClf          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | KNeighborsReg          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | RadiusClf              | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | NearestCentroidClf     | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | MLPClassif             | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | MLPReg                 | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | DummyClassifier        | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | DummyRegressor         | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | IsotonicRegressor      | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | KernelRidgeRegressor   | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | BGMClassifier          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | GMClassifier           | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |


DummyRegressor_ = SKDU.DummyRegressor
@sk_reg mutable struct DummyRegressor <: MLJBase.Deterministic
    strategy::String = "mean"::(_ in ("mean", "median", "quantile", "constant"))
    constant::Any     = nothing
    quantile::Float64 = 0.5::(0 ≤ _ ≤ 1)
end
MLJBase.fitted_params(m::DummyRegressor, (f, _, _)) = (
    constant  = f.constant_,
    n_outputs = f.n_outputs_
    )
metadata_model(DummyRegressor,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{MLJBase.Continuous},
    weights=false,
    descr="DummyRegressor is a regressor that makes predictions using simple rules."
    )

# ----------------------------------------------------------------------------
DummyClassifier_ = SKDU.DummyClassifier
@sk_clf mutable struct DummyClassifier <: MLJBase.Probabilistic
    strategy::String = "stratified"::(_ in ("stratified", "most_frequent", "prior", "uniform", "constant"))
    constant::Any     = nothing
    random_state::Any = nothing
end
MLJBase.fitted_params(m::DummyClassifier, (f, _, _)) = (
    classes   = f.classes_,
    n_classes = f.n_classes_,
    n_outputs = f.n_outputs_
    )
metadata_model(DummyClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="DummyClassifier is a classifier that makes predictions using simple rules."
    )

# ============================================================================
GaussianNBClassifier_ = SKNB.GaussianNB
@sk_clf mutable struct GaussianNBClassifier <: MLJBase.Probabilistic
    priors::Option{AbstractVector{Float64}} = nothing::(_ === nothing || all(_ .≥ 0))
    var_smoothing::Float64                  = 1e-9::(_ > 0)
end
MLJBase.fitted_params(m::GaussianNBClassifier, (f, _, _)) = (
    class_prior = f.class_prior_,
    class_count = f.class_count_,
    theta       = f.theta_,
    sigma       = f.sigma_,
    epsilon     = f.epsilon_,
    )
metadata_model(GaussianNBClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Gaussian naive bayes model."
    )

# ============================================================================
BernoulliNBClassifier_ = SKNB.BernoulliNB
@sk_clf mutable struct BernoulliNBClassifier <: MLJBase.Probabilistic
    alpha::Float64            = 1.0::(_ ≥ 0)
    binarize::Option{Float64} = 0.0
    fit_prior::Bool           = true
    class_prior::Option{AbstractVector} = nothing::(_ === nothing || all(_ .≥ 0))
end
MLJBase.fitted_params(m::BernoulliNBClassifier, (f, _, _)) = (
    class_log_prior  = f.class_log_prior_,
    feature_log_prob = f.feature_log_prob_,
    class_count      = f.class_count_,
    feature_count    = f.feature_count_
    )
metadata_model(BernoulliNBClassifier,
    input=MLJBase.Table(MLJBase.Count),      # it expects binary but binarize takes care of that
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Binomial naive bayes classifier. It is suitable for classification with binary features; features will be binarized based on the `binarize` keyword (unless it's `nothing` in which
    case the features are assumed to be binary)."
    )

# ============================================================================
MultinomialNBClassifier_ = SKNB.MultinomialNB
@sk_clf mutable struct MultinomialNBClassifier <: MLJBase.Probabilistic
    alpha::Float64  = 1.0::(_ ≥ 0)
    fit_prior::Bool = true
    class_prior::Option{AbstractVector} = nothing::(_ === nothing || all(_ .≥ 0))
end
MLJBase.fitted_params(m::MultinomialNBClassifier, (f, _, _)) = (
    class_log_prior  = f.class_log_prior_,
    intercept        = f.intercept_,
    feature_log_prob = f.feature_log_prob_,
    coef             = f.coef_,
    class_count      = f.class_count_,
    feature_count    = f.feature_count_
    )
metadata_model(MultinomialNBClassifier,
    input=MLJBase.Table(MLJBase.Count),        # NOTE: sklearn may also accept continuous (tf-idf)
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Multinomial naive bayes classifier. It is suitable for classification with discrete features (e.g. word counts for text classification)."
    )

# ============================================================================
ComplementNBClassifier_ = SKNB.ComplementNB
@sk_clf mutable struct ComplementNBClassifier <: MLJBase.Probabilistic
    alpha::Float64  = 1.0::(_ ≥ 0)
    fit_prior::Bool = true
    class_prior::Option{AbstractVector} = nothing::(_ === nothing || all(_ .≥ 0))
    norm::Bool      = false
end
MLJBase.fitted_params(m::ComplementNBClassifier, (f, _, _)) = (
    class_log_prior  = f.class_log_prior_,
    feature_log_prob = f.feature_log_prob_,
    class_count      = f.class_count_,
    feature_count    = f.feature_count_,
    feature_all      = f.feature_all_
    )
metadata_model(ComplementNBClassifier,
    input=MLJBase.Table(MLJBase.Count),        # NOTE: sklearn may also accept continuous (tf-idf)
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Similar to Multinomial NB classifier but with more robust assumptions. Suited for imbalanced datasets."
    )

# ============================================================================
KNeighborsRegressor_ = SKNE.KNeighborsRegressor
@sk_reg mutable struct KNeighborsRegressor <: MLJBase.Deterministic
    n_neighbors::Int    = 5::(_ > 0)
    weights::Union{String,Function} = "uniform"::((_ isa Function) || _ in ("uniform", "distance"))
    algorithm::String   = "auto"::(_ in ("auto", "ball_tree", "kd_tree", "brute"))
    leaf_size::Int      = 30::(_ > 0)
    p::Int              = 2::(_ ≥ 0)
    metric::Any         = "minkowski"
    metric_params::Any  = nothing
    n_jobs::Option{Int} = nothing
end
MLJBase.fitted_params(m::KNeighborsRegressor, (f, _, _)) = (
    effective_metric        = f.effective_metric_,
    effective_metric_params = f.effective_metric_params_
    )
metadata_model(KNeighborsRegressor,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{MLJBase.Continuous},
    weights=false,
    descr="K-Nearest Neighbors regressor: predicts the response associated with a new point
    by taking an average of the response of the K-nearest points."
    )

# ----------------------------------------------------------------------------
KNeighborsClassifier_ = SKNE.KNeighborsClassifier
@sk_clf mutable struct KNeighborsClassifier <: MLJBase.Probabilistic
    n_neighbors::Int    = 5::(_ > 0)
    weights::Union{String,Function} = "uniform"::((_ isa Function) || _ in ("uniform", "distance"))
    algorithm::String   = "auto"::(_ in ("auto", "ball_tree", "kd_tree", "brute"))
    leaf_size::Int      = 30::(_ > 0)
    p::Int              = 2::(_ ≥ 0)
    metric::Any         = "minkowski"
    metric_params::Any  = nothing
    n_jobs::Option{Int} = nothing
end
MLJBase.fitted_params(m::KNeighborsClassifier, (f, _, _)) = (
    classes                 = f.classes_,
    effective_metric        = f.effective_metric_,
    effective_metric_params = f.effective_metric_params_,
    outputs_2d              = f.outputs_2d_
    )
metadata_model(KNeighborsClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="K-Nearest Neighbors classifier: predicts the class associated with a new point
    by taking a vote over the classes of the K-nearest points."
    )
