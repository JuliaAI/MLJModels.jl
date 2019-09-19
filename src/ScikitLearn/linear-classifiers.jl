# | model                  | build  | fitted_params | report | metadata | tests 1 | tests 2 |
# | ---------------------- | ------ | ------------- | ------ | -------- | ------- | ------- |
# | Logistic               | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | LogisticCV             | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | Perceptron             | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | Ridge                  | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | RidgeCV                | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | SGDClassifier          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |

LogisticClassifier_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LogisticRegression
@sk_model mutable struct LogisticClassifier <: MLJBase.Probabilistic
    penalty::String            = "l2"::(_ in ("l1", "l2", "elasticnet", "none"))
    dual::Bool                 = false
    tol::Float64               = 1e-4::(_ > 0)
    C::Float64                 = 1.0::(_ > 0)
    fit_intercept::Bool        = true
    intercept_scaling::Float64 = 1.0::(_ > 0)
    class_weight::Any          = nothing
    random_state::Any          = nothing
    solver::String             = "lbfgs"::(_ in ("lbfgs", "newton-cg", "liblinear", "sag", "saga"))
    max_iter::Int              = 100::(_ > 0)
    multi_class::String        = "auto"::(_ in ("ovr", "multinomial", "auto"))
    verbose::Int               = 0
    warm_start::Bool           = false
    n_jobs::Option{Int}        = nothing
    l1_ratio::Option{Float64}  = nothing::(_ === nothing || 0 ≤ _ ≤ 1)
end
MLJBase.fitted_params(m::LogisticClassifier, (f, _, _)) = (
    classes   = f.classes_,
    coef      = f.coef_,
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
metadata_model(LogisticClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Logistic regression classifier."
    )

# ============================================================================
LogisticCVClassifier_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LogisticRegressionCV
@sk_model mutable struct LogisticCVClassifier <: MLJBase.Probabilistic
    Cs::Union{Int,AbstractVector{Float64}} = 10::((_ isa Int && _ > 0) || all(_ .> 0))
    fit_intercept::Bool        = true
    cv::Any                    = 5
    dual::Bool                 = false
    penalty::String            = "l2"::(_ in ("l1", "l2", "elasticnet", "none"))
    scoring::Any               = nothing
    solver::String             = "lbfgs"::(_ in ("lbfgs", "newton-cg", "liblinear", "sag", "saga"))
    tol::Float64               = 1e-4::(_ > 0)
    max_iter::Int              = 100::(_ > 0)
    class_weight::Any          = nothing
    n_jobs::Option{Int}        = nothing
    verbose::Int               = 0
    refit::Bool                = true
    intercept_scaling::Float64 = 1.0::(_ > 0)
    multi_class::String        = "auto"::(_ in ("ovr", "multinomial", "auto"))
    random_state::Any          = nothing
    l1_ratios::Option{AbstractVector{Float64}}=nothing::(_ === nothing || all(0 .≤ _ .≤ 1))
end
MLJBase.fitted_params(m::LogisticCVClassifier, (f, _, _)) = (
    classes     = f.classes_,
    coef        = f.coef_,
    intercept   = m.fit_intercept ? f.intercept_ : nothing,
    Cs          = f.Cs_,
    l1_ratios   = ifelse(m.penalty == "elasticnet", f.l1_ratios_, nothing),
    coefs_paths = f.coefs_paths_,
    scores      = f.scores_,
    C           = f.C_,
    l1_ratio    = f.l1_ratio_
    )
metadata_model(LogisticCVClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Logistic regression classifier with internal cross-validation."
    )

# ============================================================================
PerceptronClassifier_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).Perceptron
@sk_model mutable struct PerceptronClassifier <: MLJBase.Deterministic
    penalty::Union{Nothing,String} = nothing::(_ === nothing || _ in ("l2", "l1", "elasticnet"))
    alpha::Float64                 = 1e-4::(_ > 0)
    fit_intercept::Bool            = true
    max_iter::Int                  = 1_000::(_ > 0)
    tol::Union{Nothing,Float64}    = 1e-3
    shuffle::Bool                  = true
    verbose::Int                   = 0
    eta0::Float64                  = 1.0::(_ > 0)
    n_jobs::Union{Nothing,Int}     = nothing
    random_state::Any              = 0
    early_stopping::Bool           = false
    validation_fraction::Float64   = 0.1::(_ > 0)
    n_iter_no_change::Int          = 5::(_ > 0)
    class_weight::Any              = nothing
    warm_start::Bool               = false
end
MLJBase.fitted_params(m::PerceptronClassifier, (f, _, _)) = (
    coef      = f.coef_,
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
metadata_model(PerceptronClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Perceptron classifier."
    )

# ============================================================================
RidgeClassifier_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RidgeClassifier
@sk_model mutable struct RidgeClassifier <: MLJBase.Deterministic
    alpha::Float64    = 1.0
    fit_intercept::Bool = true
    normalize::Bool   = false
    copy_X::Bool      = true
    max_iter::Option{Int} = nothing::(_ === nothing || _ > 0)
    tol::Float64      = 1e-3::(arg>0)
    class_weight::Any = nothing
    solver::String    = "auto"::(arg in ("auto","svd","cholesky","lsqr","sparse_cg","sag","saga"))
    random_state::Any = nothing
end
MLJBase.fitted_params(m::RidgeClassifier, (f, _, _)) = (
    coef      = f.coef_,
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
metadata_model(RidgeClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Ridge regression classifier."
    )

# ============================================================================
RidgeCVClassifier_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RidgeClassifierCV
@sk_model mutable struct RidgeCVClassifier <: MLJBase.Deterministic
    alphas::AbstractArray{Float64} = [0.1,1.0,10.0]::(all(0 .≤ _))
    fit_intercept::Bool   = true
    normalize::Bool       = false
    scoring::Any          = nothing
    cv::Int               = 5
    class_weight::Any     = nothing
    store_cv_values::Bool = false
end
MLJBase.fitted_params(m::RidgeCVClassifier, (f, _, _)) = (
    coef      = f.coef_,
    intercept = ifelse(m.fit_intercept, f.intercept_, nothing)
    )
metadata_model(RidgeCVClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Ridge regression classifier."
    )
