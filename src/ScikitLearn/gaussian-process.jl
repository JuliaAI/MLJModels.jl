# | model                  | build  | fitted_params | report | metadata | tests 1 | tests 2 |
# | ---------------------- | ------ | ------------- | ------ | -------- | ------- | ------- |
# | GPRegressor            | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | GPClassif              | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |

GaussianProcessRegressor_ = SKGP.GaussianProcessRegressor
@sk_model mutable struct GaussianProcessRegressor <: MLJBase.Deterministic
    kernel::Any        = nothing
    alpha::Union{Float64,AbstractArray} = 1.0e-10
    optimizer::Any     = "fmin_l_bfgs_b"
    n_restarts_optimizer::Int = 0
    normalize_y::Bool  = false
    copy_X_train::Bool = true
    random_state::Any  = nothing
end
MLJBase.fitted_params(model::GaussianProcessRegressor, (fitresult, _, _)) = (
    X_train = fitresult.X_train_,
    y_train = fitresult.y_train_,
    kernel  = fitresult.kernel_,
    L       = fitresult.L_,
    alpha   = fitresult.alpha_,
    log_marginal_likelihood_value = fitresult.log_marginal_likelihood_value_
    )

MLJBase.input_scitype(::Type{<:GaussianProcessRegressor})  = MLJBase.Table(MLJBase.Continuous)
MLJBase.target_scitype(::Type{<:GaussianProcessRegressor}) = AbstractVector{MLJBase.Continuous}

# ============================================================================
GaussianProcessClassifier_ = SKGP.GaussianProcessClassifier
@sk_model mutable struct GaussianProcessClassifier <: MLJBase.Probabilistic
    kernel::Any        = nothing
    optimizer::Any     = "fmin_l_bfgs_b"
    n_restarts_optimizer::Int = 0
    copy_X_train::Bool = true
    random_state::Any  = nothing
    max_iter_predict::Int = 100::(_ > 0)
    warm_start::Bool   = false
    multi_class::String = "one_vs_rest"::(_ in ("one_vs_one", "one_vs_rest"))
end
MLJBase.fitted_params(m::GaussianProcessClassifier, (f, _, _)) = (
    kernel    = f.kernel_,
    log_marginal_likelihood_value = f.log_marginal_likelihood_value_,
    classes   = f.classes_,
    n_classes = f.n_classes_
    )
metadata_model(GaussianProcessClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Gaussian process classifier."
    )
