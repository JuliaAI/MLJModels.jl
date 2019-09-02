# | model                  | build  | fitted_params | report | metadata | tests 1 | tests 2 |
# | ---------------------- | ------ | ------------- | ------ | -------- | ------- | ------- |
# | GPRegressor            | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | GPClassif              | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |

export GaussianProcessRegressor #, GaussianProcesseClassifier

GaussianProcessRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.gaussian_process")).GaussianProcessRegressor
@sk_model mutable struct GaussianProcessRegressor <: MLJBase.Deterministic
    kernel::Any        = nothing
    alpha::Union{Float64, Any}    = 1.0e-10
    optimizer::Union{String, Any} = "fmin_l_bfgs_b"
    n_restarts_optimizer::Int = 0
    normalize_y::Bool  = false
    copy_X_train::Bool = true
    random_state::Any  = nothing
end
