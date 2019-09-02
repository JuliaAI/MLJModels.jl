# | model                  | build  | fitted_params | report | metadata | tests 1 | tests 2 |
# | ---------------------- | ------ | ------------- | ------ | -------- | ------- | ------- |
# | Logistic               | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | LogisticCV             | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | Perceptron             | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | Ridge                  | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | RidgeCV                | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | SGDClassifier          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |

export # LogisticClassifier, LogisticCVClassifier,
       # PassiveAgressiveClassifier, PerceptronClassifier,
       RidgeClassifier, RidgeCVClassifier,
       # SGDClassifier

RidgeClassifier_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RidgeClassifier
@sk_model mutable struct RidgeClassifier <: MLJBase.Deterministic
    alpha::Float64      = 1.0
    fit_intercept::Bool = true
    normalize::Bool     = false
    copy_X::Bool        = true
    max_iter::Int       = 300::(arg>0)
    tol::Float64        = 1e-6::(arg>0)
    class_weight::Union{Any, Any} = nothing
    solver::String      = "auto"::(arg in ("auto","svg","cholesky","lsqr","sparse_cg","sag","saga"))
    random_state::Any   = nothing
end

RidgeClassifierCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RidgeClassifierCV
@sk_model mutable struct RidgeCVClassifier <: MLJBase.Deterministic
    alphas::Any           = nothing::(isnothing(arg) || all(0 .≤ arg .≤ 1))
    fit_intercept::Bool   = true
    normalize::Bool       = false
    scoring::Union{Nothing,String} = nothing
    cv::Int               = 5
    class_weight::Any     = nothing
    store_cv_values::Bool = false
end
