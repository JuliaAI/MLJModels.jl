# | model                  | build  | fitted_params | report | metadata | tests 1 | tests 2 |
# | ---------------------- | ------ | ------------- | ------ | -------- | ------- | ------- |
# | DummyClassifier        | ✓      | ✓             | ✗      | ✓        |  ✓      | ✓       |
# | DummyRegressor         | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | IsotonicRegressor      | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | KernelRidgeRegressor   | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | BGMClassifier          | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |
# | GMClassifier           | ✗      | ✗             | ✗      | ✗        |  ✗      | ✗       |

DummyClassifier_ = ((ScikitLearn.Skcore).pyimport("sklearn.dummy")).DummyClassifier
@sk_model mutable struct DummyClassifier <: MLJBase.Probabilistic
    strategy::String = "stratified"::(_ in ("stratified", "most_frequent", "prior", "uniform", "constant"))
    constant::Any     = nothing
    random_state::Any = nothing
end
MLJBase.fitted_params(model::DummyClassifier, (fitresult, _, _)) = (
    classes   = fitresult.classes_,
    n_classes = fitresult.n_classes_,
    n_outputs = fitresult.n_outputs_
    )
metadata_model(DummyClassifier,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="DummyClassifier is a classifier that makes predictions using simple rules."
    )

#DummyRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.dummy")).DummyClassifier
