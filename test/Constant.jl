module TestConstant

using Test, MLJBase, MLJModels
using Distributions

## REGRESSOR

# Any X will do for constant models:
X = NamedTuple{(:x1,:x2,:x3)}((rand(10), rand(10), rand(10)))
y = [1.0, 1.0, 2.0, 2.0]

model = ConstantRegressor(distribution_type=
                          Distributions.Normal{Float64})
fitresult, cache, report = MLJBase.fit(model, 1, X, y)

d=Distributions.Normal(1.5, 0.5)
@test fitresult.μ ≈ d.μ
@test fitresult.σ ≈ d.σ
@test MLJBase.predict(model, fitresult, X)[7].μ ≈ d.μ
@test MLJBase.predict_mean(model, fitresult, X) ≈ fill(1.5, 10)

MLJBase.info_dict(model)
MLJBase.info_dict(DeterministicConstantRegressor)


## CLASSIFIER

yraw = ["Perry", "Antonia", "Perry", "Skater"]
y = categorical(yraw)

model = ConstantClassifier()
fitresult, cache, report =  MLJBase.fit(model, 1, X, y)
d = MLJBase.UnivariateFinite([y[1], y[2], y[4]], [0.5, 0.25, 0.25])
@test all([pdf(d, c) ≈ pdf(fitresult, c) for c in MLJBase.classes(d)])

yhat = MLJBase.predict_mode(model, fitresult, X)
@test MLJBase.classes(yhat[1]) == MLJBase.classes(y[1])
@test yhat[5] == y[1]
@test length(yhat) == 10

yhat = MLJBase.predict(model, fitresult, X)
yhat1 = yhat[1]
@test all([pdf(yhat1, c) ≈ pdf(d, c) for c in MLJBase.classes(d)])

MLJBase.info_dict(model)
MLJBase.info_dict(DeterministicConstantClassifier)

end # module
true
