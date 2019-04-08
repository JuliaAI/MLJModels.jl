module TestXGBoost

using MLJBase
using Test
import MLJModels
import XGBoost
using CategoricalArrays
using MLJModels.XGBoost_

@test_logs (:warn, "Only \"linear\", \"gamma\" and \"tweedie\" objectives are supported . Setting objective=\"linear\". ") XGBoostRegressor(objective="wrong")
@test_logs (:warn, "Changing objective to \"automatic\", the only supported value. ") XGBoostClassifier(objective="wrong")
@test_logs (:warn, "Changing objective to \"poisson\", the only supported value. ") XGBoostCount(objective="wrong")


## REGRESSOR

using Random: seed!
seed!(0)
plain_regressor = XGBoostRegressor()
n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = features * weights;
features = MLJBase.table(features)
fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 0, features, labels);
rpred = predict(plain_regressor, fitresultR, features);
@test fitresultR isa MLJBase.fitresult_type(plain_regressor)
info(XGBoostRegressor)

plain_regressor.objective = "gamma"
labels = abs.(labels)
fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 0, features, labels);
rpred = predict(plain_regressor, fitresultR, features);


## COUNT

count_regressor = XGBoostCount(num_round=10)
using Random: seed!
using Distributions

seed!(0)

X = randn(100, 3) .* randn(3)'
Xtable = table(X)

α = 0.1
β = [-0.3, 0.2, -0.1]
λ = exp.(α .+ X * β)
y = [rand(Poisson(λᵢ)) for λᵢ ∈ λ]

fitresultC, cacheC, reportC = MLJBase.fit(count_regressor, 0, Xtable, y);
cpred = predict(count_regressor, fitresultC, Xtable);
@test fitresultC isa MLJBase.fitresult_type(count_regressor)
info(XGBoostCount)


plain_classifier = XGBoostClassifier(num_round=100, seed=0)

# test binary case:
N=2
seed!(0)
X = (x1=rand(1000), x2=rand(1000), x3=rand(1000))
y = map(X.x1) do x
    mod(round(Int, 10*x), N)
end |> categorical
train, test = partition(eachindex(y), 0.6) 
fitresult, cache, report = MLJBase.fit(plain_classifier, 0,
                                            selectrows(X, train), y[train];)
yhat = mode.(predict(plain_classifier, fitresult, selectrows(X, test)))
misclassification_rate = sum(yhat .!= y[test])/length(test)
@test misclassification_rate < 0.1

# Multiclass{10} case:
N=10
seed!(0)
X = (x1=rand(1000), x2=rand(1000), x3=rand(1000))
y = map(X.x1) do x
    mod(round(Int, 10*x), N)
end |> categorical
train, test = partition(eachindex(y), 0.6) 
fitresult, cache, report = MLJBase.fit(plain_classifier, 0,
                                            selectrows(X, train), y[train];)
yhat = mode.(predict(plain_classifier, fitresult, selectrows(X, test)))
misclassification_rate = sum(yhat .!= y[test])/length(test)
@test misclassification_rate < 0.1


# check target pool preserved:
X = (x1=rand(400), x2=rand(400), x3=rand(400))
y = vcat(fill(:x, 100), fill(:y, 100), fill(:z, 200)) |>categorical
train, test = partition(eachindex(y), 0.5) 
@assert length(unique(y[train])) == 2
@assert length(unique(y[test])) == 1
fitresult, cache, report = MLJBase.fit(plain_classifier, 0,
                                            selectrows(X, train), y[train];)
yhat = predict(plain_classifier, fitresult, selectrows(X, test))
@test Set(levels(yhat[1])) == Set(levels(y[train]))

@test fitresult isa MLJBase.fitresult_type(plain_classifier)
info(XGBoostClassifier)

end
true
