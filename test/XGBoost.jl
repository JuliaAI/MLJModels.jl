module TestXGBoost

using MLJBase
using Test
import MLJModels
import XGBoost
using CategoricalArrays
using MLJModels.XGBoost_
using Random: seed!

@test_logs (:warn, "Only \"linear\", \"gamma\" and \"tweedie\" objectives are supported . Setting objective=\"linear\". ") XGBoostRegressor(objective="wrong")
@test_logs (:warn, "Changing objective to \"automatic\", the only supported value. ") XGBoostClassifier(objective="wrong")
@test_logs (:warn, "Changing objective to \"poisson\", the only supported value. ") XGBoostCount(objective="wrong")


## REGRESSOR

seed!(0)
plain_regressor = XGBoostRegressor()
n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = features * weights;
features = MLJBase.table(features)
fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 0, features, labels);
rpred = predict(plain_regressor, fitresultR, features);
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
info(XGBoostCount)


## CLASSIFIER

plain_classifier = XGBoostClassifier(num_round=100, seed=0)

# test binary case:
N=2
seed!(0)
X = (x1=rand(1000), x2=rand(1000), x3=rand(1000))
ycat = map(X.x1) do x
    string(mod(round(Int, 10*x), N))
end |> categorical
y = identity.(ycat) # make plain Vector with categ. elements
train, test = partition(eachindex(y), 0.6) 
fitresult, cache, report = MLJBase.fit(plain_classifier, 0,
                                            selectrows(X, train), y[train];)
yhat = mode.(predict(plain_classifier, fitresult, selectrows(X, test)))
misclassification_rate = sum(yhat .!= y[test])/length(test)
@test misclassification_rate < 0.01

# Multiclass{10} case:
N=10
seed!(0)
X = (x1=rand(1000), x2=rand(1000), x3=rand(1000))
ycat = map(X.x1) do x
    string(mod(round(Int, 10*x), N))
end |> categorical
y = identity.(ycat) # make plain Vector with categ. elements

train, test = partition(eachindex(y), 0.6) 
fitresult, cache, report = MLJBase.fit(plain_classifier, 0,
                                            selectrows(X, train), y[train];)
yhat = mode.(predict(plain_classifier, fitresult, selectrows(X, test)))
misclassification_rate = sum(yhat .!= y[test])/length(test)
@test misclassification_rate < 0.01

# check target pool preserved:
X = (x1=rand(400), x2=rand(400), x3=rand(400))
ycat = vcat(fill(:x, 100), fill(:y, 100), fill(:z, 200)) |>categorical
y = identity.(ycat)
train, test = partition(eachindex(y), 0.5) 
@test length(unique(y[train])) == 2
@test length(unique(y[test])) == 1
fitresult, cache, report = MLJBase.fit(plain_classifier, 0,
                                            selectrows(X, train), y[train];)
yhat = predict_mode(plain_classifier, fitresult, selectrows(X, test))
@test Set(MLJBase.classes(yhat[1])) == Set(MLJBase.classes(y[train][1]))

info(XGBoostClassifier)

end
true
