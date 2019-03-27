module TestXGBoost

using MLJBase
using Test
import MLJModels
import XGBoost
using CategoricalArrays
using MLJModels.XGBoost_


@test_logs (:warn,"\n objective function is more suited to XGBoostClassifier or XGBoostCount") XGBoostRegressor(objective="wrong")
@test_logs (:warn,"\n objective function is more suited to XGBoostClassifier or XGBoostRegressor") XGBoostCount(objective="wrong")
@test_logs (:warn,"\n objective function is more suited to XGBoostRegressor or XGBoostCount") XGBoostClassifier(objective="wrong")


plain_regressor = XGBoostRegressor()
n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = features * weights;
fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 1, features, labels);
rpred = predict(plain_regressor, fitresultR, features);
@test fitresultR isa MLJBase.fitresult_type(plain_regressor)
info(XGBoostRegressor)


count_regressor = XGBoostCount()
using Random: seed!
using Distributions

seed!(0)

X = randn(100, 3) .* randn(3)'
Xtable = table(X)

α = 0.1
β = [-0.3, 0.2, -0.1]
λ = exp.(α .+ X * β)
y = [rand(Poisson(λᵢ)) for λᵢ ∈ λ]

fitresultC, cacheC, reportC = MLJBase.fit(count_regressor, 1, Xtable, y);
cpred = predict(count_regressor, fitresultC, Xtable);
@test fitresultC isa MLJBase.fitresult_type(count_regressor)
info(XGBoostCount)




plain_classifier = XGBoostClassifier()
task = load_iris();
X, y = X_and_y(task)
train, test = partition(eachindex(y), 0.6) # levels of y are split across split



fitresultCl, cacheCl, reportCl = MLJBase.fit(plain_classifier, 0,
                                            selectrows(X, train), y[train];)

println(fitresultCl)
clpred = predict(plain_classifier, fitresultCl, selectrows(X, test));
@test levels(clpred) == levels(y[train])

@test fitresultCl isa MLJBase.fitresult_type(plain_regressor)
info(XGBoostClassifier)

end
true
