module TestScikitLearn

# using Revise
using MLJBase
using Test
using LinearAlgebra
import Random.seed!
seed!(1234)

import MLJModels
import ScikitLearn
using MLJModels.ScikitLearn_
using CategoricalArrays


## CLASSIFIERS
@test_logs (:warn,"kernel parameter is not valid, setting to default=\"rbf\" \n") SVMClassifier(kernel="wrong")
@test_logs (:warn,"penalty parameter is not valid, setting to default=\"l2\" \n") SVMLClassifier(penalty="wrong")
@test_logs (:warn,"loss parameter is not valid, setting to default=\"epsilon_insensitive\" \n") SVMLRegressor(loss="wrong")

plain_classifier = SVMClassifier()
nu_classifier = SVMNuClassifier()
linear_classifier = SVMLClassifier(max_iter=10000)

# test preservation of categorical levels:
task = load_iris();
X, y = X_and_y(task)
train, test = partition(eachindex(y), 0.6) # levels of y are split across split

fitresultC, cacheC, reportC = MLJBase.fit(plain_classifier, 1,
                                          selectrows(X, train), y[train]);
fitresultCnu, cacheCnu, reportCnu = MLJBase.fit(nu_classifier, 1,
                                          selectrows(X, train), y[train]);
fitresultCL, cacheCL, reportCL = MLJBase.fit(linear_classifier, 1,
                                          selectrows(X, train), y[train]);
pcpred = predict(plain_classifier, fitresultC, selectrows(X, test));
nucpred = predict(nu_classifier, fitresultCnu, selectrows(X, test));
lcpred = predict(linear_classifier, fitresultCL, selectrows(X, test));

@test Set(classes(pcpred[1])) == Set(classes(y[1]))
@test Set(classes(nucpred[1])) == Set(classes(y[1]))
@test Set(classes(lcpred[1])) == Set(classes(y[1]))

# test with linear data:
x1 = randn(3000);
x2 = randn(3000);
x3 = randn(3000);
X = (x1=x1, x2=x2, x3=x3);
y = x1 - x2 -2x3;
ycat = map(y) do η
    η > 0 ? "go" : "stop"
end |> categorical;
train, test = partition(eachindex(ycat), 0.8);
fitresultC, cacheC, reportC = MLJBase.fit(plain_classifier, 1,
                                          selectrows(X, train), ycat[train]);
fitresultCnu, cacheCnu, reportCnu = MLJBase.fit(nu_classifier, 1,
                                          selectrows(X, train), ycat[train]);
fitresultCL, cacheCL, reportCL = MLJBase.fit(linear_classifier, 1,
                                          selectrows(X, train), ycat[train]);
pcpred = predict(plain_classifier, fitresultC, selectrows(X, test));
nucpred = predict(nu_classifier, fitresultCnu, selectrows(X, test));
lcpred = predict(linear_classifier, fitresultCL, selectrows(X, test));
@test sum(pcpred .!= ycat[test])/length(ycat) < 0.05
@test sum(nucpred .!= ycat[test])/length(ycat) < 0.05
@test sum(lcpred .!= ycat[test])/length(ycat) < 0.05


## REGRESSORS

plain_regressor = SVMRegressor()
nu_regressor = SVMNuRegressor()
linear_regressor = SVMLRegressor(max_iter=10000)

# test with linear data:
fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 1,
                                          selectrows(X, train), y[train]);
fitresultRnu, cacheRnu, reportRnu = MLJBase.fit(nu_regressor, 1,
                                                selectrows(X, train), y[train]);
fitresultRL, cacheRL, reportRL = MLJBase.fit(linear_regressor, 1,
                                             selectrows(X, train), y[train]);

rpred = predict(plain_regressor, fitresultR, selectrows(X, test));
nurpred = predict(nu_regressor, fitresultRnu, selectrows(X, test));
Lrpred = predict(linear_regressor, fitresultRL, selectrows(X, test));

@test norm(rpred - y[test])/sqrt(length(y)) < 0.2
@test norm(nurpred - y[test])/sqrt(length(y)) < 0.2
@test norm(Lrpred - y[test])/sqrt(length(y)) < 0.2


## ELASTIC NET

# generate some synthetic linear data:

x1 = randn(3000);
x2 = randn(3000);
x3 = randn(3000);
X = (x1=x1, x2=x2, x3=x3);
y = x1 - x2 -2x3 + 0.02*randn(3000);

train, test = partition(eachindex(y), 0.7)

# test CV version:
rgsCV = SCElasticNetCV(copy_X=true, cv=5, eps=0.001,
       fit_intercept=true, l1_ratio=[0.5, 0.9], max_iter=1000, n_alphas=100,
       normalize=false, positive=false,
       precompute="auto", selection="cyclic",
       tol=0.0001)


fitresult, cache, report = MLJBase.fit(rgsCV, 0,
                                       selectrows(X, train),
                                       y[train]);

yhat = predict(rgsCV, fitresult, selectrows(X, test));
@test norm(yhat - y[test])/sqrt(length(y)) < 0.2

# test intercept and coefficients close to true:
fitted = fitted_params(rgsCV, fitresult)
@test abs(fitted.intercept) < 0.01
@test abs(fitted.coef[3] .+ 2) < 0.01

alpha, l1_ratio = report.alpha, report.l1_ratio

# test non-CV version:
rgs = SCElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=true,
                 normalize=false, precompute=false, max_iter=1000,
                 copy_X=false, tol=0.0001, warm_start=false,
                 positive=false, selection="random")

fitresult, cache, report = MLJBase.fit(rgs, 0,
                                       selectrows(X, train),
                                       y[train]);
yhat = predict(rgs, fitresult, selectrows(X, test));
@test norm(yhat - y[test])/sqrt(length(y)) < 0.2
@test keys(report) == (:n_iters,:placeholder)
fitted = fitted_params(rgs, fitresult)
@test abs(fitted.intercept) < 0.01
@test abs(fitted.coef[3] .+ 2) < 0.01






info(SVMClassifier)
info(SVMNuClassifier)
info(SVMLClassifier)
info(SVMRegressor)
info(SVMNuRegressor)
info(SVMLRegressor)

info(SCElasticNet)
info(SCElasticNetCV)


end
true
