module TestScikitLearn

# using Revise
using MLJBase
using RDatasets
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
iris = dataset("datasets", "iris")
X = iris[:, 1:4]
y = iris[:, 5]

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


## enet = ElasticNet()
## enetCV = ElasticNetCV()


## ELASTIC NET

# generate some synthetic linear data:

x1 = randn(3000);
x2 = randn(3000);
x3 = randn(3000);
X = (x1=x1, x2=x2, x3=x3);
y = x1 - x2 -2x3 + 0.02*randn(3000);

train, test = partition(eachindex(y), 0.7)

# test CV version:
rgsCV = ElasticNetCV(copy_X=true, cv=5, eps=0.001,
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
rgs = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=true,
                 normalize=false, precompute=false, max_iter=1000,
                 copy_X=false, tol=0.0001, warm_start=false,
                 positive=false, selection="random")

fitresult, cache, report = MLJBase.fit(rgs, 0,
                                       selectrows(X, train),
                                       y[train]);
yhat = predict(rgs, fitresult, selectrows(X, test));
@test norm(yhat - y[test])/sqrt(length(y)) < 0.2
@test keys(report) == (:n_iter,:dual_gap)
fitted = fitted_params(rgs, fitresult)
@test abs(fitted.intercept) < 0.01
@test abs(fitted.coef[3] .+ 2) < 0.01



import ScikitLearn: @sk_import
@sk_import datasets: (make_friedman2, make_regression, load_breast_cancer, load_diabetes)
@sk_import gaussian_process.kernels: (DotProduct, WhiteKernel)


@testset "ARDRegression" begin
    reg = ScikitLearn_.ARDRegression()
    res, cache, rep = MLJBase.fit(reg, 1, [0 0; 1 1; 2 2], [0,1,2])
    @test MLJBase.predict(reg, res, [1 1]) == [1.0]
end

@testset "BayesianRidge" begin
    reg = ScikitLearn_.BayesianRidge()
    res, cache, rep = MLJBase.fit(reg, 1, [0 0; 1 1; 2 2], [0,1,2])
@test MLJBase.predict(reg, res, [1 1]) == [1.0]
end

@testset "HuberRegressor" begin
    X, y = make_regression(n_samples=200, n_features=2, noise = 4.0, coef = true, random_state=0)
    X[1:4,:] = rand(4,2).*10 .+ 10
    y[1:4] = rand(4).*10 .+ 10
    reg = ScikitLearn_.HuberRegressor()
    res, cache, rep = MLJBase.fit(reg, 1, X, y)
    res.score(X,y)
    MLJBase.predict(reg, res, X[1:1,:])
end

@testset "Lars" begin
    reg = ScikitLearn_.Lars()
    res, cache, rep = MLJBase.fit(reg, 1, [-1 1; 0 0; 1 1], [-1.1111, 0, -1.1111])
    @test norm(res.coef_ - [0,-1.1111]) < 1e-5
end

@testset "LarsCV" begin
    X, y = make_regression(n_samples=200, noise = 4.0, random_state=0)
    reg = ScikitLearn_.LarsCV(cv = 5)
    res, cache, rep = MLJBase.fit(reg, 1, X, y)
    @test abs(res.score(X, y) - 0.9996) < 1e-4
end

@testset "Lasso" begin
    reg = ScikitLearn_.Lasso(alpha = 0.1)
    res, cache, rep = MLJBase.fit(reg, 1, [0 0; 1 1; 2 2], [0, 1, 2])
    @test norm(res.coef_ - [0.85, 0.]) == 0.0
end

@testset "LassoCV" begin
    X, y = make_regression(noise = 4.0, random_state=0)
    reg = ScikitLearn_.LassoCV(cv = 5, random_state = 0)
    res, cache, rep = MLJBase.fit(reg, 1, X, y)
    @test abs(res.score(X,y) - 0.9993) < 1e-4
    @test norm(MLJBase.predict(reg, res, X[1:1, :]) - [-78.4951]) < 1e-4
end

@testset "LassoLars" begin
    reg = ScikitLearn_.LassoLars(alpha = 0.01)
    res, cache, rep = MLJBase.fit(clf, 1, [-1 1; 0 0; 1 1], [-1,0,-1])
    @test norm(res.coef_ - [0, -0.963257]) < 1e-4
end

@testset "LassoLarsCV" begin
    X, y = make_regression(noise = 4.0, random_state=0)
    reg = ScikitLearn_.LassoLarsCV(cv = 5)
    res, cache, rep = MLJBase.fit(reg, 1, X, y)
    @test abs(res.score(X,y) - 0.9992) < 1e-4
    @test norm(MLJBase.predict(reg, res, X[1:1, :]) - [-77.8723]) < 1e-4
end

@testset "LinearRegression" begin
    reg = ScikitLearn_.LinearRegression()
    X = [1 1; 1 2; 2 2; 2 3.]
    y = X[:,1] .* X[:,2] .+ 3
    res, cache, rep = MLJBase.fit(reg, 1, X, y)
    @test abs(res.score(X,y) - 0.9992) < 1e-4
    @test norm(MLJBase.predict(reg, res, X[1:1, :]) - [-77.8723]) < 1e-4
end

@testset "MultiTaskElasticNet" begin
    reg = ScikitLearn_.MultiTaskElasticNet(alpha = 0.1, l1_ratio = 0.5)
    res, cache, rep = MLJBase.fit(reg, 1, [0 0; 1 1; 2 2], [0 0; 1 1; 2 2])
    res.coef_
end

@testset "MultiTaskElasticNetCV" begin
    reg = ScikitLearn_.MultiTaskElasticNetCV(cv = 3, n_alphas = 100, l1_ratio = 0.5)
    res, cache, rep = MLJBase.fit(reg, 1, [0 0; 1 1; 2 2], [0 0; 1 1; 2 2])
end

@testset "MultiTaskLassoCV" begin
    X, y = make_regression(n_targets = 2, noise = 4, random_state=0)
    reg = ScikitLearn_.MultiTaskLassoCV(cv = 5, random_state = 0)
    res, cache, rep = MLJBase.fit(reg, 1, X, y)
    @test abs(res.score(X,y) - 0.9994) < 1e-4
    @test norm(MLJBase.predict(reg, res, X[1:1, :]) - [153.7971 94.9015]) < 1e-2
end

@testset "OrthogonalMatchingPursuit" begin
    X, y = make_regression(noise = 4, random_state=0)
    reg = ScikitLearn_.OrthogonalMatchingPursuit()
    res, cache, rep = MLJBase.fit(reg, 1, X, y)
    @test abs(res.score(X,y) - 0.9991) < 1e-4
    @test norm(MLJBase.predict(reg, res, X[1:1, :]) - [-78.3854]) < 1e-4
end

@testset "OrthogonalMatchingPursuitCV" begin
    X, y = make_regression(n_features = 100, n_informative = 10, noise = 4, random_state=0)
    reg = ScikitLearn_.OrthogonalMatchingPursuitCV(cv = 5)
    res, cache, rep = MLJBase.fit(reg, 1, X, y)
    @test abs(res.score(X,y) - 0.9991) < 1e-4
    @test res.n_nonzero_coefs_ == 10
    @test norm(MLJBase.predict(reg, res, X[1:1, :]) - [-78.3854]) < 1e-4
end

@testset "Ridge" begin
    seed!(0)
    X,y = randn(10, 5), randn(10)
    reg = ScikitLearn_.Ridge(alpha = 1.0)
    res, cache, rep = MLJBase.fit(reg, 1, X, y)
    @test norm(res.intercept_ - -0.78011) < 1e-5
end

@testset "AdaBoostRegressor" begin
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=false)
    reg = ScikitLearn_.AdaBoostRegressor(random_state=0, n_estimators=100)
    res, cache, rep = MLJBase.fit(reg, 1, X,y);
    @test norm(predict(regr, res, [0 0 0 0])[1] - 4.7972) < 1e-4
    @test norm(res.score(X, y) - 0.9771) < 1e-4
end

@testset "GaussianProcesses.jl" begin
    X,y = make_friedman2()
    gpr = ScikitLearn_.GaussianProcessRegressor(kernel = DotProduct() + WhiteKernel(), random_state = 1)
    res, cache, rep = MLJBase.fit(gpr, 1, X,y);
    res.score(X,y)
    MLJBase.predict(gpr, res, X[2:end, :])
end

@testset "RidgeClassifier" begin
    X, y = load_breast_cancer(return_X_y = true)
    reg = ScikitLearn_.RidgeClassifier()
    res, cache, rep = MLJBase.fit(reg, 1, X,y);
    @test norm(res.score(X, y) - 0.9595) < 1e-4
end

@testset "RidgeClassifierCV" begin
    reg = ScikitLearn_.RidgeClassifierCV(alphas = [1e-3, 1e-2, 1e-1, 1])
    res, cache, rep = MLJBase.fit(reg, 1, X,y);
    @test norm(res.score(X, y) - 0.9630) < 1e-4
end

@testset "RidgeCV" begin
    X, y = load_diabetes(return_X_y = true)
    reg = ScikitLearn_.RidgeCV(alphas = [1e-3, 1e-2, 1e-1, 1])
    res, cache, rep = MLJBase.fit(reg, 1, X,y);
    @test norm(res.score(X, y) - 0.9630) < 1e-4
end

info(SVMClassifier)
info(SVMNuClassifier)
info(SVMLClassifier)
info(SVMRegressor)
info(SVMNuRegressor)
info(SVMLRegressor)

info(ElasticNet)
info(ElasticNetCV)


end
true
