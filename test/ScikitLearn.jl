module TestScikitLearn

using MLJBase
using Test
using LinearAlgebra
import Random.seed!
seed!(1234)

import MLJModels
import ScikitLearn
using MLJModels.ScikitLearn_
using CategoricalArrays
using RDatasets

###############
# SVM MODELS ## NOTE: these do not use the @sk_model macro
###############

## METADATA

plain_classifier  = SVMClassifier()
nu_classifier     = SVMNuClassifier()
linear_classifier = SVMLClassifier(max_iter=10000)

plain_regressor  = SVMRegressor()
nu_regressor     = SVMNuRegressor()
linear_regressor = SVMLRegressor(max_iter=10_000)

i = info(plain_classifier)

@test i[:name]          == "SVMClassifier"
@test i[:package_name]  == "ScikitLearn"
@test i[:package_url]   == "https://github.com/cstjean/ScikitLearn.jl"
@test i[:load_path]     == "MLJModels.ScikitLearn_.SVMClassifier"
@test i[:is_wrapper]    == false
@test i[:is_pure_julia] == false
@test i[:is_supervised] == true

for m in (nu_classifier, linear_classifier, plain_regressor, nu_regressor,
          linear_regressor)
    info(m)
end

## CLASSIFIERS
@test_logs (:warn,"kernel parameter is not valid, setting to default=\"rbf\" \n") SVMClassifier(kernel="wrong")
@test_logs (:warn,"penalty parameter is not valid, setting to default=\"l2\" \n") SVMLClassifier(penalty="wrong")
@test_logs (:warn,"loss parameter is not valid, setting to default=\"epsilon_insensitive\" \n") SVMLRegressor(loss="wrong")

# test preservation of categorical levels:

iris = dataset("datasets", "iris")

X = iris[:, 1:4]
y = iris[:, 5]

train, test = partition(eachindex(y), 0.6) # levels of y are split across split

fitresultC, cacheC, reportC = fit(plain_classifier, 1,
                                          selectrows(X, train), y[train])
fitresultCnu, cacheCnu, reportCnu = fit(nu_classifier, 1,
                                          selectrows(X, train), y[train])
fitresultCL, cacheCL, reportCL = fit(linear_classifier, 1,
                                          selectrows(X, train), y[train])
pcpred  = predict(plain_classifier, fitresultC, selectrows(X, test))
nucpred = predict(nu_classifier, fitresultCnu, selectrows(X, test))
lcpred  = predict(linear_classifier, fitresultCL, selectrows(X, test))

@test Set(classes(pcpred[1])) == Set(classes(y[1]))
@test Set(classes(nucpred[1])) == Set(classes(y[1]))
@test Set(classes(lcpred[1])) == Set(classes(y[1]))

# test with linear data:
x1 = randn(3_000)
x2 = randn(3_000)
x3 = randn(3_000)
X  = (x1=x1, x2=x2, x3=x3)
y  = x1 - x2 -2x3

ycat = map(y) do η
    η > 0 ? "go" : "stop"
end |> categorical

@testset "SV classifiers" begin
    train, test = partition(eachindex(ycat), 0.8)
    fitresultC, cacheC, reportC = fit(plain_classifier, 1,
                                              selectrows(X, train), ycat[train])
    fitresultCnu, cacheCnu, reportCnu = fit(nu_classifier, 1,
                                              selectrows(X, train), ycat[train])
    fitresultCL, cacheCL, reportCL = fit(linear_classifier, 1,
                                              selectrows(X, train), ycat[train])
    pcpred  = predict(plain_classifier, fitresultC, selectrows(X, test))
    nucpred = predict(nu_classifier, fitresultCnu, selectrows(X, test))
    lcpred  = predict(linear_classifier, fitresultCL, selectrows(X, test))

    @test sum(pcpred  .!= ycat[test])/length(ycat) < 0.05
    @test sum(nucpred .!= ycat[test])/length(ycat) < 0.05
    @test sum(lcpred  .!= ycat[test])/length(ycat) < 0.05
end

## REGRESSORS

# test with linear data:
fitresultR, cacheR, reportR = fit(plain_regressor, 1,
                                          selectrows(X, train), y[train])
fitresultRnu, cacheRnu, reportRnu = fit(nu_regressor, 1,
                                                selectrows(X, train), y[train])
fitresultRL, cacheRL, reportRL = fit(linear_regressor, 1,
                                             selectrows(X, train), y[train])

@testset "SV regressors" begin
    rpred   = predict(plain_regressor, fitresultR, selectrows(X, test))
    nurpred = predict(nu_regressor, fitresultRnu, selectrows(X, test))
    Lrpred  = predict(linear_regressor, fitresultRL, selectrows(X, test))

    @test norm(rpred   - y[test])/sqrt(length(y)) < 0.2
    @test norm(nurpred - y[test])/sqrt(length(y)) < 0.2
    @test norm(Lrpred  - y[test])/sqrt(length(y)) < 0.2
end


##################
# LINEAR MODELS ## NOTE: these do use the @sk_model macro
##################

# Generate some simple random data just to make sure the
# models work; note that we don't check the validity of
# the models, we assume they work fine, we just check
# they get called properly and have appropriate metadata

using Random
Random.seed!(1551)
n, p = 100, 5
X  = randn(n, p)
X1 = hcat(X, ones(100))
θ  = randn(p+1)
y  = X1 * θ .+ 0.1 .* randn(n)

θ_ls = X1 \ y

@testset "LinearRegression" begin
    m = LinearRegression()
    f, _, _ = fit(m, 1, X, y)
    # test if things work
    @test isapprox(f.score(X, y), 0.998859, rtol=1e-4)
    @test isapprox(norm(predict(m, f, X) .- y) / norm(y), 0.032691, rtol=1e-4)
    fp = fitted_params(m, f)
    @test fp.coef      ≈ θ_ls[1:p]
    @test fp.intercept ≈ θ_ls[end]
end

@testset "ARDRegression" begin
    m = ARDRegression()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y),  0.032693, rtol=1e-5)
end

@testset "BayesianRidge" begin
    m = BayesianRidge()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
end

@testset "ElasticNet" begin
    m = ElasticNet()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.769795, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.464447, rtol=1e-5)
end

@testset "ElasticNetCV" begin
    m = ElasticNetCV()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.99884, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0328523, rtol=1e-5)
end

@testset "HuberRegressor" begin
    m = HuberRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.99884, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.032935, rtol=1e-5)
end

@testset "Lars" begin
    m = Lars()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
end

@testset "LarsCV" begin
    m = LarsCV()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
end

@testset "Lasso" begin
    m = Lasso(alpha = 0.1)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.9946343, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0709070, rtol=1e-5)
end

@testset "LassoCV" begin
    m = LassoCV(random_state=0, cv=6)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.998857, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.03272594, rtol=1e-5)
end

@testset "LassoLars" begin
    m = LassoLars(alpha = 0.01)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.994317, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.072973, rtol=1e-5)
end

@testset "LassoLarsCV" begin
    m = LassoLarsCV(cv=4)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
end

# @testset "GraphicalLassoCV" begin
#     m = GraphicalLassoCV(cv=4)
#     f, _, _ = fit(m, 1, X, y)
#     @test isapprox(f.score(X, y), 0.998859, rtol=1e-5)
#     @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
# end

@testset "OrthogonalMatchingPursuit" begin
    m = OrthogonalMatchingPursuit()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.4593868, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.711741, rtol=1e-5)
end

@testset "OrthogonalMatchingPursuitCV" begin
    m = OrthogonalMatchingPursuitCV(cv = 5)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
end

@testset "Ridge" begin
    m = Ridge(alpha = 1.0)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f.score(X, y), 0.998793, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0336254, rtol=1e-5)
end

# @testset "RidgeCV" begin
#     m = RidgeCV()
#     f, _, _ = fit(m, 1, X, y)
#     @test isapprox(f.score(X, y), 0.998793, rtol=1e-5)
#     @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0336254, rtol=1e-5)
# end

@testset "MultiTaskElasticNet" begin
    m = MultiTaskElasticNet(alpha = 0.1, l1_ratio = 0.5)
    f, _, _ = fit(m, 1, [0 0; 1 1; 2 2], [0 0; 1 1; 2 2])
    @test f.coef_ isa Matrix
end

@testset "MultiTaskElasticNetCV" begin
    m = MultiTaskElasticNetCV(cv = 3, n_alphas = 100, l1_ratio = 0.5)
    f, _, _ = fit(m, 1, [0 0; 1 1; 2 2], [0 0; 1 1; 2 2])
    @test f.coef_ isa Matrix
end

y2 = hcat(y, y)

@testset "MultiTaskLassoCV" begin
    m = MultiTaskLassoCV(cv = 5, random_state = 0)
    f, _, _ = fit(m, 1, X, y2)
    @test f.coef_ isa Matrix
end

###########
# GP ##
#######

@testset "GaussianProcesses" begin
    gpr = GaussianProcessRegressor(random_state = 1)
    res, _, _ = fit(gpr, 1, X, y)
    @test res.score(X,y) ≈ 1.0
    @test norm(predict(gpr, res, X) .- y) / norm(y) ≤ 1e-10 # overfitting to the max
end

#######################
# Ridge Classifiers
######################

iris = dataset("datasets", "iris")
X = iris[:, 1:4]
y = iris[:, 5]

yplain = ones(length(y))
yplain[y .== "setosa"] .= 2
yplain[y .== "virginica"] .= 3

# @testset "RidgeClassifier" begin
#     m = RidgeClassifier()
#     f, _, _ = fit(reg, 1, X, yplain)
#     @test norm(res.score(X, y) - 0.9595) < 1e-4
# end

# @testset "RidgeClassifierCV" begin
#     m = RidgeClassifierCV(alphas = [1e-3, 1e-2, 1e-1, 1])
#     f, _, _ = fit(reg, 1, X, yplain)
#     @test norm(res.score(X, y) - 0.9630) < 1e-4
# end

###############
## ENSEMBLES ##
###############

# @testset "AdaBoostRegressor" begin
#     m = AdaBoostRegressor(random_state=0, n_estimators=100)
#     f, _, _ = fit(reg, 1, X, y)
#     @test norm(predict(regr, res, [0 0 0 0])[1] - 4.7972) < 1e-4
#     @test norm(res.score(X, y) - 0.9771) < 1e-4
# end

end
true
