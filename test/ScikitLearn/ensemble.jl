Random.seed!(1351)
n, p = 100, 5
X  = randn(n, p)
X1 = hcat(X, ones(100))
θ  = randn(p+1)
y  = X1 * θ .+ 0.1 .* randn(n)

θ_ls = X1 \ y

@testset "AdaBoostReg" begin
    m = AdaBoostRegressor(random_state=0, n_estimators=100)
    f, _, _ = fit(m, 1, X, y)
    @test 0.9 ≤ f.score(X, y) ≤ 0.999
    @test 0.2 ≤ norm(predict(m, f, X) .- y)/norm(y) ≤ 0.25
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:estimators, :estimator_weights, :estimator_errors, :feature_importances_)
end

@testset "BaggingReg" begin
    m = BaggingRegressor(random_state=0)
    f, _, _ = fit(m, 1, X, y)
    @test 0.9 ≤ f.score(X, y) ≤ 0.999
    @test 0.15 ≤ norm(predict(m, f, X) .- y)/norm(y) ≤ 0.25
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:estimators, :estimators_samples, :estimators_features, :oob_score, :oob_prediction)
end

# @testset "XTreeReg" begin
#     m = ExtraTreeRegressor(random_state=0)
#     f, _, _ = fit(m, 1, X, y)
#     @test isapprox(f.score(X, y), 0.9356045, rtol=1e-5)
#     @test isapprox(norm(predict(m, f, X) .- y)/norm(y),  0.2352736, rtol=1e-5)
# end

@testset "GBReg" begin
    m = GradientBoostingRegressor(random_state=0)
    f, _, _ = fit(m, 1, X, y)
    @test 0.9 ≤ f.score(X, y) ≤ 0.999
    @test 0.03 ≤ norm(predict(m, f, X) .- y)/norm(y) ≤ 0.05
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:feature_importances, :train_score, :loss, :init, :estimators)
end

@testset "RFReg" begin
    m = RandomForestRegressor(random_state=0, oob_score=true)
    f, _, _ = fit(m, 1, X, y)
    @test 0.9 ≤ f.score(X, y) ≤ 0.999
    @test 0.15 ≤ norm(predict(m, f, X) .- y)/norm(y) ≤ 0.25
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:estimators, :feature_importances, :n_features, :n_outputs, :oob_score, :oob_prediction)
end
