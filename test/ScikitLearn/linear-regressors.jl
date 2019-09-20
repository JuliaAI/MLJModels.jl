# Generate some simple random data just to make sure the
# models work; note that we don't check the validity of
# the models, we assume they work fine, we just check
# they get called properly and have appropriate metadata

Random.seed!(1551)
n, p = 100, 5
X  = randn(n, p)
X1 = hcat(X, ones(100))
θ  = randn(p+1)
y  = X1 * θ .+ 0.1 .* randn(n)

θ_ls = X1 \ y

@testset "ARD" begin
    m = ARDRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y),  0.032693, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :alpha, :lambda, :sigma, :scores)
end

@testset "BayesianRidge" begin
    m = BayesianRidgeRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :alpha, :lambda, :sigma, :scores)
end

@testset "ElasticNet" begin
    m = ElasticNetRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.769795, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.464447, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
end

@testset "ElasticNetCV" begin
    m = ElasticNetCVRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.99884, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0328523, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :l1_ratio, :mse_path, :alphas)
end

@testset "Huber" begin
    m = HuberRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.99884, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.032935, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :scale, :outliers)
end

@testset "Lars" begin
    m = LarsRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :alphas, :active, :coef_path)
end

@testset "LarsCV" begin
    m = LarsCVRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :alpha, :alphas, :cv_alphas, :mse_path, :coef_path)
end

@testset "Lasso" begin
    m = LassoRegressor(alpha = 0.1)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.9946343, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0709070, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
end

@testset "LassoCV" begin
    m = LassoCVRegressor(random_state=0, cv=6)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.998857, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.03272594, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :alpha, :alphas, :mse_path, :dual_gap)
end

@testset "LassoLars" begin
    m = LassoLarsRegressor(alpha = 0.01)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.994317, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.072973, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :alphas, :active, :coef_path)
end

@testset "LassoLarsCV" begin
    m = LassoLarsCVRegressor(cv=4)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :coef_path, :alpha, :alphas, :cv_alphas, :mse_path)
end

@testset "LassoLarsIC" begin
    m = LassoLarsICRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.9920703, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.08619978, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :alpha)
end

@testset "LinReg" begin
    m = LinearRegressor()
    f, _, _ = fit(m, 1, X, y)
    # test if things work
    @test isapprox(f[1].score(X, y), 0.998859, rtol=1e-4)
    @test isapprox(norm(predict(m, f, X) .- y) / norm(y), 0.032691, rtol=1e-4)
    fp = fitted_params(m, f)
    @test fp.coef      ≈ θ_ls[1:p]
    @test fp.intercept ≈ θ_ls[end]
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
end

@testset "OMP" begin
    m = OrthogonalMatchingPursuitRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.4593868, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.711741, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
end

@testset "OMPCV" begin
    m = OrthogonalMatchingPursuitCVRegressor(cv = 5)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :n_nonzero_coefs)
end

@testset "OMPCV" begin
    m = OrthogonalMatchingPursuitCVRegressor(cv = 5)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.998859, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0326918, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :n_nonzero_coefs)
end

@testset "PassAggr" begin
    m = PassiveAggressiveRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test 0.980 ≤ f[1].score(X, y) ≤ 0.999
    @test norm(predict(m, f, X) .- y)/norm(y) ≤ 0.1
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
end

@testset "RANSAC" begin
    m = RANSACRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test 0.99 ≤ f[1].score(X, y) ≤ 0.999
    @test 0.02 ≤ norm(predict(m, f, X) .- y)/norm(y) ≤ 0.05
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:estimator, :n_trials, :inlier_mask, :n_skips_no_inliers, :n_skips_invalid_data, :n_skips_invalid_model)
end

@testset "Ridge" begin
    m = RidgeRegressor(alpha = 1.0)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.998793, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0336254, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
end

@testset "RidgeCV" begin
    m = RidgeCVRegressor(cv=nothing, store_cv_values=true)
    f, _, _ = fit(m, 1, X, y)
    @test isapprox(f[1].score(X, y), 0.998858, rtol=1e-5)
    @test isapprox(norm(predict(m, f, X) .- y)/norm(y), 0.0327014, rtol=1e-5)
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :alpha, :cv_values)
    @test fp[:cv_values] isa Matrix
end

@testset "SGDReg" begin
    m = SGDRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test 0.980 ≤ f[1].score(X, y) ≤ 0.999
    @test norm(predict(m, f, X) .- y)/norm(y) ≤ 0.1
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :average_coef, :average_intercept)
end

@testset "TheilSen" begin
    m = TheilSenRegressor()
    f, _, _ = fit(m, 1, X, y)
    @test 0.980 ≤ f[1].score(X, y) ≤ 0.999
    @test norm(predict(m, f, X) .- y)/norm(y) ≤ 0.1
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :breakdown, :n_subpopulation)
end

##############
# MULTI TASK #
##############

y2 = (t1=y, t2=y)

@testset "MTLassoCV" begin
    m = MultiTaskLassoRegressor()
    f, _, _ = fit(m, 1, X, y2)
    @test f[1].coef_ isa Matrix
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
    pred = predict(m, f, X)
    @test pred isa Tables.MatrixTable
    @test MLJBase.schema(pred).names == (:t1, :t2)
end

@testset "MTLassoCV" begin
    m = MultiTaskLassoCVRegressor(cv = 5, random_state = 0)
    f, _, _ = fit(m, 1, X, y2)
    @test f[1].coef_ isa Matrix
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :alpha, :mse_path, :alphas)
    @test predict(m, f, X) isa Tables.MatrixTable
end

X = MLJBase.table([0 0; 1 1; 2 2])
y = MLJBase.table([0 0; 1 1; 2 2])

@testset "MTElNet" begin
    m = MultiTaskElasticNetRegressor(alpha = 0.1, l1_ratio = 0.5)
    f, _, _ = fit(m, 1, X, y)
    @test f[1].coef_ isa Matrix
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
    @test predict(m, f, X) isa Tables.MatrixTable
end

@testset "MTElNetCV" begin
    m = MultiTaskElasticNetCVRegressor(cv = 3, n_alphas = 100, l1_ratio = 0.5)
    f, _, _ = fit(m, 1, X, y)
    @test f[1].coef_ isa Matrix
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :alpha, :mse_path, :l1_ratio)
    @test predict(m, f, X) isa Tables.MatrixTable
end
