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
    @test 0.9 ≤ f[1].score(X, y) ≤ 0.999
    @test 0.2 ≤ norm(predict(m, f, X) .- y)/norm(y) ≤ 0.25
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:estimators, :estimator_weights, :estimator_errors, :feature_importances_)
end

@testset "BaggingReg" begin
    m = BaggingRegressor(random_state=0)
    f, _, _ = fit(m, 1, X, y)
    @test 0.9 ≤ f[1].score(X, y) ≤ 0.999
    @test 0.15 ≤ norm(predict(m, f, X) .- y)/norm(y) ≤ 0.25
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:estimators, :estimators_samples, :estimators_features, :oob_score, :oob_prediction)
end

@testset "XTreeReg" begin
    m = ExtraTreesRegressor(random_state=0)
    f, _, _ = fit(m, 1, X, y)
    @test 0.9 ≤ f[1].score(X, y) ≤ 0.999
    @test 0.15 ≤ norm(predict(m, f, X) .- y)/norm(y) ≤ 0.25
    fp = fitted_params(m, f)
    @test keys(fp) == (:estimators, :feature_importances, :n_features, :n_outputs, :oob_score, :oob_prediction)
end

@testset "GBReg" begin
    m = GradientBoostingRegressor(random_state=0)
    f, _, _ = fit(m, 1, X, y)
    @test 0.9 ≤ f[1].score(X, y) ≤ 0.999
    @test 0.03 ≤ norm(predict(m, f, X) .- y)/norm(y) ≤ 0.05
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:feature_importances, :train_score, :loss, :init, :estimators, :oob_improvement)
end

@testset "RFReg" begin
    m = RandomForestRegressor(random_state=0, oob_score=true)
    f, _, _ = fit(m, 1, X, y)
    @test 0.9 ≤ f[1].score(X, y) ≤ 0.999
    @test 0.15 ≤ norm(predict(m, f, X) .- y)/norm(y) ≤ 0.25
    # testing that the fitted params is proper
    fp = fitted_params(m, f)
    @test keys(fp) == (:estimators, :feature_importances, :n_features, :n_outputs, :oob_score, :oob_prediction)
end

# =================
# classifiers
# =================

Xc2, yc2 = gen_classif(classes=["A", "B", "C"])

@testset "AdaboostClf" begin
    m, f = simple_test_classif_prob(AdaBoostClassifier(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:estimators, :estimator_weights, :estimator_errors, :classes, :n_classes)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "BaggingClf" begin
    m, f = simple_test_classif_prob(BaggingClassifier(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:base_estimator, :estimators, :estimators_samples, :estimators_features, :classes, :n_classes, :oob_score, :oob_decision_function)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "GradBoostClf" begin
    m, f = simple_test_classif_prob(GradientBoostingClassifier(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:n_estimators, :feature_importances, :train_score, :loss, :init, :estimators, :oob_improvement)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "RForestClf" begin
    m, f = simple_test_classif_prob(RandomForestClassifier(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:estimators, :classes, :n_classes, :n_features, :n_outputs, :feature_importances, :oob_score, :oob_decision_function)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "XTreeClf" begin
end
