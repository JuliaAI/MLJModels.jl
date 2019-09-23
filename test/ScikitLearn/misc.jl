Xc1, yc1 = gen_classif(classes=["M", "F"])
Xc2, yc2 = gen_classif(classes=["A", "B", "C"])
Xr, yr   = gen_reg()

@testset "DummyReg" begin
    m, f = simple_test_reg(DummyRegressor(), Xr, yr)
    fp = fitted_params(m, f)
    @test keys(fp) == (:constant, :n_outputs)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{MLJBase.Continuous}
end

@testset "DummyClf" begin
    m, f = simple_test_classif_prob(DummyClassifier(), Xc2, yc2; nodummy=true)
    fp = fitted_params(m, f)
    @test keys(fp) == (:classes, :n_classes, :n_outputs)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "GaussianNBClf" begin
    m, f = simple_test_classif_prob(GaussianNBClassifier(), Xc2, yc2)
    fp =  fitted_params(m, f)
    @test keys(fp) == (:class_prior, :class_count, :theta, :sigma, :epsilon)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "KNNReg" begin
    m, f = simple_test_reg(KNeighborsRegressor(), Xr, yr)
    fp = fitted_params(m, f)
    @test keys(fp) == (:effective_metric, :effective_metric_params)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{MLJBase.Continuous}
    @test !isempty(infos[:docstring])
end

@testset "KNNClf" begin
    m, f = simple_test_classif_prob(KNeighborsClassifier(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:classes, :effective_metric, :effective_metric_params, :outputs_2d)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end


X3 = 5 * abs.(MLJBase.matrix(Xc2))
X3 .= ceil.(X3)
Xc3 = MLJBase.table(Int.(X3))

@testset "BernNBClf" begin
    m, f = simple_test_classif_prob(BernoulliNBClassifier(binarize=2.0), Xc3, yc2; nodummy=true)
    fp =  fitted_params(m, f)
    @test keys(fp) == (:class_log_prior, :feature_log_prob, :class_count, :feature_count)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Count)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "MultiNBClf" begin
    m, f = simple_test_classif_prob(MultinomialNBClassifier(), Xc3, yc2; nodummy=true)
    fp =  fitted_params(m, f)
    @test keys(fp) == (:class_log_prior, :intercept, :feature_log_prob, :coef, :class_count, :feature_count)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Count)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "ComplNBClf" begin
    m, f = simple_test_classif_prob(ComplementNBClassifier(), Xc3, yc2; nodummy=true)
    fp =  fitted_params(m, f)
    @test keys(fp) == (:class_log_prior, :feature_log_prob, :class_count, :feature_count, :feature_all)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Count)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end
