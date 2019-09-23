Xc1, yc1 = gen_classif(classes=["M", "F"])
Xc2, yc2 = gen_classif(classes=["A", "B", "C"])

@testset "LogRegClf" begin
    m, f = simple_test_classif_prob(LogisticClassifier(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:classes, :coef, :intercept)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "LogRegCVClf" begin
    m, f = simple_test_classif_prob(LogisticCVClassifier(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:classes, :coef, :intercept, :Cs, :l1_ratios, :coefs_paths, :scores, :C, :l1_ratio)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "PAClf" begin
    m, f = simple_test_classif(PassiveAggressiveClassifier(), Xc1, yc1; dummybinary=true)
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "PerceptronClf" begin
    m, f = simple_test_classif(PerceptronClassifier(), Xc2, yc2; dummybinary=true)
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset "RidgeClf" begin
    m, f = simple_test_classif(RidgeClassifier(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

@testset  "RidgeCVClf" begin
    # NOTE: this is a hack to prevent getting a bunch of irrelevant sklearn warnings
    # which are due to version compat when using ridgecv (not relevant for us)
    old_stdout = stdout
    old_stderr = stderr
    _ = redirect_stdout()
    _ = redirect_stderr()
    m, f = simple_test_classif(RidgeCVClassifier(), Xc1, yc1)
    redirect_stdout(old_stdout)
    redirect_stderr(old_stderr)
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end

# NOTE: SGD classifier with few points is tricky which is why we remove the dummy test
@testset "SGDClf" begin
    m, f = simple_test_classif(SGDClassifier(), Xc2, yc2; nodummy=true)
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])

    m, f = simple_test_classif_prob(ProbabilisticSGDClassifier(), Xc2, yc2; nodummy=true)
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end
