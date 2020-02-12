Xc2, yc2 = gen_classif(classes=["A", "B", "C"])

@testset "LDA" begin
    m, f = simple_test_classif_prob(BayesianLDA(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:coef, :intercept, :covariance, :means,
                       :priors, :scalings, :xbar, :classes,
                       :explained_variance_ratio)
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test infos[:target_scitype] == AbstractVector{<:Finite}
    @test !isempty(infos[:docstring])
end

@testset "QDA" begin
    m, f = simple_test_classif_prob(BayesianQDA(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:covariance, :means, :priors,
                       :rotations, :scalings)
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test infos[:target_scitype] == AbstractVector{<:Finite}
    @test !isempty(infos[:docstring])
end
