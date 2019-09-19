Xc1, yc1 = gen_classif(classes=["M", "F"])
Xc2, yc2 = gen_classif(classes=["A", "B", "C"])

@testset "DummyClf" begin
    m, f = simple_test_classif_prob(DummyClassifier(), Xc2, yc2)
    fp = fitted_params(m, f)
    @test keys(fp) == (:classes, :n_classes, :n_outputs)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test !isempty(infos[:docstring])
end
