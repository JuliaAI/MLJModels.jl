X, _ = make_blobs(100, 3, rng=555)

@testset "KMeans" begin
    m = KMeans(n_clusters=4)
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    @test size(fp.cluster_centers) == (4, 3)
    @test keys(fp) == (:cluster_centers, :labels, :inertia)
    infos = info_dict(m)
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test infos[:output_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test !isempty(infos[:docstring])
end
