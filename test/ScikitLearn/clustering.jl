X, _ = make_blobs(500, 3, rng=555)

@testset "AffinityPropagation" begin
    m = AffinityPropagation()
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    if fp.labels !== nothing
        p  = predict(mach, X)
        @test p isa CategoricalArray
    end
    @test keys(fp) == (:cluster_centers_indices, :cluster_centers, :labels, :affinity_matrix)
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test !isempty(infos[:docstring])
    @test infos[:target_scitype] == AbstractVector{Multiclass}
end

@testset "AgglomerativeClustering" begin
    m = AgglomerativeClustering()
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    @test keys(fp) == (:n_clusters, :labels, :n_leaves, :n_connected_components, :children)
    @test fp.labels isa CategoricalArray
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test !isempty(infos[:docstring])
end

@testset "Birch" begin
    m = Birch()
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    p = predict(mach, X)
    Xt = transform(mach, X)
    @test p isa CategoricalArray
    @test Xt isa Tables.MatrixTable
    @test keys(fp) == (:root, :dummy_leaf, :subcluster_centers, :subcluster_labels, :labels)
    @test fp.labels isa CategoricalArray
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test !isempty(infos[:docstring])
    @test infos[:output_scitype] == Table(Continuous)
    @test infos[:target_scitype] == AbstractVector{Multiclass}
end

@testset "DBSCAN" begin
    m = DBSCAN()
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    @test keys(fp) == (:core_sample_indices,  :components, :labels)
    @test fp.labels isa CategoricalArray
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test !isempty(infos[:docstring])
end

@testset "FeatureAgglomeration" begin
    m = FeatureAgglomeration()
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    @test keys(fp) == (:n_clusters, :labels, :n_leaves, :n_connected_components,  :children, :distances)
    @test fp.distances === nothing
    @test fp.labels isa CategoricalArray
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test infos[:output_scitype] == Table(Continuous)
    @test !isempty(infos[:docstring])

    # NOTE: they're not equal (not sure why)
    Xt  = transform(mach, X)
    Xit = inverse_transform(mach, Xt)
    @test Xit isa Tables.MatrixTable
end

@testset "KMeans" begin
    m = KMeans(n_clusters=4)
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    Xt = transform(mach, X)
    p  = predict(mach, X)
    @test size(fp.cluster_centers) == (4, 3)
    @test keys(fp) == (:cluster_centers, :labels, :inertia)
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test infos[:output_scitype] == Table(Continuous)
    @test !isempty(infos[:docstring])
    # first fix #163 on MLJBase
    @test infos[:target_scitype] == AbstractVector{Multiclass}
end

@testset "MBKMeans" begin
    m = MiniBatchKMeans(n_clusters=4)
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    Xt = transform(mach, X)
    p  = predict(mach, X)
    @test size(fp.cluster_centers) == (4, 3)
    @test keys(fp) == (:cluster_centers, :labels, :inertia)
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test infos[:output_scitype] == Table(Continuous)
    @test !isempty(infos[:docstring])
    @test infos[:target_scitype] == AbstractVector{Multiclass}
end

@testset "MeanShift" begin
    m = MeanShift()
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    p  = predict(mach, X)
    @test p isa CategoricalArray
    @test keys(fp) == (:cluster_centers, :labels)
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test !isempty(infos[:docstring])
    @test infos[:target_scitype] == AbstractVector{Multiclass}
end

@testset "OPTICS" begin
    m = OPTICS()
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    @test keys(fp) == (:labels, :reachability, :ordering, :core_distances, :predecessor, :cluster_hierarchy)
    @test fp.labels isa CategoricalArray
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test !isempty(infos[:docstring])
end

@testset "SpectralClustering" begin
    m = SpectralClustering()
    mach = machine(m, X)
    fit!(mach)
    fp = fitted_params(mach)
    @test keys(fp) == (:labels, :affinity_matrix)
    @test fp.labels isa CategoricalArray
    infos = info_dict(m)
    @test infos[:input_scitype] == Table(Continuous)
    @test !isempty(infos[:docstring])
end
