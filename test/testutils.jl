using Random, MLJBase

function gen_reg(; n=100, p=5, seed=143)
    Random.seed!(143)
    X = randn(n, p)
    y = randn(n)
    return MLJBase.table(X), y
end

function gen_classif(; n=100, p=5, seed=145, classes=["A", "B"])
    Random.seed!(seed)
    X = randn(n, p)
    # gen [1, 2, 1, 3,..]
    M = exp.(randn(n, length(classes)))
    Mn = M ./ sum(M, dims=2)
    z = multi_rand(Mn)
    # associate labels
    y = [classes[zᵢ] for zᵢ in z]
    return MLJBase.table(X), MLJBase.categorical(y)
end

function gen_dummy_classif_binary(; n=50, p=5, seed=1566)
    # create clouds of points that are super separated
    cloud1 = randn(n, p)  .+ 5.0
    cloud2 = randn(n, p)  .- 5.0
    test1  = randn(10, p) .+ 5.0
    test2  = randn(10, p) .- 5.0
    X = MLJBase.table(vcat(cloud1, cloud2))
    Xt = MLJBase.table(vcat(test1, test2))
    y1 = fill("A", n)
    y2 = fill("B", n)
    yt1 = fill("A", 10)
    yt2 = fill("B", 10)
    y = MLJBase.categorical(vcat(y1, y2))
    yt = MLJBase.categorical(vcat(yt1, yt2))
    return X, Xt, y, yt
end

function gen_dummy_classif(; n=50, p=5, seed=1566)
    # create clouds of points that are super separated
    cloud1 = randn(n, p)  .+ 5.0
    cloud2 = randn(n, p)
    cloud3 = randn(n, p)  .- 5.0
    test1  = randn(10, p) .+ 5.0
    test2  = randn(10, p)
    test3  = randn(10, p) .- 5.0
    X = MLJBase.table(vcat(cloud1, cloud2, cloud3))
    Xt = MLJBase.table(vcat(test1, test2, test3))
    y1 = fill("A", n)
    y2 = fill("B", n)
    y3 = fill("C", n)
    yt1 = fill("A", 10)
    yt2 = fill("B", 10)
    yt3 = fill("C", 10)
    y = MLJBase.categorical(vcat(y1, y2, y3))
    yt = MLJBase.categorical(vcat(yt1, yt2, yt3))
    return X, Xt, y, yt
end

# simple function to sample multinomial
function multi_rand(Mp)
    n, c = size(Mp)
    be   = reshape(rand(length(Mp)), n, c)
    y    = zeros(Int, n)
    @inbounds for i in eachindex(y)
        rp = 1.0
        for k in 1:c-1
            if (be[i, k] < Mp[i, k] / rp)
                y[i] = k
                break
            end
            rp -= Mp[i, k]
        end
    end
    y[y .== 0] .= c
    return y
end

function simple_test_reg(m, X, y)
    f, _, _ = fit(m, 1, X, y)
    p = predict(m, f, X)
    @test norm(p .- y) / norm(y) < 1
    m, f
end

function test_dummy_classif(m; seed=5154, binary=false)
    if binary
        X, Xt, y, yt = gen_dummy_classif_binary(seed=seed)
    else
        X, Xt, y, yt = gen_dummy_classif(seed=seed)
    end
    f, _, _ = fit(m, 1, X, y)
    p = typeof(m) <: Probabilistic ? predict_mode(m, f, Xt) : predict(m, f, Xt)
    @test sum(p .== yt) / length(yt) ≥ 0.75
end

function simple_test_classif(m, X, y; dummybinary=false, nodummy=false)
    f, _, _ = fit(m, 1, X, y)
    p = predict(m, f, X)
    @test eltype(p) == eltype(y)
    @test Set(unique(p)) == Set(unique(y))
    nodummy || test_dummy_classif(m; binary=dummybinary)
    m, f
end

function simple_test_classif_prob(m, X, y; dummybinary=false, nodummy=false)
    f, _, _ = fit(m, 1, X, y)
    p = predict_mode(m, f, X)
    @test eltype(p) == eltype(y)
    @test Set(unique(p)) == Set(unique(y))
    p = predict(m, f, X)
    @test eltype(p) <: UnivariateFinite
    nodummy || test_dummy_classif(m; binary=dummybinary)
    m, f
end
