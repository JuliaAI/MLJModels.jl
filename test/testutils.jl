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

function simple_test_classif(m, X, y)
    f, _, _ = fit(m, 1, X, y)
    p = predict(m, f, X)
    @test eltype(p) == eltype(y)
    @test Set(unique(p)) == Set(unique(y))
    m, f
end

function simple_test_classif_prob(m, X, y)
    f, _, _ = fit(m, 1, X, y)
    p = predict_mode(m, f, X)
    @test eltype(p) == eltype(y)
    @test Set(unique(p)) == Set(unique(y))
    p = predict(m, f, X)
    @test eltype(p) <: UnivariateFinite
    m, f
end
