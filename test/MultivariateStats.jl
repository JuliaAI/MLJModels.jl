module TestMultivariateStats

using Test
using MLJBase
using RDatasets
import MultivariateStats,  Random
using MLJModels.MultivariateStats_

using LinearAlgebra
import Random.seed!
seed!(1234)

@testset "Ridge" begin
    ## SYNTHETIC DATA TEST

    # Define some linear, noise-free, synthetic data:
    bias = -42.0
    coefficients = Float64[1, 3, 7]
    n = 1000
    A = randn(n, 3)
    Xtable = MLJBase.table(A)
    y = A*coefficients

    # Train model on all data with no regularization and no
    # standardization of target:
    ridge = RidgeRegressor(lambda=0.0)

    fitresult, report, cache = fit(ridge, 0, Xtable, y)

    # Training error:
    yhat = predict(ridge, fitresult, Xtable)
    @test norm(yhat - y)/sqrt(n) < 1e-12

    # Get the true bias?
    fr = fitted_params(ridge, fitresult)
    @test abs(fr.bias) < 1e-10
    @test norm(fr.coefficients - coefficients) < 1e-10

    info_dict(ridge)
end

data = dataset("MASS", "crabs")
X = MLJBase.selectcols(data, [:FL, :RW, :CL, :CW, :BD])
y = MLJBase.selectcols(data, :Sp)

@testset "PCA" begin
    X_array = MLJBase.matrix(X)
    pratio = 0.9999

    # MultivariateStats PCA
    pca_ms = MultivariateStats.fit(MultivariateStats.PCA, permutedims(X_array), pratio=pratio)
    Xtr_ms = permutedims(MultivariateStats.transform(pca_ms, permutedims(X_array)))

    # MLJ PCA
    pca_mlj = PCA(pratio=pratio)
    fitresult, _, _ = MLJBase.fit(pca_mlj, 1, X)
    Xtr_mlj = MLJBase.matrix(MLJBase.transform(pca_mlj, fitresult, X))

    @test Xtr_mlj ≈ Xtr_ms
end

@testset "KernelPCA" begin
    X_array = MLJBase.matrix(X)

    # MultivariateStats KernelPCA
    kpca_ms = MultivariateStats.fit(MultivariateStats.KernelPCA
                                  , permutedims(X_array))
    Xtr_ms = permutedims(MultivariateStats.transform(kpca_ms, permutedims(X_array)))

    # MLJ KernelPCA
    kpca_mlj = KernelPCA()
    fitresult, _, _ = MLJBase.fit(kpca_mlj, 1, X)
    Xtr_mlj = MLJBase.matrix(MLJBase.transform(kpca_mlj, fitresult, X))

    @test Xtr_mlj ≈ Xtr_ms
end

@testset "ICA" begin
    X_array = MLJBase.matrix(X)
    k = 5
    tolerance = 5.0

    # MultivariateStats ICA
    seed!(1234) # winit gets randomly initialised
    ica_ms = MultivariateStats.fit(MultivariateStats.ICA
                                 , permutedims(X_array)
                                 , k
                                 ; tol=tolerance)
    Xtr_ms = permutedims(MultivariateStats.transform(ica_ms, permutedims(X_array)))

    # MLJ ICA
    seed!(1234) # winit gets randomly initialised
    ica_mlj = ICA(;k=k, tol=tolerance)
    fitresult, _, _ = MLJBase.fit(ica_mlj, 1, X)
    Xtr_mlj = MLJBase.matrix(MLJBase.transform(ica_mlj, fitresult, X))

    @test Xtr_mlj ≈ Xtr_ms
end

@testset "MulticlassLDA-basic" begin
    Random.seed!(34568)
    # this just reproduces an example they have in the MV repo
    ## prepare data
    d = 5
    ns = [10, 15, 20]
    nc = length(ns)
    n = sum(ns)
    Xs = Matrix{Float64}[]
    ys = Vector{Int}[]
    Ss = Matrix{Float64}[]
    cmeans = zeros(d, nc)

    for k = 1:nc
        R = qr(randn(d, d)).Q
        nk = ns[k]

        Xk = R * Diagonal(2 * rand(d) .+ 0.5) * randn(d, nk) .+ randn(d)
        yk = fill(k, nk)
        uk = vec(mean(Xk, dims=2))
        Zk = Xk .- uk
        Sk = Zk * Zk'

        push!(Xs, Xk)
        push!(ys, yk)
        push!(Ss, Sk)
        cmeans[:,k] .= uk
    end

    X = hcat(Xs...)
    y = vcat(ys...)

    # regular call
    M = fit(MultivariateStats.MulticlassLDA, nc, X, y; regcoef=1e-3)
    fr, = fit(LDA(regcoef=1e-3), 1, X, yc)

    @test fr[2].pmeans ≈ M.pmeans
    @test fr[2].proj ≈ M.proj
end

@testset "MulticlassLDA" begin
    Smarket = dataset("ISLR", "Smarket")
    X      = selectcols(Smarket, [:Lag1,:Lag2])
    y      = selectcols(Smarket, :Direction)
    train  = selectcols(Smarket, :Year) .< 2005
    test   = .!train
    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    Xtest  = selectrows(X, test)
    ytest  = selectrows(y, test)

    LDA_model = LDA()
    fitresult, = fit(LDA_model, 1, Xtrain, ytrain)
    class_means, projection_matrix, prior_probabilities = MLJBase.fitted_params(LDA_model, fitresult)


# XXX predict is still wrong

    predicted_posteriors = predict(LDA_model, fitresult, Xtest)
    predicted_class = predict_mode(LDA_model, fitresult, Xtest)

    test_unit_projection_vector = projection_matrix / norm(projection_matrix)
    R_unit_projection_vector = [-0.642, -0.514] / norm([-0.642, -0.514])
    accuracy = 1 - misclassification_rate(predicted_class, ytest)

    ##tests based on example from Introduction to Statistical Learning in R
    ##
    @test round.(class_means', sigdigits = 3) == [0.0428 0.0339; -0.0395 -0.0313]
    @test round.(prior_probabilities, sigdigits = 3) == [0.492, 0.508]

    d = info_dict(LDA)
    @test d[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test d[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test d[:name] == "LDA"

    # Elementary test
end

end
true
