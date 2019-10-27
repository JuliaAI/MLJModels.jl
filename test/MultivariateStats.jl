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
    intercept = -42.0
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

    # Get the true intercept?
    fr = fitted_params(ridge, fitresult)
    @test abs(fr.intercept) < 1e-10
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
    ica_ms = MultivariateStats.fit(MultivariateStats.ICA, permutedims(X_array), k;
                                   tol=tolerance)
    Xtr_ms = permutedims(MultivariateStats.transform(ica_ms, permutedims(X_array)))

    # MLJ ICA
    seed!(1234) # winit gets randomly initialised
    ica_mlj = ICA(;k=k, tol=tolerance)
    fitresult, _, _ = MLJBase.fit(ica_mlj, 1, X)
    Xtr_mlj = MLJBase.matrix(MLJBase.transform(ica_mlj, fitresult, X))

    @test Xtr_mlj ≈ Xtr_ms
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
    class_means, projection_matrix = MLJBase.fitted_params(LDA_model, fitresult)

    preds = predict(LDA_model, fitresult, Xtest)

    mce = MLJBase.cross_entropy(preds, ytest) |> mean

    @test 0.69 ≤ mce ≤ 0.695

    @test round.(class_means', sigdigits = 3) == [0.0428 0.0339; -0.0395 -0.0313]

    d = info_dict(LDA)
    @test d[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
    @test d[:target_scitype] == AbstractVector{<:MLJBase.Finite}
    @test d[:name] == "LDA"
end

@testset "MLDA-2" begin
    Random.seed!(1125)
    X1 = -2 .+ randn(100, 2)
    X2 = randn(100, 2)
    X3 = 2 .+ randn(100, 2)
    y1 = ones(100)
    y2 = 2ones(100)
    y3 = 3ones(100)
    X = vcat(X1, X2, X3)
    y = vcat(y1, y2, y3)
    p = Random.randperm(300)
    X = X[p, :]
    y = y[p]
    X = MLJBase.table(X)
    y = categorical(y)
    train, test = partition(eachindex(y), 0.7)
    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    Xtest = selectrows(X, test)
    ytest = selectrows(y, test)

    lda_model = LDA()
    fitresult, = fit(lda_model, 1, Xtrain, ytrain)
    preds = predict_mode(lda_model, fitresult, Xtest)
    mcr = misclassification_rate(preds, ytest)
    @test mcr ≤ 0.15
end

end
true
