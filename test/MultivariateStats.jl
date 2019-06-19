module TestMultivariateStats

# using Revise
using Test
using MLJBase
import MultivariateStats
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

    info(ridge)

end

@testset "PCA" begin

    task = load_crabs()
    X, y = X_and_y(task)
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
    task = load_crabs()

    X, y = X_and_y(task)

    kpca = KernelPCA()
    fitresult, cache, report = MLJBase.fit(kpca, 1, X)
    Xtr = MLJBase.matrix(MLJBase.transform(kpca, fitresult, X))
    X_array = MLJBase.matrix(X)

    # TODO: implement a test for synthetic / crabs dataset
    @test false
end

@testset "ICA" begin
    task = load_crabs()

    X, y = X_and_y(task)

    # TODO: implement a test for synthetic / crabs dataset
    @test false
end

end
true
