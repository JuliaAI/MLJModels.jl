module TestMultivariateStats

# using Revise
using Test
using MLJBase
using RDatasets
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
    ica_mlj = ICA(k; tol=tolerance)
    fitresult, _, _ = MLJBase.fit(ica_mlj, 1, X)
    Xtr_mlj = MLJBase.matrix(MLJBase.transform(ica_mlj, fitresult, X))

    @test Xtr_mlj ≈ Xtr_ms

end

end
true
