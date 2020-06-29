module TestMultivariateStats

using Test
using MLJBase
import MultivariateStats, Random
using MLJModels.MultivariateStats_
import Dates

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
    Xtable = table(A)
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

X, y = @load_crabs

@testset "PCA" begin
    X_array = matrix(X)
    pratio = 0.9999

    # MultivariateStats PCA
    pca_ms = MultivariateStats.fit(MultivariateStats.PCA,
                                   permutedims(X_array),
                                   pratio=pratio)
    Xtr_ms = permutedims(MultivariateStats.transform(pca_ms,
                                                     permutedims(X_array)))

    # MLJ PCA
    pca_mlj = PCA(pratio=pratio)
    fitresult, _, _ = fit(pca_mlj, 1, X)
    Xtr_mlj = matrix(transform(pca_mlj, fitresult, X))

    @test Xtr_mlj ≈ Xtr_ms
end

@testset "KernelPCA" begin
    X_array = matrix(X)

    # MultivariateStats KernelPCA
    kpca_ms = MultivariateStats.fit(MultivariateStats.KernelPCA
                                  , permutedims(X_array))
    Xtr_ms = permutedims(MultivariateStats.transform(kpca_ms,
                                                     permutedims(X_array)))

    # MLJ KernelPCA
    kpca_mlj = KernelPCA()
    fitresult, _, _ = fit(kpca_mlj, 1, X)
    Xtr_mlj = matrix(transform(kpca_mlj, fitresult, X))

    @test Xtr_mlj ≈ Xtr_ms
end

@testset "ICA" begin
    X_array = matrix(X)
    k = 5
    tolerance = 5.0

    # MultivariateStats ICA
    seed!(1234) # winit gets randomly initialised
    ica_ms = MultivariateStats.fit(MultivariateStats.ICA,
                                   permutedims(X_array),
                                   k;
                                   tol=tolerance)
    Xtr_ms = permutedims(MultivariateStats.transform(ica_ms,
                                                     permutedims(X_array)))

    # MLJ ICA
    seed!(1234) # winit gets randomly initialised
    ica_mlj = ICA(;k=k, tol=tolerance)
    fitresult, _, _ = fit(ica_mlj, 1, X)
    Xtr_mlj = matrix(transform(ica_mlj, fitresult, X))

    @test Xtr_mlj ≈ Xtr_ms
end

@testset "MulticlassLDA" begin
    Xfull, y = @load_smarket
    X = selectcols(Xfull, [:Lag1,:Lag2])

    train  = selectcols(Xfull, :Year) .< Dates.Date(2005)
    test   = .!train
    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    Xtest  = selectrows(X, test)
    ytest  = selectrows(y, test)

    LDA_model = LDA()
    fitresult, = fit(LDA_model, 1, Xtrain, ytrain)
    class_means, projection_matrix = fitted_params(LDA_model, fitresult)

    preds = predict(LDA_model, fitresult, Xtest)

    mce = cross_entropy(preds, ytest) |> mean

    @test 0.685 ≤ mce ≤ 0.695

    @test round.(class_means', sigdigits = 3) == [0.0428 0.0339; -0.0395 -0.0313]

    d = info_dict(LDA)
    @test d[:input_scitype] == Table(Continuous)
    @test d[:target_scitype] == AbstractVector{<:Finite}
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
    X = table(X)
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

@testset "BayesianMulticlassLDA" begin
    Xfull, y = @load_smarket
    X = selectcols(Xfull, [:Lag1,:Lag2])

    train  = selectcols(Xfull, :Year) .< Dates.Date(2005)
    test   = .!train
    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    Xtest  = selectrows(X, test)
    ytest  = selectrows(y, test)

    BLDA_model = BayesianLDA()
    fitresult, = fit(BLDA_model, 1, Xtrain, ytrain)
    class_means, projection_matrix, priors = fitted_params(BLDA_model, fitresult)

    preds = predict(BLDA_model, fitresult, Xtest)

    mce = cross_entropy(preds, ytest) |> mean

    @test 0.685 ≤ mce ≤ 0.695

    @test round.(class_means', sigdigits = 3) ==
        [0.0428 0.0339; -0.0395 -0.0313]

    d = info_dict(BayesianLDA)
    @test d[:input_scitype] == Table(Continuous)
    @test d[:target_scitype] == AbstractVector{<:Finite}
    @test d[:name] == "BayesianLDA"
end

@testset "BayesianSubspaceLDA" begin
    X, y = @load_iris

    LDA_model = BayesianSubspaceLDA()

    fitresult, _, report = fit(LDA_model, 1, X, y)
    class_means,projection_matrix,prior_probabilities =
        fitted_params(LDA_model, fitresult)
    preds=predict(LDA_model, fitresult, X)
    predicted_class = predict_mode(LDA_model, fitresult, X)
    mcr = misclassification_rate(predicted_class, y)
    mce = cross_entropy(preds, y) |> mean

    @test mean(abs.(class_means' - [5.006 3.428 1.462 0.246;
                          5.936 2.770 4.260 1.326;
                          6.588 2.974 5.552 2.026])) < 0.01

    @test mean(abs.(projection_matrix ≈  [0.8293776  0.02410215;
                                1.5344731  2.16452123;
                                -2.2012117 -0.93192121;
                                          -2.8104603  2.83918785])) < 0.05

    @test round.(prior_probabilities, sigdigits=7) ==
        [0.3333333, 0.3333333, 0.3333333]
    @test round.(mcr, sigdigits=1) == 0.02

    @test round.(report.explained_variance_ratio, digits=4) == [0.9915, 0.0085]
    @test 0.04 ≤ mce ≤ 0.045

    d = info_dict(BayesianSubspaceLDA)
    @test d[:input_scitype] == Table(Continuous)
    @test d[:target_scitype] == AbstractVector{<:Finite}
    @test d[:name] == "BayesianSubspaceLDA"
end

@testset "SubspaceLDA" begin
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
    X = table(X)
    y = categorical(y)
    train, test = partition(eachindex(y), 0.7)
    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    Xtest = selectrows(X, test)
    ytest = selectrows(y, test)

    lda_model = SubspaceLDA()
    fitresult, = fit(lda_model, 1, Xtrain, ytrain)
    preds = predict_mode(lda_model, fitresult, Xtest)
    mcr = misclassification_rate(preds, ytest)
    @test mcr ≤ 0.15

    # MultivariateStats Linear Discriminant Analysis transform
     proj    = fitresult[1].projw
     projLDA = fitresult[1].projLDA
     proj =  proj * projLDA
     XWt = matrix(X) * proj
     tlda_ms = table(XWt, prototype=X)

    # MLJ Linear Discriminant Analysis transform
     tlda_mlj = transform(lda_model, fitresult, X)

     @test tlda_mlj == tlda_ms

     d = info_dict(SubspaceLDA)
     @test d[:input_scitype] == Table(Continuous)
     @test d[:target_scitype] == AbstractVector{<:Finite}
     @test d[:name] == "SubspaceLDA"
end

end
true
