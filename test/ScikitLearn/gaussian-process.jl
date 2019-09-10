Random.seed!(515)
n, p = 100, 5
X  = randn(n, p)
X1 = hcat(X, ones(100))
θ  = randn(p+1)
y  = X1 * θ .+ 0.1 .* randn(n)

@testset "GPRegressor" begin
    gpr = GaussianProcessRegressor(random_state = 1)
    res, _, _ = fit(gpr, 1, X, y)
    @test res[1].score(X,y) ≈ 1.0
    @test norm(predict(gpr, res, X) .- y) / norm(y) ≤ 1e-10 # overfitting to the max
    fp = fitted_params(gpr, res)
    @test keys(fp) == (:X_train, :y_train, :kernel, :L, :alpha, :log_marginal_likelihood_value)
end
