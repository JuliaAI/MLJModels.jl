@testset "GaussianProcesses" begin
    gpr = GaussianProcessRegressor(random_state = 1)
    res, _, _ = fit(gpr, 1, X, y)
    @test res.score(X,y) ≈ 1.0
    @test norm(predict(gpr, res, X) .- y) / norm(y) ≤ 1e-10 # overfitting to the max
end
