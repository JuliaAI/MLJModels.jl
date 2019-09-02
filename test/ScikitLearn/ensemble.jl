# @testset "AdaBoostRegressor" begin
#     m = AdaBoostRegressor(random_state=0, n_estimators=100)
#     f, _, _ = fit(reg, 1, X, y)
#     @test norm(predict(regr, res, [0 0 0 0])[1] - 4.7972) < 1e-4
#     @test norm(res.score(X, y) - 0.9771) < 1e-4
# end
