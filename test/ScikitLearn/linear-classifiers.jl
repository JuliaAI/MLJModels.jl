# iris = dataset("datasets", "iris")
# X = iris[:, 1:4]
# y = iris[:, 5]
#
# yplain = ones(length(y))
# yplain[y .== "setosa"] .= 2
# yplain[y .== "virginica"] .= 3
#
# @testset "RidgeClassifier" begin
#     m = RidgeClassifier()
#     f, _, _ = fit(reg, 1, X, yplain)
#     @test norm(res.score(X, y) - 0.9595) < 1e-4
# end
#
# @testset "RidgeClassifierCV" begin
#     m = RidgeClassifierCV(alphas = [1e-3, 1e-2, 1e-1, 1])
#     f, _, _ = fit(reg, 1, X, yplain)
#     @test norm(res.score(X, y) - 0.9630) < 1e-4
# end
