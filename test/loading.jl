module TestLoading

using Test
using MLJModels

@loadcode AdaBoostStumpClassifier pkg=DecisionTree verbosity=0
@test "AdaBoostStumpClassifier" in map(localmodels()) do m
    m.name
end

@load RidgeRegressor pkg=MultivariateStats verbosity=0

@test isdefined(TestLoading, :RidgeRegressor)
@test MLJModels.info("RidgeRegressor", pkg="MultivariateStats") in
    localmodels(modl=TestLoading)

# if we load the same model again:
# @test_logs((:info, r"For silent"),
#            (:info, r"Model code"),
#            @load(RidgeRegressor,
#                  pkg=MultivariateStats,
#                  scope=:local,
#                  verbosity=1))
@load(RidgeRegressor,
      pkg=MultivariateStats,
      scope=:local,
      verbosity=1)

@test !isdefined(TestLoading, :RidgeRegressor2)

# load the same model again, with a different binding:
@test_logs((:info, r"For silent"),
           (:info, r"Model code"),
           (:warn, r"Ignoring specification"),
           @load(RidgeRegressor,
                 pkg=MultivariateStats,
                 name=Foo,
                 scope=:local,
                 verbosity=1))

# deprecated methods:
@test_throws Exception load("model", pkg = "pkg")
@test_throws Exception load(models()[1])

@testset "scope=:local inside a @testset" begin
    @load RidgeRegressor pkg=MultivariateStats verbosity=0 scope=:local
end

@testset "install_pkgs=true" begin
    @load KMeans pkg=Clustering verbosity=0 scope=:local install_pkgs=true
    @load KMeans pkg=Clustering verbosity=0 scope=:local install=true
end

end # module

true
