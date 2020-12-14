module TestLoading

# using Revise
using Test
using MLJModels

@loadcode RandomForestClassifier pkg=DecisionTree verbosity=0

@load RidgeRegressor pkg=MultivariateStats verbosity=0

@test isdefined(TestLoading, :RidgeRegressor)
@test MLJModels.info("RidgeRegressor", pkg="MultivariateStats") in
    localmodels(modl=TestLoading)

# if we load the same model again:
program, _ = @test_logs((:info, r"For silent"),
           (:info, r"Model code"),
           MLJModels._load(TestLoading,
                           :(RidgeRegressor),
                           :(pkg=MultivariateStats)))
eval(program)

@test !isdefined(TestLoading, :RidgeRegressor2)

# load the same model again, with a different binding:
@load RidgeRegressor pkg=MultivariateStats name=Foo

@test Foo() == RidgeRegressor()

# try to use the name of an existing object for new type name
program, _ = MLJModels._load(TestLoading,
                :(DecisionTreeClassifier),
                :(pkg=DecisionTree),
                             :(name=RidgeRegressor))

@test_throws Exception eval(program)

@test_throws Exception load("model", pkg = "pkg")
@test_throws Exception load(models()[1])

@testset "scope=:local inside a @testset" begin
    @load RidgeRegressor pkg=MultivariateStats verbosity=0 scope=:local
end

@testset "install_pkgs=true" begin
    @load KMeans pkg=Clustering verbosity=0 scope=:local install_pkgs=true
    @load KMeans pkg=Clustering verbosity=0 scope=:local install_pkgs=true
end

end # module

true
