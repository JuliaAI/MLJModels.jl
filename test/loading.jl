module TestLoading

# using Revise
using Test
using MLJModels

@testset "`load` function" begin

    load("RidgeRegressor",
         pkg="MultivariateStats",
         modl=TestLoading,
         verbosity=1)

    @test isdefined(TestLoading, :RidgeRegressor)
    @test MLJModels.info("RidgeRegressor", pkg="MultivariateStats") in
    localmodels(modl=TestLoading)

    # load the same model again:
    @test_logs (:info, r"Model code") begin
        load("RidgeRegressor",
             pkg="MultivariateStats",
             modl=TestLoading,
             verbosity=1)
    end

    @test !isdefined(TestLoading, :RidgeRegressor2)

    # load the same model again, with a different binding:
    load("RidgeRegressor",
         pkg="MultivariateStats",
         modl=TestLoading,
         verbosity=1,
         name="Foo")

    @test Foo() == RidgeRegressor()

    # load a model with same name from different package:
    @test_logs (:warn, r"New model type") begin
         load("RidgeRegressor",
              pkg="MLJLinearModels",
              modl=TestLoading,
              verbosity=0)
    end

    @test typeof(RidgeRegressor2()) != typeof(RidgeRegressor())

    # try to use the name of an existing object for new type name
    @test_throws Exception load("DecisionTreeClassifier",
                                pkg="DecisionTree",
                                modl=TestLoading,
                                verbosity=0,
                                name="RidgeRegressor")
end

@testset "@load macro" begin
    @load PCA
    @test isdefined(TestLoading, :PCA)
    @test_throws(ArgumentError, load("RidgeRegressor", modl=TestLoading))
end

end # module

true

