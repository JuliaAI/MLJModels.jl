module TestLoading

using Test
using MLJModels

function isloaded(name::String, pkg::String)
    (name, pkg) in map(localmodels()) do m
        (m.name, m.package_name)
    end
end

@load AdaBoostStumpClassifier pkg=DecisionTree verbosity=0
@test isloaded("AdaBoostStumpClassifier", "DecisionTree")

# built-ins load fine:
@load Standardizer verbosity=0

# load one version of a RidgeRegressor:
@test !isloaded("RidgeRegressor", "MultivariateStats")
@load RidgeRegressor pkg=MultivariateStats verbosity=0
@test isloaded("RidgeRegressor", "MultivariateStats")

# error if ambiguous:
@test_throws ArgumentError @load RidgeRegressor

# error if not in project:
@test !isloaded("KMeans", "Clustering")
@test_throws ArgumentError @load KMeans pkg=Clustering verbosity=0

# use add option:
@load KMeans pkg=Clustering verbosity=0 add=true
@test isloaded("KMeans", "Clustering")

# deprecated methods:
@test_throws Exception load("model", pkg = "pkg")
@test_throws Exception load(models()[1])

module FooBar
using MLJModels
function regressor()
    Regressor = @load LinearRegressor pkg=MultivariateStats verbosity=0
    return Regressor()
end
end
using .FooBar

@testset "@load from within a function within a module" begin
    model = FooBar.regressor()
    @test isdefined(model, :bias)
end

end # module

true
