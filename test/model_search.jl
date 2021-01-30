module TestModelSearch

using Test
using MLJModels
import MLJBase
using MLJScientificTypes

pca = info("PCA", pkg="MultivariateStats")
cnst = info("ConstantRegressor", pkg="MLJModels")

@test_throws ArgumentError info("Julia")

@info(ConstantRegressor) == cnst
@test info(Standardizer()) == info("Standardizer", pkg="MLJModels")

@testset "localmodels" begin
    tree = info("DecisionTreeRegressor")
    @test cnst in localmodels(modl=TestModelSearch)
    @test !(tree in localmodels(modl=TestModelSearch))
    import MLJDecisionTreeInterface.DecisionTreeRegressor
    @test tree in localmodels(modl=TestModelSearch)
end

@testset "matching" begin
    X = ones(2,3)
    y = ones(2)
    w = ones(2)
    w_class = Dict(:male => 1.0, :female => 10.0)

    task = matching(X)
    @test !task.is_supervised
    @test ismissing(task.supports_weights)
    @test ismissing(task.supports_class_weights)
    @test task.input_scitype == scitype(X)
    @test ismissing(task.target_scitype)

    task = matching(X, y)
    @test task.is_supervised
    @test ismissing(task.supports_weights)
    @test ismissing(task.supports_class_weights)
    @test task.input_scitype == scitype(X)
    @test task.target_scitype == scitype(y)

    task = matching(X, y, w)
    @test task.is_supervised
    @test task.supports_weights
    @test !task.supports_class_weights
    @test task.input_scitype == scitype(X)
    @test task.target_scitype == scitype(y)

    task = matching(X, y, w_class)
    @test task.is_supervised
    @test !task.supports_weights
    @test task.supports_class_weights
    @test task.input_scitype == scitype(X)
    @test task.target_scitype == scitype(y)
end

@testset "models() and localmodels()" begin
    t(model) = model.is_pure_julia
    mods = models(t)
    @test pca in mods
    @test cnst in mods
    @test !(info("SVC") in mods)
    mods = localmodels(t, modl=TestModelSearch)
    @test cnst in mods
    @test !(pca in mods)
    u(model) = !(model.is_supervised)
    @test pca in models(u, t)
    @test !(cnst in models(u, t))
end

@testset "models(matching())" begin
    X = MLJBase.table(ones(2,3))
    y = MLJBase.coerce(["a", "b"], Multiclass)
    w = ones(2)
    ms = models(matching(X, y, w))

    # by hand:
    ms2 = models() do m
        m.is_supervised &&
            AbstractVector{Multiclass{2}} <: m.target_scitype &&
            Table(Continuous) <: m.input_scitype &&
            m.supports_weights
    end
    @test ms == ms2
end

@testset "models(needle) and localmodels(needle)" begin
    @test pca in models("PCA")
    @test pca ∉ models("PCA′")

    @test pca in models(r"PCA")
    @test pca in models(r"pca"i)
    @test pca ∉ models(r"PCA′")

    info("DecisionTreeRegressor") in localmodels("Decision"; modl = TestModelSearch)
    info("DecisionTreeRegressor") in localmodels(r"Decision"; modl = TestModelSearch)
end

end
true
