module TestModelSearch

using Test
using MLJModels
import MLJBase.info

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

@testset "models(needle::Union{AbstractString,Regex}) and localmodels(needle::Union{AbstractString,Regex})" begin
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
