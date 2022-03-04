module TestModelSearch

using Test
using MLJModels
using MLJBase
using ScientificTypes
using Markdown

@testset "user interface to get handle" begin
    @test_throws MLJModels.err_handle_missing_name("julia") MLJModels.handle("julia")
    @test_throws(MLJModels.err_handle_name_not_in_pkg("PCA", "MLJModels"),
                 MLJModels.handle("PCA", pkg="MLJModels"))

    # can't use @test_throws because there are 2 possible correct throws:
    success = false
    try
        MLJModels.handle("DecisionTreeClassifier")
    catch exception
        if exception in [MLJModels.err_handle_ambiguous_name("DecisionTreeClassifier",
                                                             ["DecisionTree", "BetaML"]),
                         MLJModels.err_handle_ambiguous_name("DecisionTreeClassifier",
                                                             ["BetaML", "DecisionTreee"])]
            success = true
        end
    end
    @test success
end

pca = info("PCA", pkg="MultivariateStats")
cnst = info("ConstantRegressor", pkg="MLJModels")
tree = info("DecisionTreeRegressor", pkg="DecisionTree")

@testset "info and doc" begin
    @test_throws ArgumentError info("Julia")

    # Note that these tests assume model registry metadata is up to date
    # with the latest trait values in `src/builtins/`:
    @test info(ConstantRegressor) == cnst
    @test info(Standardizer()) == info("Standardizer", pkg="MLJModels")
    @test doc("ConstantRegressor", pkg="MLJModels") == cnst.docstring |> Markdown.parse
    @test_throws MLJModels.ERR_DOC_EXPECTS_STRING doc(ConstantRegressor)
    @test_throws MLJModels.ERR_DOC_EXPECTS_STRING doc(ConstantRegressor())
end

@testset "localmodels" begin
    @test issubset(Set([DeterministicConstantClassifier,
                        DeterministicConstantRegressor,
                        ConstantClassifier,
                        ConstantRegressor,
                        FeatureSelector,
                        OneHotEncoder,
                        Standardizer,
                        UnivariateBoxCoxTransformer,
                        UnivariateStandardizer]),
                   MLJModels.localmodeltypes(MLJModels))
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

@testset "models(needle)" begin
    @test pca in models("PCA")
    @test pca ∉ models("PCA′")
    @test pca in models(r"PCA")
    @test pca ∉ models(r"PCA′")
end

end
true
