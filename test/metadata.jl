module TestMetadata

using Test
using MLJModels
import MLJBase

metadata_file = joinpath(@__DIR__, "..", "src",
                         "registry", "Metadata.toml")
pca = MLJModels.Handle("PCA", "MultivariateStats")
cnst = MLJModels.Handle("ConstantRegressor", "MLJModels")
i = MLJModels.info_given_handle(metadata_file)[pca]

@testset "Handle constructors" begin
    @test MLJModels.Handle("PCA") ==
        MLJModels.Handle("PCA", "MultivariateStats")
    # TODO: add tests here when duplicate model names enter registry
end

@testset "building INFO_GIVEN_HANDLE" begin
    @test isempty(MLJModels.localmodeltypes(MLJBase))
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
    @test MLJModels.info_given_handle(metadata_file)[pca][:name] == "PCA"
    d1 = MLJModels.info_given_handle(metadata_file)[cnst]
    d2 = MLJBase.info_dict(ConstantRegressor)
    for (k, v) in d1
        if v isa Vector
            @test Set(v) == Set(d2[k])
        else
            @test v == d2[k]
        end
    end
end

h = Vector{Any}(undef, 7)
h[1] = MLJModels.Handle("1", "a")
h[3] = MLJModels.Handle("1", "b")
h[2] = MLJModels.Handle("2", "b")
h[4] = MLJModels.Handle("3", "b")
h[5] = MLJModels.Handle("4", "c")
h[6] = MLJModels.Handle("3", "d")
h[7] = MLJModels.Handle("5", "e")
info_given_handle = Dict([h[j]=>i for j in 1:7]...)

@testset "building AMBIGUOUS_NAMES" begin
    @test Set(MLJModels.ambiguous_names(info_given_handle)) == Set(["1", "3"])
end

@testset "building PKGS_GIVEN_NAME" begin
    d = MLJModels.pkgs_given_name(info_given_handle)
    @test Set(d["1"]) == Set(["a", "b"])
    @test d["2"]==["b",]
    @test Set(d["3"]) == Set(["b", "d"])
    @test d["4"] == ["c",]
    @test d["5"] == ["e",]
end

@testset "building NAMES" begin
    model_names = MLJModels.model_names(info_given_handle)
    @test Set(model_names) == Set(["1", "2", "3", "4", "5"])
end

end
true
