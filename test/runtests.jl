# It is suggested that test code for each include file be placed in
# a file of the same name under "test/" (and included below) and that
# this test code be wrapped in a module. Any new module name will do -
# eg, `module TestDatasets` for code testing `datasets.jl`.

using Test

@testset "metadata" begin

    @testset "reading and extracting model metadata" begin
        @test include("metadata.jl")
    end

    @testset "model search" begin
        @test include("model_search.jl")
    end

    @testset "loading model code" begin
        @test include("loading.jl")
    end

end

@testset "built-in models" begin

    @testset "Constant" begin
        @test include("Constant.jl")
    end

    @testset "Transformers" begin
        @test include("Transformers.jl")
    end

end

@testset "@mlj_model macro" begin
    include("parameters_utils.jl")
end

@testset "MultivariateStats  " begin
    @test include("MultivariateStats.jl")
end

@testset "DecisionTree       " begin
    @test include("DecisionTree.jl")
end

@testset "GaussianProcesses  " begin
    @test include("GaussianProcesses.jl")
end

@testset "Clustering         " begin
    @test include("Clustering.jl")
end

@testset "GLM                " begin
    @test include("GLM.jl")
end

@testset "ScikitLearn        " begin
    @test include("ScikitLearn/ScikitLearn.jl")
end

@testset "LIBSVM             " begin
    @test include("LIBSVM.jl")
end

@testset "NaiveBayes         " begin
    @test include("NaiveBayes.jl")
end

@testset "XGBoost" begin
    @test include("XGBoost.jl")
end

@testset "NearestNeighbors" begin
    @test include("NearestNeighbors.jl")
end
