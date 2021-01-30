using Test, MLJModels

@testset "metadata" begin
    @testset "info_dict" begin
        @test include("info_dict.jl")
    end

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
        @test include("builtins/Constant.jl")
    end
    @testset "Transformers" begin
        @test include("builtins/Transformers.jl")
    end
    @testset "ThresholdPredictors" begin
        @test include("builtins/ThresholdPredictors.jl")
    end
end

@testset "NearestNeighbors" begin
    @test include("NearestNeighbors.jl")
end
