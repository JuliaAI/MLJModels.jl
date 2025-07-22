import Pkg

using Test, MLJModels

@testset "registry_project" begin
    @test include("registry_project.jl")
end

@testset "metadata" begin
    @testset "metadata.jl" begin
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
    @testset "Constant.jl" begin
        @test include("builtins/Constant.jl")
    end
    @testset "Transformers.jl" begin
        @test include("builtins/Transformers.jl")
    end
    @testset "ThresholdPredictors" begin
        @test include("builtins/ThresholdPredictors.jl")
    end
end
