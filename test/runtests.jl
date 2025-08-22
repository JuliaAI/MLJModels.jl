import Pkg

using Test, MLJModels, MLJTransforms

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
    @testset "ThresholdPredictors" begin
        @test include("builtins/ThresholdPredictors.jl")
    end
end

if parse(Bool, get(ENV, "MLJ_TEST_REGISTRY", "false"))
    @testset "registry" begin
        @test include("registry.jl")
    end
else
    @info "Test of the MLJ Registry is being skipped. Set environment variable "*
        "MLJ_TEST_REGISTRY = \"true\" to include them.\n"*
        "The Registry test takes at least one hour. "
end
