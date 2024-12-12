import Pkg

if Base.VERSION >= v"1.10-"
    # The issue with stdlib versions being fixed to 0.0.0 has been fixed in new versions of Julia
else
    # The next line added as a workaround to
    # https://github.com/JuliaLang/Pkg.jl/issues/3628 (Julia 1.6):
    Pkg.add(name="Statistics", version=VERSION, uuid="10745b16-79ce-11e8-11f9-7d13ad32a3b2")
end

using Test, MLJModels

@testset "registry" begin
    @test include(joinpath("..", "src", "registry", "test", "runtests.jl"))
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
