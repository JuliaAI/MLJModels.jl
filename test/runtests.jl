using Test, MLJModels, Conda

#Conda.add("numpy")
#Conda.add("nomkl")
Conda.add("scikit-learn")
#Conda.add("numexpr")
#Conda.rm("mkl")


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
        @test include("builtins/Constant.jl")
    end
    @testset "Transformers" begin
        @test include("builtins/Transformers.jl")
    end
end

@testset "MultivariateStats  " begin
    @test include("MultivariateStats.jl")
end

@testset "DecisionTree       " begin
    @test include("DecisionTree.jl")
end

# @testset "GaussianProcesses  " begin
#     @test include("GaussianProcesses.jl")
# end

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
