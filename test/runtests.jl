# It is suggested that test code for each include file be placed in
# a file of the same name under "test/" (and included below) and that
# this test code be wrapped in a module. Any new module name will do -
# eg, `module TestDatasets` for code testing `datasets.jl`.

using Test
using MLJBase
using CSV

@testset "MultivariateStats" begin
  @test include("MultivariateStats.jl")
end

@testset "DecisionTree" begin
  @test include("DecisionTree.jl")
end

@testset "GaussianProcesses" begin
  @test include("GaussianProcesses.jl")
end

@testset "Clustering" begin
    @test include("Clustering.jl")
end

@testset "GLM" begin
    @test include("GLM.jl")
end

@testset "SVM" begin
    @test include("ScikitLearn.jl")
end

@testset "LIBSVM" begin
    @test include("LIBSVM.jl")
end

@testset "NaiveBayes" begin
    @test include("NaiveBayes.jl")
end

# brittle hack b/s of https://github.com/dmlc/XGBoost.jl/issues/58:
# using Pkg
#Pkg.add(PackageSpec(url="https://github.com/dmlc/XGBoost.jl"))
#Pkg.add(PackageSpec(name="XGBoost", version="0.3.0"))

@testset "XGBoost" begin
    @test include("XGBoost.jl")
 end
