# It is suggested that test code for each include file be placed in
# a file of the same name under "test/" (and included below) and that
# this test code be wrapped in a module. Any new module name will do -
# eg, `module TestDatasets` for code testing `datasets.jl`.

using Test

# using Pkg
# Pkg.add(PackageSpec(url="https://github.com/alan-turing-institute/MLJ.jl", rev="master"))

@testset "DecisionTree" begin
  @test include("DecisionTree.jl")
end

@testset "GaussianProcesses" begin
  @test include("GaussianProcesses.jl")
end

# @testset "MultivariateStats" begin
#   @test include("LocalMultivariateStats.jl")
# end

@testset "Clustering" begin
    @test include("Clustering.jl")
end

@testset "GLM" begin
    @test include("GLM.jl")
end

@testset "SVM" begin
    @test include("ScikitLearn.jl")
end

@testset "NaiveBayes" begin
    @test include("NaiveBayes.jl")
end

# @testset "XGBoost" begin
#   @test_broken include("XGBoost.jl")
# end
