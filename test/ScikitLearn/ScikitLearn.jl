module TestScikitLearn

import MLJModels, ScikitLearn
using MLJBase
using Tables
using Test
using LinearAlgebra
using Random
using MLJModels.ScikitLearn_
using CategoricalArrays

include("../testutils.jl")
include("clustering.jl")
include("svm.jl")
include("linear-regressors.jl")
include("linear-classifiers.jl")
include("gaussian-process.jl")
include("ensemble.jl")
include("discriminant-analysis.jl")
include("misc.jl")

end
true
