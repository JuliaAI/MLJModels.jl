module TestScikitLearn

using MLJBase
using Tables
using Test
using LinearAlgebra
using Random

import MLJModels
import ScikitLearn
using MLJModels.ScikitLearn_
using CategoricalArrays
using RDatasets

include("../testutils.jl")

include("svm.jl")
include("linear-regressors.jl")
include("linear-classifiers.jl")
include("gaussian-process.jl")
include("ensemble.jl")
include("misc.jl")

end
true
