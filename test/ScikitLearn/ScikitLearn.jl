module TestScikitLearn

import MLJModels, ScikitLearn
using MLJBase, Tables, Test, LinearAlgebra, Random, MLJModels.ScikitLearn_, CategoricalArrays, RDatasets
include("../testutils.jl")

include("svm.jl")
include("linear-regressors.jl")
include("linear-classifiers.jl")
include("gaussian-process.jl")
include("ensemble.jl")
include("misc.jl")

end
true
