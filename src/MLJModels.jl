module MLJModels

# from "datasets.jl":

using Requires

const srcdir = dirname(@__FILE__) # the directory containing this file:

function __init__()
    @require DecisionTree="7806a523-6efd-50cb-b5f6-3fa6f1930dbb" include("DecisionTree.jl")
    @require GaussianProcesses="891a1506-143c-57d2-908e-e1f8e92e6de9" include("GaussianProcesses.jl")
    @require GLM="38e38edf-8417-5370-95a0-9cbb8c7f171a" include("GLM.jl")
    @require Clustering="aaaa29a8-35af-508c-8bc3-b662a17a0fe5" include("Clustering.jl")
    @require ScikitLearn="3646fa90-6ef7-5e7e-9f22-8aca16db6324" include("ScikitLearn.jl")
end



end # module
