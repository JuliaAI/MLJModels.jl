module MLJModels

# from "datasets.jl":

using Requires

const srcdir = dirname(@__FILE__) # the directory containing this file:

function __init__()
    @require DecisionTree="7806a523-6efd-50cb-b5f6-3fa6f1930dbb" include("DecisionTree.jl")
end


end # module
