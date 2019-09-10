module MLJModels
__precompile__(false)

# for administrators to update Metadata.toml:
export @update

# from builtins/Transformers.jl:
export StaticTransformer, FeatureSelector,
    UnivariateStandardizer, Standardizer,
    UnivariateBoxCoxTransformer,
    OneHotEncoder

# from builtins/Constant.jl:
export ConstantRegressor, ConstantClassifier

# from builtins/KNN.jl:
export KNNRegressor

# from loading.jl:
export load, @load, info

# from model_search:
export models, localmodels

using Requires
using OrderedCollections
using MLJBase
using ScientificTypes
using Tables
using ColorTypes
using StatsBase

using Pkg.TOML

const srcdir = dirname(@__FILE__) # the directory containing this file

include("metadata.jl")
include("model_search.jl")
include("loading.jl")
include("registry/src/Registry.jl")
import .Registry.@update

# load built-in models:
include("builtins/Transformers.jl")
include("builtins/Constant.jl")
include("builtins/KNN.jl")

function __init__()

    # load and extract the registry model metadata from the:
    try
        global metadata_file = joinpath(srcdir, "registry", "Metadata.toml")
        global INFO_GIVEN_HANDLE = info_given_handle(metadata_file)
        global AMBIGUOUS_NAMES = ambiguous_names(INFO_GIVEN_HANDLE)
        global PKGS_GIVEN_NAME = pkgs_given_name(INFO_GIVEN_HANDLE)
        global NAMES = model_names(INFO_GIVEN_HANDLE)
        @info "Model metadata loaded from registry. "
    catch
        @warn "Problem loading registry from $metadata_file. "*
        "Model search and model code loading disabled. "
    end
    
    # lazily load in strap-on model interfaces for external packages:
    @require MultivariateStats="6f286f6a-111f-5878-ab1e-185364afe411" include("MultivariateStats.jl")
    @require DecisionTree="7806a523-6efd-50cb-b5f6-3fa6f1930dbb" include("DecisionTree.jl")
    @require GaussianProcesses="891a1506-143c-57d2-908e-e1f8e92e6de9" include("GaussianProcesses.jl")
    @require GLM="38e38edf-8417-5370-95a0-9cbb8c7f171a" include("GLM.jl")
    @require Clustering="aaaa29a8-35af-508c-8bc3-b662a17a0fe5" include("Clustering.jl")
    @require NaiveBayes="9bbee03b-0db5-5f46-924f-b5c9c21b8c60" include("NaiveBayes.jl")
    @require ScikitLearn="3646fa90-6ef7-5e7e-9f22-8aca16db6324" include("ScikitLearn/ScikitLearn.jl")
    @require XGBoost = "009559a3-9522-5dbb-924b-0b6ed2b22bb9" include("XGBoost.jl")
    @require LIBSVM="b1bec4e5-fd48-53fe-b0cb-9723c09d164b" include("LIBSVM.jl")
    
end

end # module
