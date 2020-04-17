module MLJModels

# temporary fix for #212
using MKL_jll

import MLJModelInterface
import MLJModelInterface: MODEL_TRAITS

using ScientificTypes
using MLJBase
import MLJBase: @load
import MLJBase: Table, Continuous, Count, Finite, OrderedFactor, Multiclass

using Requires, Pkg, Pkg.TOML, OrderedCollections, Parameters
using Tables, CategoricalArrays, StatsBase, Statistics
import Distributions

# for administrators to update Metadata.toml:
export @update, check_registry

# from loading.jl:
export load, @load, info

# from model_search:
export models, localmodels

# from model/Constant
export ConstantRegressor, ConstantClassifier,
        DeterministicConstantRegressor, DeterministicConstantClassifier

# from model/Transformers
export FeatureSelector, StaticTransformer, UnivariateDiscretizer, UnivariateStandardizer,
        Standardizer, UnivariateBoxCoxTransformer, OneHotEncoder, FillImputer

const srcdir = dirname(@__FILE__) # the directory containing this file

# TODO remove when the functionality has been merged in ScientificTypes.jl
# and use ScientificTypes.nonmissing then.
if VERSION < v"1.3"
    nonmissingtype(::Type{T}) where T =
        T isa Union ? ifelse(T.a == Missing, T.b, T.a) : T
end
nonmissing = nonmissingtype

include("metadata.jl")
include("model_search.jl")
include("loading.jl")
include("registry/src/Registry.jl")
include("registry/src/check_registry.jl")
import .Registry.@update

# load built-in models:
include("builtins/Constant.jl")
include("builtins/Transformers.jl")

const INFO_GIVEN_HANDLE = Dict{Handle,Any}()
const PKGS_GIVEN_NAME   = Dict{String,Vector{String}}()
const AMBIGUOUS_NAMES   = String[]
const NAMES             = String[]

metadata_file = joinpath(srcdir, "registry", "Metadata.toml")

merge!(INFO_GIVEN_HANDLE, info_given_handle(metadata_file))
merge!(PKGS_GIVEN_NAME, pkgs_given_name(INFO_GIVEN_HANDLE))
append!(AMBIGUOUS_NAMES, ambiguous_names(INFO_GIVEN_HANDLE))
append!(NAMES, model_names(INFO_GIVEN_HANDLE))
@info "Model metadata loaded from registry. "

# lazily load in strap-on model interfaces for external packages:
function __init__()
    @require(DecisionTree="7806a523-6efd-50cb-b5f6-3fa6f1930dbb",
             include("DecisionTree.jl"))
    @require(GaussianProcesses="891a1506-143c-57d2-908e-e1f8e92e6de9",
             include("GaussianProcesses.jl"))
    @require(GLM="38e38edf-8417-5370-95a0-9cbb8c7f171a",
             include("GLM.jl"))
    @require(Clustering="aaaa29a8-35af-508c-8bc3-b662a17a0fe5",
             include("Clustering.jl"))
    @require(NaiveBayes="9bbee03b-0db5-5f46-924f-b5c9c21b8c60",
             include("NaiveBayes.jl"))
    @require(ScikitLearn="3646fa90-6ef7-5e7e-9f22-8aca16db6324",
             include("ScikitLearn/ScikitLearn.jl"))
    @require(XGBoost = "009559a3-9522-5dbb-924b-0b6ed2b22bb9",
             include("XGBoost.jl"))
    @require(LIBSVM="b1bec4e5-fd48-53fe-b0cb-9723c09d164b",
             include("LIBSVM.jl"))
    @require(NearestNeighbors="b8a86587-4115-5ab1-83bc-aa920d37bbce",
             include("NearestNeighbors.jl"))
    @require(MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411",
             include("MultivariateStats.jl"))
end

end # module
