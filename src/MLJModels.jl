module MLJModels

import MLJModelInterface
import MLJModelInterface: MODEL_TRAITS

using MLJScientificTypes
const ScientificTypes = MLJScientificTypes.ScientificTypes

using MLJBase
import MLJBase.@load

using Requires, Pkg, Pkg.TOML, OrderedCollections, Parameters
using Tables, CategoricalArrays, StatsBase, Statistics, Dates
using MacroTools
import Distributions
import REPL # stdlib, needed for Term

# for administrators to update Metadata.toml:
export @update, check_registry

# from loading.jl:
export load, @load, @iload, @loadcode, info

# from model_search:
export models, localmodels

# from model/Constant
export ConstantRegressor, ConstantClassifier,
        DeterministicConstantRegressor, DeterministicConstantClassifier

# from model/ThresholdPredictors
export BinaryThresholdPredictor

# from model/Transformers
export FeatureSelector, StaticTransformer, UnivariateDiscretizer,
    UnivariateStandardizer, Standardizer, UnivariateBoxCoxTransformer,
    OneHotEncoder, ContinuousEncoder, FillImputer, UnivariateFillImputer,
    UnivariateTimeTypeToContinuous

const srcdir = dirname(@__FILE__) # the directory containing this file
const MMI = MLJModelInterface

if VERSION < v"1.3"
    nonmissingtype = MLJScientificTypes.ScientificTypes.nonmissing
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
include("builtins/ThresholdPredictors.jl")

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
    @require(NearestNeighbors="b8a86587-4115-5ab1-83bc-aa920d37bbce",
             include("NearestNeighbors.jl"))
end

end # module
