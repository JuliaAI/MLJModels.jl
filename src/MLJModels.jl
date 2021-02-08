module MLJModels

import MLJModelInterface

using MLJScientificTypes
const ScientificTypes = MLJScientificTypes.ScientificTypes

# still needed b/s UnivariateFinite type used in ThresholdPredictor:
using MLJBase

using Pkg, Pkg.TOML, OrderedCollections, Parameters
using Tables, CategoricalArrays, StatsBase, Statistics, Dates
import Distributions
import REPL # stdlib, needed for Term

# for administrators to update Metadata.toml:
export @update, check_registry

# from loading.jl:
export load, @load, @iload, @loadcode, info

# from model_search:
export models, localmodels, matching

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

include("utilities.jl")

Handle = NamedTuple{(:name, :pkg), Tuple{String,String}}
(::Type{Handle})(name,string) = NamedTuple{(:name, :pkg)}((name, string))

# load utilities for reading model metadata from file:
include("metadata.jl")

# read in the metadata:
metadata_file = joinpath(srcdir, "registry", "Metadata.toml")
const INFO_GIVEN_HANDLE = info_given_handle(metadata_file)
const PKGS_GIVEN_NAME = pkgs_given_name(INFO_GIVEN_HANDLE)
const AMBIGUOUS_NAMES = ambiguous_names(INFO_GIVEN_HANDLE)
const NAMES = model_names(INFO_GIVEN_HANDLE)
const MODEL_TRAITS_IN_REGISTRY = model_traits_in_registry(INFO_GIVEN_HANDLE)

# model search and registry code:
include("model_search.jl")
include("loading.jl")
include("registry/src/info_dict.jl")
include("registry/src/Registry.jl")
include("registry/src/check_registry.jl")
import .Registry.@update

# load built-in models:
include("builtins/Constant.jl")
include("builtins/Transformers.jl")
include("builtins/ThresholdPredictors.jl")

end # module
