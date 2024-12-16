module MLJModels

import MLJModelInterface
import MLJModelInterface: Model, metadata_pkg, metadata_model, @mlj_model, info,
    nrows, selectrows, reformat, selectcols, transform, inverse_transform, fitted_params
for T in MLJModelInterface.ABSTRACT_MODEL_SUBTYPES
    @eval(import MLJModelInterface.$T)
end
for T in MLJModelInterface.MODEL_TRAITS
    @eval(import MLJModelInterface.$T)
end

using ScientificTypes
using RelocatableFolders

const MMI = MLJModelInterface

using Pkg, Pkg.TOML, OrderedCollections, Parameters
using Tables, CategoricalArrays, StatsBase, Statistics, Dates
using InteractiveUtils, Markdown
using Combinatorics
import Distributions
import REPL # stdlib, needed for `Term`
import PrettyPrinting
import CategoricalDistributions: UnivariateFinite, UnivariateFiniteArray,
    classes
import StatisticalTraits # for `info`

# from loading.jl:
export load, @load, @iload, @loadcode

# from model_search:
export models, localmodels, matching, doc

# extended in model_search.jl:
export info

# from model/Constant
export ConstantRegressor, ConstantClassifier,
    DeterministicConstantRegressor, DeterministicConstantClassifier

# from model/ThresholdPredictors
export BinaryThresholdPredictor

# from model/Transformers
export UnivariateDiscretizer,
    UnivariateStandardizer, Standardizer, UnivariateBoxCoxTransformer,
    OneHotEncoder, ContinuousEncoder, FillImputer, UnivariateFillImputer,
    UnivariateTimeTypeToContinuous, InteractionTransformer

const srcdir = dirname(@__FILE__) # the directory containing this file
const MMI = MLJModelInterface

if VERSION < v"1.3"
    const nonmissingtype = ScientificTypes.ScientificTypesBase.nonmissing
end

nonmissing = nonmissingtype

include("utilities.jl")

# load built-in models:
include("builtins/Constant.jl")
include("builtins/Transformers.jl")
include("builtins/ThresholdPredictors.jl")

Handle = NamedTuple{(:name, :pkg), Tuple{String,String}}
(::Type{Handle})(name,string) = NamedTuple{(:name, :pkg)}((name, string))

# load utilities for reading model metadata from file:
include("metadata.jl")

# read in the metadata:
metadata_file = joinpath(srcdir, "registry", "Metadata.toml")
Base.include_dependency(metadata_file)
const INFO_GIVEN_HANDLE = info_given_handle(metadata_file)
const PKGS_GIVEN_NAME = pkgs_given_name(INFO_GIVEN_HANDLE)
const AMBIGUOUS_NAMES = ambiguous_names(INFO_GIVEN_HANDLE)
const NAMES = model_names(INFO_GIVEN_HANDLE)
const MODEL_TRAITS_IN_REGISTRY = model_traits_in_registry(INFO_GIVEN_HANDLE)

# model search and registry code:
include("model_search.jl")
include("loading.jl")
include("registry/src/Registry.jl")
using .Registry

# finalize:
include("init.jl")

end # module
