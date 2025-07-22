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

# declare paths to the metadata and associated project file:
const REGISTRY_PROJECT = @path joinpath(@__DIR__, "registry", "Project.toml")
const REGISTRY_METADATA = @path joinpath(@__DIR__, "registry", "Metadata.toml")
Base.include_dependency(REGISTRY_PROJECT)
Base.include_dependency(REGISTRY_METADATA)

# load utilities for reading model metadata from file:
include("metadata.jl")

# read in metadata:
const INFO_GIVEN_HANDLE = info_given_handle(REGISTRY_METADATA)
const PKGS_GIVEN_NAME = pkgs_given_name(INFO_GIVEN_HANDLE)
const AMBIGUOUS_NAMES = ambiguous_names(INFO_GIVEN_HANDLE)
const NAMES = model_names(INFO_GIVEN_HANDLE)
const MODEL_TRAITS_IN_REGISTRY = model_traits_in_registry(INFO_GIVEN_HANDLE)

# include tools to search the model registry:
include("model_search.jl")

# include tools to load model code:
include("loading.jl")

end # module
