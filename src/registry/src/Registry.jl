module Registry 

using Pkg
import Pkg.TOML
using MLJModels
import MLJModelInterface
import MLJModelInterface.Model
for T in MLJModelInterface.MODEL_TRAITS
    @eval(import MLJModelInterface.$T)
end

using OrderedCollections
using InteractiveUtils

# TODO: is this import really needed??
# for testing decoding of metadata:
import ScientificTypes: Found, Continuous, Finite, Infinite
import ScientificTypes: OrderedFactor, Count, Multiclass, Binary

export @update, check_registry, activate_registry_project, info_dict

const srcdir = dirname(@__FILE__) # the directory containing this file
const environment_path = joinpath(srcdir, "..")

# for extracting model traits from a loaded model type
include("info_dict.jl")

# for generating and serializing the complete model metadata database
include("update.jl")

# for checking `@load` works for all models in the database
include("check_registry.jl")

# for activating a clone of the registry environment:
include("activate_registry_project.jl")


end # module
