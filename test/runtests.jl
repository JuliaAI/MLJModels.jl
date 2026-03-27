import Pkg

using Test, MLJModels, MLJTransforms

test_files = [
    "metadata.jl",
    "model_search.jl",
    "loading.jl",
    joinpath("builtins", "Constant.jl"),
    joinpath("builtins", "ThresholdPredictors.jl"),
]

if parse(Bool, get(ENV, "MLJ_TEST_REGISTRY", "false"))
    push!(test_files, "registry.jl")
else
    @info "Test of the MLJ Registry is being skipped. Set environment variable "*
        "MLJ_TEST_REGISTRY = \"true\" to include them.\n"*
        "The Registry test takes about ten minutes. "
end

files = isempty(ARGS) ? test_files : ARGS

for file in files
    quote
        @testset $file begin
            include($file)
        end
    end |> eval
end
