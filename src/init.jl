const REGISTRY_PATH = @path joinpath(@__DIR__, "registry", "Project.toml")
function __init__()
    project = open(REGISTRY_PATH) do io
        readlines(io)
    end
    global REGISTRY_PROJECT = Ref{Vector{String}}(project)
end
