function __init__()
    project = open(@path(joinpath(@__DIR__, "registry", "Project.toml"))) do io
        readlines(io)
    end
    global REGISTRY_PROJECT = Ref{Vector{String}}(project)
end
