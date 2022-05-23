"""
    registry_project()

Experimental, private method.

Return, as a `Vector{String}`, the lines of the Project.toml used to
generate MLJ Model Registry (aka, model metadata). This Project.toml
file lists as dependencies all packages that provide registered
models.

"""
registry_project() = MLJModels.REGISTRY_PROJECT[]

"""
    activate_registry_project()
    activate_registry_project(path)

Experimental, private method.

In the first case, activate a temporary environment using a copy of
the [MLJ Project
Registry](https://github.com/JuliaAI/MLJModels.jl/tree/dev/src/registry)
Project.toml file.  This environment will include all packages
providing registered models.

In the second case, create the environment at the specified `path`.

To instantiate the environment (for which no Manifest.toml will exist)
run `using Pkg; Pkg.instantiate()`.

"""
function activate_registry_project(projectdir=mktempdir(; cleanup=false))
    filename, stream = mktemp(projectdir)
    for line in registry_project()
        write(stream, line*"\n")
    end
    close(stream)
    project_filename = joinpath(first(splitdir(filename)), "Project.toml")
    cp(filename, project_filename)
    Pkg.activate(projectdir)
    return nothing
end
