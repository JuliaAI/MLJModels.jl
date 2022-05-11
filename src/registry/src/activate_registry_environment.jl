"""
    registry_project()

Return, as a `Vector{String}`, the lines of the Project.toml used to
generate MLJ Model Registry (aka, model metadata). This Project.toml
file lists as dependencies all packages that provide registered
models.

"""
registry_project() = MLJModels.REGISTRY_PROJECT[]

"""
    activate_registry_environment()

Activate a temporary environment that has the same Project.toml file
as that used to generate the MLJ Model Registry (aka, model
metadata). This environment includes all packages providing registered
models.

To instantiate the environment, for which no Manifest.toml will exist,
do `using Pkg; Pkg.instantiate()`.

"""
function activate_registry_environment()
    projectdir = mktempdir()
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
