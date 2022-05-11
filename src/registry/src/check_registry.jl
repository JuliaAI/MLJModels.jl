"""
    registry_project()

Return, as a `Vector{String}`, the lines of the Project.toml used to
generate MLJ Model Registry (aka, model metadata). This Project.toml
file lists as dependencies all packages that provide registered
models.

"""
registry_project() = REGISTRY_PROJECT[]

"""
    activate_registry_environment()

Activate a temporary clone of the environment that is used to generate
the MLJ Model Registry (aka, model metadata). This environment
includes all packages providing registered models.

To instantiate the environment (for which no Manifest.toml will exist)
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

function check_registry()

    basedir = Registry.environment_path
    mljmodelsdir = joinpath(basedir, "..", "..", ".")
    Pkg.activate(basedir)
    Pkg.develop(PackageSpec(path=mljmodelsdir))
    Pkg.instantiate()
    Pkg.precompile()

    # Read Metadata.toml
    dict = TOML.parsefile(joinpath(basedir, "Metadata.toml"))

    problems = String[]
    for (package, model_dict) in dict
        for (model, meta) in model_dict
            # check if new entry or changed entry, otherwise don't test
            key = "$package.$model"
            program = quote
                @load $model pkg=$package verbosity=-1
            end
            try
                eval(program)
                # add/refresh entry
                print(rpad("Entry for $key was loaded properly ✓", 79)*"\r")
            catch ex
                push!(problems, string(key))
                @error "⚠ there was an issue trying to load $key" exception=ex
            end
        end
    end
    return problems
end
