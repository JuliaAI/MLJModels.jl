## METHODS TO GENERATE METADATA AND WRITE TO ARCHIVE

function finaltypes(T::Type)
    s = InteractiveUtils.subtypes(T)
    if isempty(s)
        return [T, ]
    else
        return reduce(vcat, [finaltypes(S) for S in s])
    end
end

const project_toml = joinpath(srcdir, "../Project.toml")
const PACKAGES = map(Symbol,
                     keys(TOML.parsefile(project_toml)["deps"])|>collect)
push!(PACKAGES, :MLJModels)
filter!(PACKAGES) do pkg
    !(pkg in (:InteractiveUtils, :Pkg, :MLJModelInterface, :MLJTestIntegration))
end

const package_import_commands =  [:(import $pkg) for pkg in PACKAGES]

macro update()
    mod = __module__
    _update(mod, false)
end

"""
    MLJModels.@update

Update the [MLJ Model
Registry](https://github.com/JuliaAI/MLJModels.jl/tree/dev/src/registry)
by loading all packages in the registry Project.toml file and
searching for types implementing the MLJ model interface.

*For MLJ administrators only.*

To register all the models in GreatNewPackage with MLJ:

- In the dev branch of a clone of the dev branch of MLJModels, change
  to the `/src/registry/` directory and, in the latest version of
  julia, activate the environment specified by the Project.toml there,
  after checking the [compat] conditions there are up to date. It is
  suggested you do not use `Revise`.

- Add `GreatNewPackage` to the environment.

- In some environment to which your MLJModels clone has been added
  (using `Pkg.dev`) execute `using MLJModels; MLJModels.@update`. This updates
  `src/registry/Metadata.toml` and `src/registry/Models.toml` (the
  latter is generated for convenience and not used by MLJ).

- Quit your REPL session and make a trivial commit to your MLJModels
  branch to force pre-compilation in a new julia session when you run
  `using MLJModels`. (For technical reasons the registry is not loaded
  in `__init__()`, so without pre-compiliation the new ]registry is not
  available.)

- Test that the interfaces load properly with
  `MLJModels.check_registry()`. (CI will fail on dev -> master if
  this test fails.)

- Push your changes to an appropriate branch of MLJModels to make
  the updated metadata available to users of the next MLJModels tagged
  release.


"""
macro update(ex)
    mod = __module__
    test_env_only = eval(ex)
    test_env_only isa Bool || "b in @update(b) must be Bool. "
    _update(mod, test_env_only)
end

function _update(mod, test_env_only)

    test_env_only && @info "Testing registry environment only. "

    program1 = quote
        @info "Packages to be searched for model implementations:"
        for pkg in $PACKAGES
            println(pkg)
        end
        using Pkg
        Pkg.activate($environment_path)
        @info "resolving registry environment..."
        Pkg.resolve()
    end

    program2 = quote

        @info "Instantiating registry environment..."
        Pkg.instantiate()

        @info "Loading registered packages..."
        import MLJModels
        using Pkg.TOML

        # import the packages
        $(Registry.package_import_commands...)

        @info "Generating model metadata..."

        modeltypes =
            MLJModels.Registry.finaltypes(MLJModels.Model)
        filter!(modeltypes) do T
            !isabstracttype(T) && !MLJModels.MLJModelInterface.is_wrapper(T)
        end

        # generate and write to file the model metadata:
        api_packages = string.(MLJModels.Registry.PACKAGES)
        meta_given_package = Dict()

        for M in modeltypes
            _info = MLJModels.info_dict(M)
            pkg = _info[:package_name]
            path = _info[:load_path]
            api_pkg = split(path, '.') |> first
            pkg in ["unknown",] &&
                @warn "$M `package_name` or `load_path` is \"unknown\")"
            modelname = _info[:name]
            api_pkg in api_packages ||
                error("Bad `load_path` trait for $M: "*
                      "$api_pkg not a registered package. ")
            haskey(meta_given_package, pkg) ||
                (meta_given_package[pkg] = Dict())
            haskey(meta_given_package, modelname) &&
                error("Encountered multiple model names for "*
                      "`package_name=$pkg`")
            meta_given_package[pkg][modelname] = _info
                println(M, "\u2714 ")
        end
        print("\r")

        open(joinpath(MLJModels.Registry.srcdir, "../Metadata.toml"), "w") do file
            TOML.print(file, MLJModels.encode_dic(meta_given_package))
        end

        # generate and write to file list of models for each package:
        models_given_pkg = Dict()
        for pkg in keys(meta_given_package)
            models_given_pkg[pkg] = collect(keys(meta_given_package[pkg]))
        end
        open(joinpath(MLJModels.Registry.srcdir, "../Models.toml"), "w") do file
            TOML.print(file, models_given_pkg)
        end

        :(println("Local Metadata.toml updated."))

    end

    mod.eval(program1)
    test_env_only || mod.eval(program2)

    println("\n You can check the registry by running "*
            "`MLJModels.check_registry() but may need to force "*
            "recompilation of MLJModels.\n\n"*
            "You can safely ignore \"conflicting import\" warnings. ")

    true
end
