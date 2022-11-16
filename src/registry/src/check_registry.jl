"""
    MLJModels.check_registry(; mod=Main, verbosity=1)

Check that every model in the [MLJ aodel
Registry](https://github.com/JuliaAI/MLJModels.jl/tree/dev/src/registry)
has a working `load_path` trait by using it to import the model
type. Here `mod` should be the module from which the method is called
- `Main` by default, but `mod=@__MODULE__` should work in general.

Returns a row table detailing the failures, which is empty in the case
of no failures.

"""
function check_registry(; mod=Main, verbosity=1)

    basedir = Registry.environment_path
    mljmodelsdir = joinpath(basedir, "..", "..", ".")
    Pkg.activate(basedir)
    Pkg.develop(PackageSpec(path=mljmodelsdir))
    Pkg.instantiate()

    models = MLJModels.models()
    pkgs = MLJModels.Registry.PACKAGES

    # import packages
    verbosity < 1 || @info "Loading model interface packages."
    program = quote end
    for pkg in pkgs
        line = :(import $pkg)
        push!(program.args, line)
    end
    mod.eval(program)

    verbosity < 1 || @info "Checking model load paths."
    quote
        modeltypes = MLJModels.Registry.finaltypes(MLJModels.Model)
        filter!(modeltypes) do T
            !isabstracttype(T) && !MLJModels.MLJModelInterface.is_wrapper(T)
        end
        using MLJTestInterface
        fails, _ = MLJTestInterface.test(
            modeltypes;
            level=1,
            mod=$mod,
            throw=false,
            verbosity=$verbosity
        )
        fails
    end |> mod.eval

end
