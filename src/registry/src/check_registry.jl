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

    quote
        using MLJTestIntegration
        fails, _ = MLJTestIntegration.test(
            MLJModels.models();
            level=1,
            mod=$mod,
            verbosity=$verbosity
        )
        fails
    end |> mod.eval

end
