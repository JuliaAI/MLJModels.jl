###############################################
# FUNCTIONS TO LOAD MODEL IMPLEMENTATION CODE #
###############################################


## HELPERS

function _append!(program, ex, doprint::Bool, tick_early::Bool)
    str = string(ex)
    doprint && push!(program.args, :(print($str)))
    doprint && !tick_early && push!(program.args, :(println(" \u2714")))
    push!(program.args, ex)
    tick_early && doprint && push!(program.args, :(println(" \u2714")))
    return program
end

function _import(modl, api_pkg, pkg, doprint)
    doprint && print("import $api_pkg")
    modl.eval(:(import $api_pkg))
    doprint && println(" \u2714")
end

function _eval(modl, path::Union{Expr,Symbol})
    modl.eval(path)
end


## OVERLOADING load_path

"""
    load_path(model::String, pkg=nothing)

Return the load path for model type with name `model`, specifying the
package name `pkg` to resolve name conflicts if necessary.

    load_path(model)

Return the load path of a `model` instance or type. Usually requires
necessary model code to have been separately loaded. Supply a string
as above if code is not loaded.

"""
function MLJModelInterface.load_path(proxy::ModelProxy)
    handle = (name=proxy.name, pkg=proxy.package_name)
    return INFO_GIVEN_HANDLE[handle][:load_path]
end
function MLJModelInterface.load_path(name::String; pkg=nothing)
    proxy = info(name; pkg=pkg)
    return load_path(proxy)
end

# to also get pkg, which could be different from glue code pkg
# appearing in load_path:
function load_path_and_pkg(name::String; pkg=nothing, interactive=false)
    proxy = info(name; pkg=pkg, interactive=interactive)
    handle = (name=proxy.name, pkg=proxy.package_name)
    _info = INFO_GIVEN_HANDLE[handle]

    return _info[:load_path], _info[:package_name]
end


## THE CODE LOADING MACROS

"""
    @load ModelName pkg=nothing verbosity=0 add=false

Import the model type the model named in the first argument into the
calling module, specfying `pkg` in the case of an ambiguous name (to
packages providing a model type with the same name). Returns the model
type.

**Warning** In older versions of MLJ/MLJModels, `@load` returned an
*instance* instead.

To automatically add required interface packages to the current
environment, specify `add=true`. For interactive loading, use
`@iload` instead.

### Examples

    Tree = @load DecisionTreeRegressor
    tree = Tree()
    tree2 = Tree(min_samples_split=6)

    SVM = @load SVC pkg=LIBSVM
    svm = SVM()

See also [`@iload`](@ref)

"""
macro load(name_ex, kw_exs...)
    _load(__module__, name_ex, kw_exs...)
end

"""
    @iload ModelName

Interactive alternative to @load. Provides user with an optioin to
install (add) the required interface package to the current
environment, and to choose the relevant model-providing package in
ambiguous cases.  See [`@load`](@ref)

"""
macro iload(name_ex, kw_exs...)
    _load(__module__, name_ex, kw_exs...; interactive=true)
end

# builds the program to be evaluated by the @load/@iload macros:
function _load(modl, name_ex, kw_exs...; interactive=false)

    # initialize:
    program = quote end

    # fallbacks:
    pkg_str = nothing
    verbosity = 1
    scope = :global
    install_pkgs = false

    # parse name_ex:
    name_ = string(name_ex)
    # get rid of parentheses in `name_`, as in
    # "(MLJModels.Clustering).KMedoids":
    name_str = filter(name_) do c !(c in ['(',')']) end
    name = Symbol(name_str)

    # parse kwargs:
    warning = "Invalid @load or @iload  syntax.\n "*
        "Sample usage: @load PCA pkg=MultivariateStats verbosity=0 install=true"
    for ex in kw_exs
        ex.head == :(=) || throw(ArgumentError(warning))
        variable_ex = ex.args[1]
        value_ex = ex.args[2]
        if variable_ex == :pkg
            pkg_str = string(value_ex)
        elseif variable_ex == :verbosity
            verbosity = value_ex
        elseif variable_ex in [:install_pkgs, :install, :add]
            install_pkgs = value_ex
        else
            throw(ArgumentError(warning))
        end
    end

    # are we printing stuff to stdout?
    doprint = verbosity > 0

    # next expression defines "program" variables `path`, `path_ex`,
    # `pkg_str`, `pkg`, `type_already_loaded`:
    ex  = quote
        $doprint && @info "For silent loading, specify `verbosity=0`. "
        proxy = MLJModels.info($name_str;
                               pkg=$pkg_str,
                               interactive=$interactive)
        handle = MLJModels.Handle(proxy.name, proxy.package_name)
        dic = MLJModels.INFO_GIVEN_HANDLE[handle]
        path_str, pkg_str = dic[:load_path], dic[:package_name]
        path = path_str |> Meta.parse
        pkg = Symbol(pkg_str)
    end
    push!(program.args, ex)

    ex = quote
        path_components = Symbol.(split(path_str, '.') )

        # get pkg containing implementation of model API implementation:
        api_pkg = path_components[1]
        api_pkg_str = string(api_pkg)

        if $install_pkgs || $interactive
            try
                MLJModels._import($modl, api_pkg, pkg, false)
            catch
                if $interactive
                    MLJModels.request(
                        "The package providing an interface "*
                        "to `$($name_str)` "*
                        "is not in your "*
                        "current environment.\n"*
                        "What do you want to do? ",
                        # choices:
                        "Install $api_pkg_str in current environment.",
                        "Abort.") == 1 || throw(InterruptException)
                end
                MLJModels._import($modl, :Pkg, :Pkg, false)
                Pkg.add(api_pkg_str)
            end
        end
        MLJModels._import($modl, api_pkg, pkg, $doprint)
        MLJModels._eval($modl, path)
    end
    push!(program.args, ex)

    return program

end


## NO LONGER SUPPORTED

_deperror() = error(
    "The `load` function is no longer supported. "*
    "Use the `@load` macro instead, as in "*
    "`@load RandomForestRegressor pkg = DecisionTree`.\n"*
    "For explicit importing, you can discover a model's "*
    "full load path with the `load_path` function, as in "*
    "`load_path(\"RandomForestRegressor\", pkg=\"DecisionTree\")`. )")

load(proxy::ModelProxy; kwargs...) = _deperror()
load(name::String; kwargs...) = _deperror()
