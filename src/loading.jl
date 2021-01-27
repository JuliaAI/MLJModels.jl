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
    # can be removed once MLJModel #331 is resolved:
    if pkg == :NearestNeighbors
        doprint && print("import NearestNeighbors")
        try
            modl.eval(:(import MLJModels))
        catch
            try
                modl.eval(:(import MLJ.MLJModels))
            catch
                error("Problem putting MLJModels into scope. ")
            end
        end
        modl.eval(:(import NearestNeighbors))
        doprint && println(" \u2714")
    else
        doprint && print("import $api_pkg")
        modl.eval(:(import $api_pkg))
        doprint && println(" \u2714")
    end
end

function _eval(modl, path::String)
    ex = Meta.parse(path)
    modl.eval(ex)
end

function _eval_and_bind(modl, path::String, name::Symbol, doprint)
    value = _eval(modl, path)
    doprint && print("const $(string(name)) = $value")
    modl.eval(:(const $name = $value))
    doprint && println(" \u2714")
end

const available_name = MLJBase.available_name


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
    @load ModelName pkg=nothing verbosity=0 name=nothing scope=:global install=false

Load the model implementation code for the model named in the first
argument into the calling module, specfying `pkg` in the case of
duplicate names. Return a model instance with default
hyper-parameters.

To bind the new model type to a name different from the first
argument, specify `name=...`.


### Examples

    @load DecisionTreeeRegressor
    @load PCA verbosity=1
    @load SVC pkg=LIBSVM name=MyOtherSVC

See also [`load`](@ref)

"""
macro load(name_ex, kw_exs...)
    _load(__module__, name_ex, kw_exs...)
end

"""
    @iload ModelName

Interactive alternative to @load. See [`@load`](@ref)

"""
macro iload(name_ex, kw_exs...)
    _load(__module__, name_ex, kw_exs...; interactive=true)
end

macro loadcode(name_ex, kw_exs...)
    _load(__module__, name_ex, kw_exs...; return_instance=false)
end

# builds the program to be evaluated by the @load macro:
function _load(modl, name_ex, kw_exs...;
               interactive=false,
               return_instance=true)

    # initialize:
    program = quote end

    # fallbacks:
    pkg = nothing
    verbosity = nothing
    name_specified = nothing
    scope = :global
    install_pkgs = false

    # parse name_ex:
    name_ = string(name_ex)
    # get rid of parentheses in `name_`, as in
    # "(MLJModels.Clustering).KMedoids":
    name = filter(name_) do c !(c in ['(',')']) end

    # parse kwargs:
    warning = "Invalid @load syntax.\n "*
        "Sample usage: @load PCA pkg=MultivariateStats verbosity=0 install=true"
    for ex in kw_exs
        ex.head == :(=) || throw(ArgumentError(warning))
        variable_ex = ex.args[1]
        value_ex = ex.args[2]
        if variable_ex == :pkg
            pkg = string(value_ex)
        elseif variable_ex == :verbosity
            verbosity = value_ex
        elseif variable_ex == :name
            name_specified = string(value_ex)
        elseif variable_ex == :scope
            scope = value_ex
        elseif variable_ex == :install_pkgs
            install_pkgs = value_ex
        elseif variable_ex == :install
            install_pkgs = value_ex
        else
            throw(ArgumentError(warning))
        end
    end

    if scope == :(:global)
        scope = :global
    end

    if scope == :(:local)
        scope = :local
    end

    if !( scope in [:global, :local] )
        throw(ArgumentError("Invalid value for `scope`: `$(scope)`. "*
                            "Valid values are `:global` and `:local`"))
    end

    if interactive && scope == :local
        throw(ArgumentError("Cannot use `@iload` with `scope=:local`. "))
    end

    if scope == :local && name === nothing
        throw(ArgumentError("When specifying `scope=:local`, you must also "*
                            "specify a name to be bound to the new model "*
                            "type, as in `name=NewModelType`. "))
    end

    if verbosity == nothing
        verbosity = scope == :global ? 1 : 0
    end

    # are we printing stuff to stdout?
    doprint = verbosity > 0

    # next expression defines run-time variables `path`, `path_ex`,
    # `pkg_str`, `pkg`, `type_already_loaded`:
    ex  = quote
        $doprint && @info "For silent loading, specify `verbosity=0`. "

        proxy = MLJModels.info($name; pkg=$pkg, interactive=$interactive)
        handle = MLJModels.Handle(proxy.name, proxy.package_name)
        dic = MLJModels.INFO_GIVEN_HANDLE[handle]
        path, pkg_str = dic[:load_path], dic[:package_name]
        path_ex = path |> Meta.parse

        # see if the model type is already in top-level scope:
        type_already_loaded =
            handle in map(MLJModels.localmodels(modl=$modl, toplevel=true)) do m
                (name=m.name, pkg=m.package_name)
            end
        if type_already_loaded && $doprint
            @info "Model code for `$($name)` (from `$pkg_str`) "*
                "already loaded. "
            $name_specified === nothing || $verbosity < 0 ||
                @warn "Ignoring specification `name=$($name_specified)`. "
        end

        pkg = Symbol(pkg_str)
    end
    push!(program.args, ex)

    # Note. In interactive mode the name to be bound to the new type,
    # `new_name`, is determined at run-time. Otherwise, they are
    # determined at compile time. We start by auto-generating the name
    # (different from `name` this is already name of object in
    # scope).:

    # create `candidate` for `new_name`:
    if name_specified === nothing
        candidate = available_name(modl, Symbol(name))
    else
        candidate = Symbol(name_specified)
    end
    candidate_str = string(candidate)

    if !interactive
        new_name = candidate
        if verbosity > -1 &&
            new_name != Symbol(name) && name_specified === nothing
            str = string(new_name)
            ex = quote
                type_already_loaded ||
                    @warn "New model type being bound to name `$($str)` "*
                    "to avoid conflict "*
                    "with an existing name. "
            end
            push!(program.args, ex)
        end
    end

    if interactive
        ex = quote
            if !type_already_loaded
                new_name = Symbol($candidate_str)
                if new_name != Symbol($name)
                    invalid = true
                    while invalid
                        choice = request(
                            "Model type name conflicts with an existing "*
                            "name in scope. \nWhat do you want to do? ",
                            ## choices:
                            "Accept autogenerated name, `$($candidate_str)`.",
                            "Enter different type name.")
                        if choice == 2
                            print("New name: ")
                            new_name = Symbol(readline())
                        end
                        invalid =
                            MLJModels.available_name($modl, new_name)!=new_name
                    end
                end
            end
            end
        push!(program.args, ex)
    end

    ex = quote
        if !type_already_loaded
            path_components = Symbol.(split(path, '.') )

            # get pkg containing implementation of model API implementation:
            api_pkg = path_components[1]
            api_pkg_str = string(api_pkg)

            if $install_pkgs || $interactive
                try
                    MLJModels._import($modl, api_pkg, pkg, false)
                catch
                    if $interactive
                        MLJModels.request(
                            "The package providing an interface to `$($name)` "*
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
        end
    end
    push!(program.args, ex)

    # next expression binds new model type to `new_name`:
    ex = if !interactive
        if scope == :local
            # we cannot use the `const` keyword in front of local variables:
            quote
                if !type_already_loaded
                    $(esc(new_name)) = MLJModels._eval($modl, path)
                end
            end
        else
            new_name_str = string(new_name)
            quote
                if !type_already_loaded
                    value = MLJModels._eval($modl, path)
                    $doprint && print("const $($new_name_str) = $value")
                    const $(esc(new_name)) = value
                    $doprint && println(" \u2714")
                end
            end
        end
    else
        quote
            if !type_already_loaded
                MLJModels._eval_and_bind($modl, path, new_name, $doprint)
            end
        end
    end
    push!(program.args, ex)

    ex = if return_instance
        quote
            model_types = MLJModels.localmodeltypes($modl);
            @show "##########"
            idx = findfirst(model_types) do M
                i = MLJModels.info(M)
                i.name == $name && i.package_name == pkg_str
            end
            M = model_types[idx]
            $doprint && print("($M)()")
            $doprint && println(" \u2714")
            M()
        end
    else
        :nothing
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
