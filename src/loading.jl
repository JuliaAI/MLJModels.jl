## FUNCTIONS TO LOAD MODEL IMPLEMENTATION CODE

### Helpers

function _append!(program, ex, doprint::Bool, tick_early::Bool)
    str = string(ex)
    doprint && push!(program.args, :(print($str)))
    doprint && !tick_early && push!(program.args, :(println(" \u2714")))
    push!(program.args, ex)
    tick_early && doprint && push!(program.args, :(println(" \u2714")))
    return program
end

function _request(query, options...)
    menu = REPL.TerminalMenus.RadioMenu([options...])
    return REPL.TerminalMenus.request(query*"\n", menu) # index of option
end

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
function load_path_and_pkg(name::String; pkg=nothing)
    proxy = info(name; pkg=pkg)
    handle = (name=proxy.name, pkg=proxy.package_name)
    _info = INFO_GIVEN_HANDLE[handle]

    return _info[:load_path], _info[:package_name]
end

## the @load and @iload macros

"""
    @load name pkg=nothing verbosity=0 name=nothing scope=:global install=false

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
    program, instance_prgm = _load(__module__, name_ex, kw_exs...)
    append!(program.args, instance_prgm.args)
    esc(program)
end

macro iload(name_ex, kw_exs...)
    program, instance_prgm = _load(__module__,
                                   name_ex,
                                   kw_exs...;
                                   interactive=true)
    append!(program.args, instance_prgm.args)
    esc(program)
end

macro loadcode(name_ex, kw_exs...)
    program, _ = _load(__module__, name_ex, kw_exs...)
    esc(program)
end

# builds the program to be evaluated by the @load macro:
function _load(modl, name_ex, kw_exs...; interactive=false)

    # initialize:
    program = quote end
    instance_prgm = quote end

    # fallbacks:
    pkg = nothing
    verbosity = 1
    new_name = nothing
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
            new_name = value_ex
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
                            "Valid values are `:global` or `:local`"))
    end

    # are we printing stuff to stdout?
    doprint = verbosity > 0

    doprint && @info "For silent loading, specify `verbosity=0`. "

    # get load path and update pkg (could be `nothing`):
    path, pkg = load_path_and_pkg(name, pkg=pkg)

    # see if the model type has already been loaded:
    handle = (name=name, pkg=pkg)
    type_already_loaded = handle in map(localmodels(modl=modl)) do m
        (name=m.name, pkg=m.package_name)
    end

    # if so, return with program generating an instance:
    if type_already_loaded && new_name == nothing
        doprint && @info "Model code for $name already loaded"
        # return an instance of the type:
        for M in localmodeltypes(modl)
            i = info(M)
            if i.name == name && i.package_name == pkg
                _append!(instance_prgm, :($M()), doprint, false)
                return program, instance_prgm
            end
        end
    end

    # determine `new_name`, to be bound to imported model type (in
    # general, different from `name`):
    if new_name === nothing
        new_name = MLJBase.available_name(modl, Symbol(name))
        if new_name != Symbol(name)
            if interactive
                choice =_request(
                    "New model type will be bound to `$new_name` to "*
                    "avoid conflict with existing type. ",
                    ## choices:
                    "Accept `$new_name`",
                    "Choose new type name")
                if choice == 2
                    print("\nNew name: ")
                    new_name = Symbol(readline())
                elseif choice == -1
                    throw(InterruptException)
                end
            elseif verbosity > -1
                @warn "New model type being bound to "*
                "`$new_name` to avoid conflict with an existing name. "
            end
        end
    else
        new_name = Symbol(new_name)
    end

    path_components = Symbol.(split(path, '.') )

    # get pkg containing implementation of model API implementation:
    api_pkg = path_components[1]
    pkg = Symbol(pkg)

    # if needed, put MLJModels in the calling module's namespace:
    if api_pkg == :MLJModels
        load_ex =
            isdefined(modl, :MLJ) ? :(import MLJ.MLJModels) :
            :(import MLJModels)
        _append!(program, load_ex, doprint, true)
        # TODO: remove next line of code after disintegration of
        # MLJModels (for triggering loading of glue code module):
        api_pkg == pkg || _append!(program, :(import $pkg), doprint, true)
    end

    import_ex = :(import $api_pkg)
    if install_pkgs
        install_ex = quote
            try
                $(import_ex)
            catch
                import Pkg
                Pkg.add($(string(api_pkg)))
            end
        end
        install_ex = MacroTools.prewalk(rmlines, install_ex)
        _append!(program, install_ex, doprint, true)
    end

    path_ex = path |> Meta.parse
    api_pkg == :MLJmodels || _append!(program, import_ex, doprint, true)
    if scope == :local
        # we cannot use the `const` keyword in front of local variables
        _append!(program, :($new_name = $path_ex), doprint, true)
    else
        _append!(program, :(const $new_name = $path_ex), doprint, true)
    end

    instance_ex = doprint ? :($new_name()) : :($new_name();)
    _append!(instance_prgm, instance_ex, doprint, false)

    return program, instance_prgm
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
