## FUNTIONS TO LOAD MODEL IMPLEMENTATION CODE

# helper:


"""
    load(name::String; pkg=nothing, modl=Main, verbosity=1, name=nothing)

Load the model implementation code for the model type with specified
`name` (first argument) into the module `modl`, specifying `pkg` if
necesssary to resolve duplicate names. Return an instance of the model type
with default hyper-parameters.

To bind the name of the model type to a name different from the first
argument, specify `name=...`.

    load(proxy; kwargs...1)

In the case that `proxy` is a return value of `traits` (ie, has the
form `(name = ..., package_name = ..., etc)`) this is equivalent to
`load(name=proxy.name, pkg=proxy.package_name; kwargs...)`.

See also [`@load`](@ref)

### Examples

    load("RidgeRegressor",
         pkg="MLJLinearModels",
         modl=MyPackage,
         name="NewRidge",
         verbosity=0)

    load(localmodels()[1])

See also [`@load`](@ref)

"""
function load(proxy::ModelProxy; modl=Main, verbosity=1, name=nothing)

    # decide what to print
    toprint = verbosity > 0

    # get name, package and load path:
    model_name = proxy.name
    pkg = proxy.package_name
    handle = (name=model_name, pkg=pkg)

    # see if the model has already been loaded:
    type_already_loaded = handle in map(localmodels(modl=modl)) do m
        (name=m.name, pkg=m.package_name)
    end

    if type_already_loaded && name == nothing
        toprint && @info "Model code for $model_name already loaded"
        # return an instance of the type:
        for M in localmodeltypes(modl)
             i = info(M)
            if i.name == model_name && i.package_name == pkg
                return M()
            end
        end
    end

    # from now on work with symbols not strings:
    path = INFO_GIVEN_HANDLE[handle][:load_path]
    path_components = Symbol.(split(path, '.') )
    model_name = Symbol(model_name)
    pkg = Symbol(pkg)

    verbosity > 0 && @info "Loading into module \"$modl\": "

    # create `name`, the actual name (symbol) to be bound to the
    # new type in the global namespace of `modl`. Note two packages
    # might provide models with the same `model_name`.
    if name === nothing
        name = MLJBase.available_name(modl, Symbol(model_name))
        name == model_name || verbosity < 0 ||
            @warn "New model type being bound to "*
            "`$name` to avoid conflict with an existing name. "
    else
        name = Symbol(name)
    end

    # the package providing the implementation of the MLJ model
    # interface could be different from `pkg`, which is the package
    # name exposed to the user (and where the core alogrithms live,
    # but that this not actually relevant here).
    api_pkg = path_components[1]

    # if needed, put MLJModels in the calling module's namespace:
    if api_pkg == :MLJModels
        toprint && print("import MLJModels ")
        # the latter pass exists as a fallback, and shouldn't happen
        # for end users, but we need that for testing, etc
        load_ex =
            isdefined(modl, :MLJ) ? :(import MLJ.MLJModels) :
            :(import MLJModels)
        modl.eval(load_ex)
        toprint && println('\u2714')
    end

    # load `api_pkg`, unless this is MLJModels, in which case load
    # `pkg` to trigger lazy-loading of implementation code:
    pkg_ex = api_pkg == :MLJModels ? pkg : api_pkg
    toprint && print("import $pkg_ex ")
    modl.eval(:(import $pkg_ex))

    toprint && println('\u2714')

    # load the model:
    # the latter pass exists as a fallback, and shouldn't happen for
    # end users, but we need that for testing, etc
    if api_pkg == "MLJModels" && isdefined(modl, :MLJ)
        pushfirst!(path_components, :MLJ)
    end
    root_components = path_components[1:(end - 1)]
    root_str = join(string.(root_components), '.')
    import_ex = Expr(:import, Expr(:(.), root_components...))
    path_str = join(string.(path_components), '.')
    path_ex = path_str |> Meta.parse
    program =
        quote
            $import_ex
            const $name = $path_ex
        end

    toprint && print("import ", root_str, " ")
    modl.eval(program)
    toprint && println('\u2714')

    return modl.eval(:($name()))

end

load(name::String; pkg=nothing, kwargs...) =
    load(info(name; pkg=pkg); kwargs...)

"""
    @load name pkg=nothing verbosity=0 name=nothing

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
    name_ = string(name_ex)

    # parse kwargs:
    warning = "Invalid @load syntax.\n "*
    "Sample usage: @load PCA pkg=\"MultivariateStats\" verbosity=1"
    for ex in kw_exs
        ex.head == :(=) || throw(ArgumentError(warning))
        variable_ex = ex.args[1]
        value_ex = ex.args[2]
        if variable_ex == :pkg
            pkg = string(value_ex)
        elseif variable_ex == :verbosity
            verbosity = value_ex
        else
            throw(ArgumentError(warning))
        end
    end
    (@isdefined pkg) || (pkg = nothing)
    (@isdefined verbosity) || (verbosity = 0)

    # get rid brackets in name_, as in
    # "(MLJModels.Clustering).KMedoids":
    name = filter(name_) do c !(c in ['(',')']) end

    load(name, modl=__module__, pkg=pkg, verbosity=verbosity)

end
