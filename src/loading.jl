## FUNTIONS TO LOAD MODEL IMPLEMENTATION CODE

"""
    load(name::String; pkg=nothing, modl=Main, verbosity=1, allow_ambiguous=false)

Load the model implementation code for the model type with specified
`name` into the module `modl`, specifying `pkg` if necesssary, to
resolve duplicate names.

    load(proxy; pkg=nothing, modl=Main, verbosity=1)

In the case that `proxy` is a return value of `traits` (ie, has the
form `(name = ..., package_name = ..., etc)`) this is equivalent to
`load(name=proxy.name, pkg=proxy.package_name)`.

If `allow_ambiguous=true` then multiple models with the same name can be imported, otherwise a warning is raised and the previously available model is kept.

See also [`@load`](@ref)

### Examples

    load("ConstantClassifier")
    load(localmodels()[1])

See also [`@load`](@ref)

"""
function load(proxy::ModelProxy; modl=Main, verbosity=0, allow_ambiguous=false)
    # get name, package and load path:
    name = proxy.name
    pkg = proxy.package_name
    handle = (name=name, pkg=pkg)

    path = INFO_GIVEN_HANDLE[handle][:load_path]
    path_components = split(path, '.')

    # decide what to print
    toprint = verbosity > 0

    # return if model is already loaded unless allow_ambiguous
    localnames = map(p->p.name, localmodels(modl=modl))
    if !allow_ambiguous && name ∈ localnames
        @info "A model type \"$name\" is already loaded. \n"*
        "No new code loaded. "
        return
    end

    verbosity > 1 && @info "Loading into module \"$modl\": "

    # the package providing the implementation of the MLJ model
    # interface could be different from `pkg`, which is the package
    # name exposed to the user (and where the core alogrithms live,
    # but that this not actually relevant here).
    api_pkg = path_components[1]
    
    # if needed, put MLJModels in the calling module's namespace:
    if api_pkg == "MLJModels"
        toprint && print("import MLJModels ")
        # the latter pass exists as a fallback, and shouldn't happen
        # for end users, but we need that for testing, etc
        load_ex =
            isdefined(modl, :MLJ) ? :(import MLJ.MLJModels) : :(import MLJModels)
        modl.eval(load_ex)
        toprint && println('\u2714')
    end

    # load `api_pkg`, unless this is MLJModels, in which case load
    # `pkg` to trigger lazy-loading of implementation code:
    pkg_ex = api_pkg == "MLJModels" ? Symbol(pkg) : Symbol(api_pkg)
    toprint && print("import $pkg_ex ")
    modl.eval(:(import $pkg_ex))

    toprint && println('\u2714')

    # load the model:
    # the latter pass exists as a fallback, and shouldn't happen for end users,
    # but we need that for testing, etc
    load_str = (api_pkg == "MLJModels" && isdefined(modl, :MLJ)) ?
        "import MLJ.$path" : "import $path"
    load_ex = Meta.parse(load_str)
    toprint && print(string(load_ex, " "))
    modl.eval(load_ex)
    toprint && println('\u2714')

    nothing
end

load(name::String; pkg=nothing, kwargs...) =
    load(info(name; pkg=pkg); kwargs...)

"""
    @load name pkg=nothing verbosity=0

Load the model implementation code for the model with specified `name`
into the calling module, provided `pkg` is specified in the case of
duplicate names.

### Examples

    @load DecisionTreeeRegressor
    @load PCA verbosity=1
    @load SVC pkg=LIBSVM

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

    esc(quote
            try
                $name_ex()
            catch
                try # hack for baremodules that have imported Base.eval:
                    $name_ex = Base.$name_ex
                    $name_ex()
                catch
                    @warn "Code is loaded but no instance returned. "
                    nothing
                end
            end
        end)
end
