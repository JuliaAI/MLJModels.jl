## FUNTIONS TO LOAD MODEL IMPLEMENTATION CODE

"""
    load(name::String; pkg=nothing, modl=Main, verbosity=1)
            
Load the model implementation code for the model type with specified
`name` into the module `modl`, specifying `pkg` if necesssary, to
resolve duplicate names. 

    load(proxy; pkg=nothing, modl=Main, verbosity=1)

In the case that `proxy` is a return value of `traits` (ie, has the
form `(name = ..., package_name = ..., etc)`) this is equivalent to
`load(name=proxy.name, pkg=proxy.package_name)`. 

See also [`@load`](@ref)

### Examples

    load("ConstantClassifier")
    load(localmodels()[1])

See also [`@load`](@ref)

"""
function load(proxy::ModelProxy; modl=Main, verbosity=0)
    # get name, package and load path:
    name = proxy.name
    pkg = proxy.package_name
    handle = (name=name, pkg=pkg)
    
    path = INFO_GIVEN_HANDLE[handle][:load_path]
    path_components = split(path, '.')

    # decide what to print
    toprint = verbosity > 0

    # return if model is already loaded
    localnames = map(p->p.name, localmodels(modl=modl))
    if name ∈ localnames
        @info "A model type \"$name\" is already loaded. \n"*
        "No new code loaded. "
        return
    end

    verbosity > 1 && @info "Loading into module \"$modl\": "

    # if needed, put MLJModels in the calling module's namespace:
    if path_components[1] == "MLJModels"
        toprint && print("import MLJModels ")
        modl.eval(:(import MLJModels))
        toprint && println('\u2714')
    end

    # load the package (triggering lazy-load of implementation code if
    # this is in MLJModels):
    pkg_ex = Symbol(pkg)
    toprint && print("import $pkg_ex ")
    modl.eval(:(import $pkg_ex))
    
    toprint && println('\u2714')

    # load the model:
    load_ex = Meta.parse("import $path")
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
