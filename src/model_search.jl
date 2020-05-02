## FUNCTIONS TO INSPECT METADATA OF REGISTERED MODELS AND TO
## FACILITATE MODEL SEARCH

property_names = sort(MODEL_TRAITS)
alpha = [:name, :package_name, :is_supervised]
omega = [:input_scitype, :target_scitype, :output_scitype]
both = vcat(alpha, omega)
filter!(!in(both), property_names)
prepend!(property_names, alpha)
append!(property_names, omega)
const PROPERTY_NAMES = Tuple(property_names)

ModelProxy = NamedTuple{PROPERTY_NAMES}

function Base.isless(p1::ModelProxy, p2::ModelProxy)
    if isless(p1.name, p2.name)
        return true
    elseif p1.name == p2.name
        return isless(p1.package_name, p2.package_name)
    else
        return false
    end
end

import MLJBase.==
function ==(m1::ModelProxy, m2::ModelProxy)
    m1.name == m2.name && m1.package_name == m2.package_name
    # tests = map(keys(m1)) do k
    #     v1 = getproperty(m1, k)
    #     v2 = getproperty(m2, k)
    #     if k isa AbstractVector
    #         Set(v1) == Set(v2)
    #     else
    #         v1 == v2
    #     end
    # end
    # return all(tests)
end

Base.show(stream::IO, p::ModelProxy) =
    print(stream, "(name = $(p.name), package_name = $(p.package_name), "*
          "... )")

function Base.show(stream::IO, ::MIME"text/plain", p::ModelProxy)
    printstyled(IOContext(stream, :color=> MLJBase.SHOW_COLOR),
                    p.docstring, bold=false, color=:magenta)
    println(stream)
    MLJBase.fancy_nt(stream, p)
end

# returns named tuple version of the dictionary i=info_dict(SomeModelType):
function info_as_named_tuple(i)
    property_values = Tuple(i[property] for property in PROPERTY_NAMES)
    return NamedTuple{PROPERTY_NAMES}(property_values)
end

MLJBase.info(handle::Handle) =
    info_as_named_tuple(INFO_GIVEN_HANDLE[handle])

"""
    info(name::String; pkg=nothing)

Returns the metadata for the registered model type with specified
`name`. The key-word argument `pkg` is required in the case of
duplicate names.

"""
function MLJBase.info(name::String; pkg=nothing)
    name in NAMES ||
        throw(ArgumentError("There is no model named \"$name\" in "*
                            "the registry. \n Run `models()` to view all "*
                            "registered models."))
    # get the handle:
    if pkg == nothing
        handle  = Handle(name)
        if ismissing(handle.pkg)
            pkgs = PKGS_GIVEN_NAME[name]
            message = "Ambiguous model name. Use pkg=... .\n"*
            "The model $name is provided by these packages:\n $pkgs.\n"
            throw(ArgumentError(message))
        end
    else
        handle = Handle(name, pkg)
        haskey(INFO_GIVEN_HANDLE, handle) ||
            throw(ArgumentError("The package \"$pkg\" does not appear to "*
                                "provide the model \"$name\". \n"*
                                "Use models() to list all models. "))
    end
    return info(handle)

end


"""
   info(model::Model)

Return the traits associated with the specified `model`. Equivalent to
`info(name; pkg=pkg)` where `name::String` is the name of the model type, and
`pkg::String` the name of the package containing it.

"""
MLJBase.info(M::Type{<:MLJBase.Model}) =
    info_as_named_tuple(MLJBase.info_dict(M))
MLJBase.info(model::MLJBase.Model) = info(typeof(model))

"""
    models()

List all models in the MLJ registry. Here and below *model* means the
registry metadata entry for a genuine model type (a proxy for types
whose defining code may not be loaded).

    models(conditions...)

List all models satisifying the specified `conditions`. A *condition*
is any `Bool`-valued function on models.

Excluded in the listings are the built-in model-wraps `EnsembleModel`,
`TunedModel`, and `IteratedModel`.

### Example

If

    task(model) = model.is_supervised && model.is_probabilistic

then `models(task)` lists all supervised models making probabilistic
predictions.

See also: [`localmodels`](@ref).

"""
function MLJBase.models(conditions...)
    unsorted = filter(info.(keys(INFO_GIVEN_HANDLE))) do model
        all(c(model) for c in conditions)
    end
    return sort!(unsorted)
end

"""
    models(regex::Regex)

List all models whole `name` or `docstring` matches a given `regex`.
"""
function MLJBase.models(regex::Regex)
    unsorted = filter(info.(keys(INFO_GIVEN_HANDLE))) do model
        occursin(regex, model.name) || occursin(regex, model.docstring)
    end
    return sort!(unsorted)
end

# function models(task::MLJBase.SupervisedTask)
#     ret = Dict{String, Any}()
#     function condition(t)
#         return t.is_supervised &&
#             task.target_scitype <: t.target_scitype &&
#             task.input_scitype <: t.input_scitype &&
#             task.is_probabilistic == t.is_probabilistic
#     end
#     return models(condition)
# end

# function models(task::UnsupervisedTask)
#     ret = Dict{String, Any}()
#     function condition(handle)
#         t = info(handle)
#         return task.input_scitype <: t.input_scitype
#     end
#     return models(condition)
# end

"""
    localmodels(; modl=Main)
    localmodels(conditions...; modl=Main)


List all models whose names are in the namespace of the specified
module `modl`, or meeting the `conditions`, if specified. Here a
*condition* is a `Bool`-valued function on models.

See also [`models`](@ref)

"""
function localmodels(args...; modl=Main)
    modeltypes = localmodeltypes(modl)
    names = map(modeltypes) do M
        MLJBase.name(M)
    end
    return filter(models(args...)) do handle
        handle.name in names
    end
end
