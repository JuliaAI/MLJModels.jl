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

import MLJModelInterface.==
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


## INFO

MLJScientificTypes.info(handle::Handle) =
    info_as_named_tuple(INFO_GIVEN_HANDLE[handle])

"""
    info(name::String; pkg=nothing)

Returns the metadata for the registered model type with specified
`name`. The key-word argument `pkg` is required in the case of
duplicate names.

"""
function MLJScientificTypes.info(name::String; pkg=nothing)
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
MLJScientificTypes.info(M::Type{<:MMI.Model}) =
    info_as_named_tuple(MLJBase.info_dict(M))
MLJScientificTypes.info(model::MMI.Model) = info(typeof(model))


## MATCHING

## SEARCH TOOL

struct Checker{is_supervised,
               supports_weights,
               supports_class_weights,
               input_scitype,
               target_scitype} end

function Base.getproperty(::Checker{is_supervised,
                                    supports_weights,
                                    supports_class_weights,
                                    input_scitype,
                                    target_scitype},
                          field::Symbol) where {is_supervised,
                                                supports_weights,
                                                supports_class_weights,
                                                input_scitype,
                                                target_scitype}
    if field === :is_supervised
        return is_supervised
    elseif field === :supports_weights
        return supports_weights
    elseif field === :supports_class_weights
        return supports_class_weights
    elseif field === :input_scitype
        return input_scitype
    elseif field === :target_scitype
        return target_scitype
    else
        throw(ArgumentError("Unsupported property. "))
    end
end

Base.propertynames(::Checker) =
    (:is_supervised,
     :supports_weights,
     :supports_class_weights,
     :input_scitype,
     :target_scitype)

function _as_named_tuple(s::Checker)
    names = propertynames(s)
    NamedTuple{names}(Tuple(getproperty(s, p) for p in names))
end

# function Base.show(io::IO, ::MIME"text/plain", S::Checker)
#     show(io, MIME("text/plain"), _as_named_tuple(S))
# end

matching(X)       = Checker{false,missing,missing,scitype(X),missing}()
matching(X, y)    = Checker{true,missing,missing,scitype(X),scitype(y)}()
matching(X, y, w) = Checker{true,true,false,scitype(X),scitype(y)}()
matching(X, y, w::AbstractDict) =
    Checker{true,false,true,scitype(X),scitype(y)}()

(f::Checker{false,
            missing,
            missing,
            XS,
            missing})(model::MLJModels.ModelProxy) where XS =
    !(model.is_supervised) &&
    XS <: model.input_scitype

(f::Checker{true,
            missing,
            missing,
            XS,
            yS})(model::MLJModels.ModelProxy) where {XS,yS} =
    model.is_supervised &&
    XS <: model.input_scitype &&
    yS <: model.target_scitype

(f::Checker{true,
            true,
            false,
            XS,
            yS})(model::MLJModels.ModelProxy) where {XS,yS} =
    model.is_supervised &&
    model.supports_weights &&
    XS <: model.input_scitype &&
    yS <: model.target_scitype

(f::Checker{true,
            false,
            true,
            XS,
            yS})(model::MLJModels.ModelProxy) where {XS,yS} =
    model.is_supervised &&
    model.supports_class_weights &&
    XS <: model.input_scitype &&
    yS <: model.target_scitype

(f::Checker)(name::String; pkg=nothing) = f(info(name, pkg=pkg))
(f::Checker)(realmodel::Model) = f(info(realmodel))

matching(model::MLJModels.ModelProxy, args...) = matching(args...)(model)
matching(name::String, args...; pkg=nothing) =
    matching(info(name, pkg=pkg), args...)
matching(realmodel::Model, args...) = matching(info(realmodel), args...)


## MODEL QUERY


"""
    models()

List all models in the MLJ registry. Here and below *model* means the
registry metadata entry for a genuine model type (a proxy for types
whose defining code may not be loaded).

    models(filters..)

List all models `m` for which `filter(m)` is true, for each `filter`
in `filters`.

    models(matching(X, y))

List all supervised models compatible with training data `X`, `y`.

    models(matching(X))

List all unsupervised models compatible with training data `X`.


Excluded in the listings are the built-in model-wraps, like `EnsembleModel`,
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
    models(needle::Union{AbstractString,Regex})

List all models whole `name` or `docstring` matches a given `needle`.
"""
function MLJBase.models(needle::Union{AbstractString,Regex})
    f = model -> occursin(needle, model.name) || occursin(needle, model.docstring)
    return MLJBase.models(f)
end

"""
    localmodels(; modl=Main)
    localmodels(conditions...; modl=Main)
    localmodels(needle::Union{AbstractString,Regex}; modl=Main)


List all models whose names are in the namespace of the specified
module `modl`, or meeting the `conditions`, if specified. Here a
*condition* is a `Bool`-valued function on models.

See also [`models`](@ref)

"""
function localmodels(args...; modl=Main)
    modeltypes = localmodeltypes(modl)
    names = map(modeltypes) do M
        MMI.name(M)
    end
    return filter(models(args...)) do handle
        handle.name in names
    end
end
