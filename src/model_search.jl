## FUNCTIONS TO INSPECT METADATA OF REGISTERED MODELS AND TO
## FACILITATE MODEL SEARCH

# sort and add to the model trait names:
const property_names = sort(MODEL_TRAITS_IN_REGISTRY)
const alpha = [:name, :package_name, :is_supervised]
const omega = [:input_scitype, :target_scitype, :output_scitype]
const both = vcat(alpha, omega)
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
    printstyled(p.docstring, bold=false, color=:magenta)
    println(stream)
    PrettyPrinting.pprint(stream, p)
end

# returns named tuple version of the dictionary i=info_dict(SomeModelType):
function info_as_named_tuple(i)
    property_values = Tuple(i[property] for property in PROPERTY_NAMES)
    return NamedTuple{PROPERTY_NAMES}(property_values)
end


## INFO

ScientificTypes.info(handle::Handle) =
    info_as_named_tuple(INFO_GIVEN_HANDLE[handle])

"""
    info(name::String; pkg=nothing, interactive=false)

Returns the metadata for the registered model type with specified
`name`. The key-word argument `pkg` is required in the case of
duplicate names.

"""
function ScientificTypes.info(name::String; pkg=nothing, interactive=false)
    name in NAMES ||
        throw(ArgumentError("There is no model named \"$name\" in "*
                            "the registry. \n Run `models()` to view all "*
                            "registered models."))
    # get the handle:
    if pkg == nothing
        handle  = Handle(name) # returns (name=..., pkg=missing) if ambiguous
        if ismissing(handle.pkg)
            pkgs = PKGS_GIVEN_NAME[name]
            if interactive
                choice = request(
                    "Multiple packages provide $name. Choose a package: ",
                    pkgs...)
                handle = Handle(name, pkgs[choice])
            else
                message = "Ambiguous model type name. Use pkg=... .\n"*
                    "The model $name is provided by these packages:\n $pkgs.\n"
                throw(ArgumentError(message))
            end
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
function ScientificTypes.info(M::Type{<:MMI.Model})
    values =
        tuple([eval(:($trait($M))) for trait in PROPERTY_NAMES]...)
    info_as_named_tuple(info_dict(M))
    return NamedTuple{PROPERTY_NAMES}(values)
end
ScientificTypes.info(model::MMI.Model) = info(typeof(model))


## MATCHING

# Note. `ModelProxy` is the type of a model's metadata entry (a named
# tuple). So, `info("PCA")` has this type, for example.


# Basic idea

if false

    matching(model::MLJModels.ModelProxy, X) =
        !(model.is_supervised) && scitype(X) <: model.input_scitype

    matching(model::MLJModels.ModelProxy, X, y) =
        model.is_supervised &&
        scitype(X) <: model.input_scitype &&
        scitype(y) <: model.target_scitype

    matching(model::MLJModels.ModelProxy, X, y, w::AbstractVector{<:Real}) =
        model.is_supervised &&
        model.supports_weights &&
        scitype(X) <: model.input_scitype &&
        scitype(y) <: model.target_scitype

end

# Implementation

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

not_missing_and_true(x) = !ismissing(x) && x

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
    not_missing_and_true(model.supports_weights) &&
    XS <: model.input_scitype &&
    yS <: model.target_scitype

(f::Checker{true,
            false,
            true,
            XS,
            yS})(model::MLJModels.ModelProxy) where {XS,yS} =
    model.is_supervised &&
    not_missing_and_true(model.supports_class_weights) &&
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
function models(conditions...)
    unsorted = filter(info.(keys(INFO_GIVEN_HANDLE))) do model
        all(c(model) for c in conditions)
    end
    return sort!(unsorted)
end

"""
    models(needle::Union{AbstractString,Regex})

List all models whole `name` or `docstring` matches a given `needle`.
"""
function models(needle::Union{AbstractString,Regex})
    f = model ->
        occursin(needle, model.name) || occursin(needle, model.docstring)
    return models(f)
end

"""
    localmodels(; modl=Main)
    localmodels(filters...; modl=Main)
    localmodels(needle::Union{AbstractString,Regex}; modl=Main)

List all models currently available to the user from the module `modl`
without importing a package, and which additional pass through the
specified filters. Here a *filter* is a `Bool`-valued function on
models.

Use `load_path` to get the path to some model returned, as in these
examples:

    ms = localmodels()
    model = ms[1]
    load_path(model)

See also [`models`](@ref), [`load_path`](@ref).

"""
function localmodels(args...; modl=Main, toplevel=false)
    modeltypes = localmodeltypes(modl, toplevel=toplevel)
    handles = map(modeltypes) do M
        Handle(MMI.name(M), MMI.package_name(M))
    end
    return filter(models(args...)) do model
        Handle(model.name, model.package_name) in handles
    end
end
