# FUNCTIONS TO INSPECT METADATA OF REGISTERED MODELS AND TO
## FACILITATE MODEL SEARCH

# sort the model trait names:
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
    doc = p.docstring
    L = min(length(doc), 50)
    doc = doc[1:L]
    L < 50 || (doc *= "...")
    pcompact = merge(p, (; docstring=doc))
    PrettyPrinting.pprint(stream, pcompact)
end

# returns named tuple version of the dictionary i=info_dict(SomeModelType):
function info_as_named_tuple(i)
    property_values = Tuple(i[property] for property in PROPERTY_NAMES)
    return NamedTuple{PROPERTY_NAMES}(property_values)
end


## INFO AND DOC

const err_handle_missing_name(name) = ArgumentError(
    "There is no model named \"$name\" in "*
    "the registry. \n Run `models()` to view all "*
    "registered models, or `models(needle)` to restrict search to "*
    "models with string `needle` in their name or documentation. ")

const err_handle_ambiguous_name(name, pkgs) = ArgumentError(
    "Ambiguous model type name. Use pkg=... .\n"*
    "The model $name is provided by these packages:\n $pkgs.\n")

const err_handle_name_not_in_pkg(name, pkg) =
    ArgumentError("The package \"$pkg\" does not appear to "*
                  "provide the model \"$name\". \n"*
                  "Use models() to list all models. ")

const ERR_DOC_EXPECTS_STRING = ArgumentError(
    "The `doc` function expects a string as its first argument. Use `@doc T` or `?T` "*
    "to extract the document string for a type or function `T` in scope. ")

doc_handle(thing::String) =
    """
    Return the $thing for the registered model type having the specified
    `name`. The key-word argument `pkg` is required in the case of
    duplicate names, unless `interactive=true` is specified instead.
    """

"""
    handle(name; pkg=nothing, interactive=false)

Private method.

$(doc_handle("handle"))

"""
function handle(name; pkg=nothing, interactive=false)
    name in NAMES ||
        throw(err_handle_missing_name(name))
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
                throw(err_handle_ambiguous_name(name, pkgs))
            end
        end
    else
        handle = Handle(name, pkg)
        haskey(INFO_GIVEN_HANDLE, handle) ||
            throw(err_handle_name_not_in_pkg(name, pkg))
    end
    return handle
end

function doc(handle::Handle)
    i = info_as_named_tuple(INFO_GIVEN_HANDLE[handle])
    return Markdown.parse(i.docstring)
end

"""
    doc(name::String; pkg=nothing, interactive=false)

$(doc_handle("documentation string"))

"""
doc(name::String; kwargs...) = doc(handle(name; kwargs...))

# because user may try to apply `doc` to model instances or types:
doc(::Any) = throw(ERR_DOC_EXPECTS_STRING)


StatisticalTraits.info(handle::Handle) =
    info_as_named_tuple(INFO_GIVEN_HANDLE[handle])


"""
    info(name::String; pkg=nothing, interactive=false)

$(doc_handle("metadata"))

To instead return the model document string (without importing
defining code) do `doc(name; pkg=...)`

See also [`doc`](@ref).

"""
StatisticalTraits.info(name::String; kwargs...) =
    info(handle(name; kwargs...))

"""
    info(model::Model)

Return the metadata (trait values) associated with the specified
`model`.

This is equivalent to `info(name; pkg=pkg)` where `name::String` is
the name of the model type, and `pkg::String` the name of the core
algorithm-providing package (assuming the model registry is
up-to-date).

"""
function StatisticalTraits.info(M::Type{<:MMI.Model})
    values =
        tuple([eval(:($trait($M))) for trait in PROPERTY_NAMES]...)
    return NamedTuple{PROPERTY_NAMES}(values)
end
StatisticalTraits.info(model::MMI.Model) = info(typeof(model))


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

const WARN_MULTITARGET = "'y' is a table with only one column. " *
                         "If 'y' is a table, we assume you want to do multi-target modeling." *
                         "If you actually want to do single-target modeling, " *
                         "you need to convert 'y' to a Vector."

function warn_if_single_column(y)
    if Tables.istable(y) && (length(Tables.columnnames(y)) == 1)
        @warn WARN_MULTITARGET
    end
    return nothing
end

matching(X) = Checker{false,missing,missing,scitype(X),missing}()
function matching(X, y)
    warn_if_single_column(y)
    return Checker{true,missing,missing,scitype(X),scitype(y)}()
end
function matching(X, y, w)
    warn_if_single_column(y)
    return Checker{true,true,false,scitype(X),scitype(y)}()
end
function matching(X, y, w::AbstractDict)
    warn_if_single_column(y)
    return Checker{true,false,true,scitype(X),scitype(y)}()
end

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
    models(; wrappers=false)

List all models in the MLJ registry. Here and below *model* means the registry metadata
entry for a genuine model type (a proxy for types whose defining code may not be
loaded). To include wrappers and other composite models, such as `TunedModel` and `Stack`,
specify `wrappers=true`.

    models(filters...; wrappers=false)

List all models `m` for which `filter(m)` is true, for each `filter`
in `filters`.

    models(matching(X, y); wrappers=false)

List all supervised models compatible with training data `X`, `y`.

    models(matching(X); wrappers=false)

List all unsupervised models compatible with training data `X`.


### Example

If

    task(model) = model.is_supervised && model.is_probabilistic

then `models(task)` lists all supervised models making probabilistic
predictions.

See also: [`localmodels`](@ref).

"""
function models(conditions...; wrappers=false)
    wrappers || (conditions = (conditions..., m-> !m.is_wrapper))
    unsorted = filter(info.(keys(INFO_GIVEN_HANDLE))) do model
        all(c(model) for c in conditions)
    end
    return sort!(unsorted)
end

"""
    models(needle::Union{AbstractString,Regex}; wrappers=false)

List all models whole `name` or `docstring` matches a given `needle`.
"""
function models(needle::Union{AbstractString,Regex}; kwargs...)
    f = model ->
        occursin(needle, model.name) || occursin(needle, model.docstring)
    return models(f; kwargs...)
end

# get the model types in top-level of given module's namespace:
function localmodeltypes(modl; toplevel=false, wrappers=false)
    ft = finaltypes(Model)
    return filter!(ft) do M
        name = MLJModelInterface.name(M)
        test1 = !toplevel || isdefined(modl, Symbol(name))
        (!MLJModelInterface.is_wrapper(M) || wrappers) && test1
    end
end

"""
    localmodels(; modl=Main, wrappers=false)
    localmodels(filters...; modl=Main, wrappers=false)
    localmodels(needle::Union{AbstractString,Regex}; modl=Main, wrappers=false)

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
function localmodels(args...; modl=Main, kwargs...)
    modeltypes = localmodeltypes(modl; kwargs...)
    handles = map(modeltypes) do M
        Handle(MMI.name(M), MMI.package_name(M))
    end
    return filter(models(args...)) do model
        Handle(model.name, model.package_name) in handles
    end
end
