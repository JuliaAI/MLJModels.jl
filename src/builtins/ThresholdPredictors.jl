##
###  BinaryThresholdPredictor
##

const THRESHOLD_SUPPORTED_ATOMS = (
    :Probabilistic,
    :ProbabilisticUnsupervisedDetector,
    :ProbabilisticSupervisedDetector)

# Each supported atomic type gets its own wrapper:

const type_given_atom = Dict(
    :Probabilistic =>
    :BinaryThresholdPredictor,
    :ProbabilisticUnsupervisedDetector =>
    :ThresholdUnsupervisedDetector,
    :ProbabilisticSupervisedDetector =>
    :ThresholdSupervisedDetector)

# ...which must have appropriate supertype:

const super_given_atom = Dict(
    :Probabilistic =>
    :Deterministic,
    :ProbabilisticUnsupervisedDetector =>
    :DeterministicUnsupervisedDetector,
    :ProbabilisticSupervisedDetector =>
    :DeterministicSupervisedDetector)

# the type definitions:

for From in THRESHOLD_SUPPORTED_ATOMS
    New = type_given_atom[From]
    To  = super_given_atom[From]
    ex = quote
        mutable struct $New{M <: $From} <: $To
            model::M
            threshold::Float64
        end
    end
    eval(ex)
end

# dict whose keys and values are now types instead of symbols:
const _type_given_atom = Dict()
for atom in THRESHOLD_SUPPORTED_ATOMS
    atom_str = string(atom)
    type = type_given_atom[atom]
    @eval(_type_given_atom[$atom] = $type)
end

const THRESHOLD_TYPES = values(_type_given_atom)
const THRESHOLD_TYPE_EXS = values(type_given_atom)
const ThresholdUnion = Union{THRESHOLD_TYPES...}
const ThresholdSupported = Union{keys(_type_given_atom)...}

const ERR_MODEL_UNSPECIFIED = ArgumentError(
    "Expecting atomic model as argument. None specified. ")
warn_classes(first_class, second_class) =
    "Taking positive class as `$(second_class)` and negative class as"*
    "`$(first_class)`."*
    "Coerce target to `OrderedFactor{2}` to suppress this warning, "*
    "ensuring that positive class > negative class. "
const ERR_CLASSES_DETECTOR = ArgumentError(
    "Targets for detector models must be ordered. Consider coercing to "*
    "`OrderedFactor`, ensuring that outlier class > inlier class. ")
const ERR_TARGET_NOT_BINARY = ArgumentError(
    "Target `y` must have two classes in its  pool, even if only one "*
    "class is manifest. ")

"""
    BinaryThresholdPredictor(model; threshold=0.5)

Wrap the `Probabilistic` model, `model`, assumed to support binary
classification, as a `Deterministic` model, by applying the specified
`threshold` to the positive class probability. Can also be applied to
outlier detection models that predict normalized scores - in the form
of appropriate `UnivariateFinite` distributions - that is, models that
subtype `AbstractProbabilisticUnsupervisedDetector` or
`AbstractProbabilisticSupervisedDetector`.

By convention the positive class is the second class returned by
`levels(y)`, where `y` is the target.

If `threshold=0.5` then calling `predict` on the wrapped model is
equivalent to calling `predict_mode` on the atomic model.

"""
function BinaryThresholdPredictor(args...;
                                  wrapped_model=nothing,
                                  model=wrapped_model,
                                  threshold=0.5)
    length(args) < 2 || throw(ArgumentError(
        "At most one non-keyword argument allowed. "))
    if length(args) === 1
        atom = first(args)
        model === nothing ||
            @warn "Using `model=$atom`. Ignoring specification `model=$model`. "
    else
        model === nothing && throw(ERR_MODEL_UNSPECIFIED)
        atom = model
    end

    metamodel =
        _type_given_atom[MMI.abstract_type(atom)](atom, Float64(threshold))
    message = clean!(metamodel)
    isempty(message) || @warn message
    return metamodel
end

######################################
# Begin code to be removed in 0.15.0 #
######################################

function Base.getproperty(model::ThresholdUnion, name::Symbol)
    name === :model && return getfield(model, :model)
    name === :threshold && return getfield(model, :threshold)
    error("type BinaryThresholdPredictor has no field $name")
end

function Base.setproperty!(model::ThresholdUnion, name::Symbol, value)
    name === :model && return setfield!(model, :model, value)
    name === :threshold && return setfield!(model, :threshold, value)
    error("type BinaryThresholdPredictor has no field $name")
end
####################################
# End code to be removed in 0.15.0 #
####################################

function clean!(model::ThresholdUnion)
    T = target_scitype(model.model)
    if !(AbstractVector{Multiclass{2}} <: T ||
        AbstractVector{OrderedFactor{2}} <: T || Unknown <: T)
        throw(ArgumentError("`model` has unsupported target_scitype "*
              "`$(target_scitype(model.model))`. "))
    end
    message = ""
    if model.threshold >= 1 || model.threshold < 0
        message = message*"`threshold` should be "*
        "in the range [0, 1). Resetting to 0.5. "
        model.threshold = 0.5
    end
    return message
end

function MMI.fit(model::ThresholdUnion, verbosity::Int, args...)

    if length(args) > 1 &&
        !(scitype(args[2]) <: AbstractVector{<:Union{Missing,OrderedFactor}})
        L = levels(args[2])
        length(L) == 2 || throw(ERR_TARGET_NOT_BINARY)
        first_class, second_class = L
        if model.model isa Probabilistic
            @warn warn_classes(first_class, second_class)
        else
            throw(ERR_CLASSES_DETECTOR)
        end
    end
    model_fitresult, model_cache, model_report = MMI.fit(
        model.model, verbosity-1, args...
    )
    cache = (model_cache = model_cache,)
    report = (model_report = model_report,)
    fitresult = (model_fitresult, model.threshold)
    return fitresult, cache, report
end

function MMI.update(
    model::ThresholdUnion, verbosity::Int, old_fitresult, old_cache, args...
)
    model_fitresult, model_cache, model_report = MMI.update(
        model.model, verbosity-1, old_fitresult[1], old_cache[1], args...
    )
    cache = (model_cache = model_cache,)
    report = (model_report = model_report,)
    fitresult = (model_fitresult, model.threshold)
    return fitresult, cache, report
end

function MMI.fitted_params(model::ThresholdUnion, fitresult)
    return (
        model_fitted_params = MMI.fitted_params(
            model.model, fitresult[1]),
        )
end

function MMI.predict(model::ThresholdUnion, fitresult, X)
    yhat = MMI.predict(model.model, fitresult[1], X)
    threshold = (1 - fitresult[2], fitresult[2])
    return _predict_threshold(yhat, threshold)
end

# `_div` and `_predict_threshold` methods are defined to help generalize to
# `MulticlassThresholdPredictor` (TODO)
_div(x,y) = ifelse( iszero(x) && x==y, Inf, x/y)

function _predict_threshold(yhat::UnivariateFinite, threshold)
    dict = yhat.prob_given_ref
    length(threshold) == length(dict) || throw(
        ArgumentError(
        "`length(threshold)` has to equal number of classes in specified "*
        "`UnivariateFinite` distribution."
        )
    )
    max_prob, max_class =
        findmax([_div(dict[ref], threshold[ref]) for ref in keys(dict)])
    return yhat.decoder(max_class)
end

function _predict_threshold(yhat::AbstractArray{<:UnivariateFinite}, threshold)
    return _predict_threshold.(yhat, (threshold,))
end

function _predict_threshold(yhat::UnivariateFiniteArray{S,V,R,P,N},
                            threshold) where{S,V,R,P,N}
    dict = yhat.prob_given_ref
    length(threshold) == length(dict) || throw(
        ArgumentError(
        "`length(threshold)` has to equal number of classes in specified "*
        "`UnivariateFiniteArray`."
        )
    )
    d = yhat.decoder(1)
    levs = levels(d)
    ord = isordered(d)
    # Array to house the predicted classes
    ret = CategoricalArray{V, N, R}(undef, size(yhat), levels=levs, ordered=ord)
    #ret = Array{CategoricalValue{V, R}, N}(undef, size(yhat))
    # `temp` vector allocted once to be used for calculations in each loop
    temp = Vector{Float64}(undef, length(dict))
    # allocate vectors of keys to enable use of @simd in innermost loop
    #kv = keys(dict) |> collect
    for i in eachindex(ret)
        #@simd for ind in eachindex(kv)
         @simd for ref in eachindex(temp)
           #@inbounds ref = kv[ind]
           #@inbounds temp[ind] = _div(dict[ref][i], threshold[ref])
           @inbounds temp[ref] = _div(dict[ref][i], threshold[ref])
        end
        max_prob, max_class = findmax(temp)
        @inbounds ret[i] = yhat.decoder(max_class)
    end
    return ret
end


## TRAITS

# Note: input traits are inherited from the wrapped model

MMI.package_name(::Type{<:ThresholdUnion}) = "MLJModels"
MMI.package_uuid(::Type{<:ThresholdUnion}) = ""
MMI.is_wrapper(::Type{<:ThresholdUnion}) = true
MMI.package_url(::Type{<:ThresholdUnion}) =
    "https://github.com/alan-turing-institute/MLJModels.jl"

for New in THRESHOLD_TYPE_EXS
    New_str = string(New)
    quote
        MMI.load_path(::Type{<:$New{M}}) where M = "MLJModels."*$New_str
    end |> eval
end


for trait in [:supports_weights,
              :supports_class_weights,
              :is_pure_julia,
              :input_scitype,
              :output_scitype,
              :supports_training_losses,
              ]

    # try to get trait at level of types ("failure" here just
    # means falling back to `Unknown`):
    for New in THRESHOLD_TYPE_EXS
        quote
            MMI.$trait(::Type{<:$New{M}}) where M = MMI.$trait(M)
        end |> eval
    end

    # needed because traits are not always deducable from
    # the type (eg, `target_scitype` and `Pipeline` models):
    eval(:(MMI.$trait(model::ThresholdUnion) = MMI.$trait(model.model)))
end


# ## Target scitype

_make_binary(::Type) = Unknown
_make_binary(::Type{<:AbstractVector{<:OrderedFactor}}) =
    AbstractVector{<:OrderedFactor{2}}
_make_binary(::Type{<:AbstractVector{<:Multiclass}}) =
    AbstractVector{<:Multiclass{2}}
_make_binary(::Type{<:AbstractVector{<:Finite}}) =
    AbstractVector{<:Finite{2}}
_make_binary(::Type{<:AbstractVector{<:Union{Missing,OrderedFactor}}}) =
    AbstractVector{<:Union{Missing,OrderedFactor{2}}}
_make_binary(::Type{<:AbstractVector{<:Union{Missing,Multiclass}}}) =
    AbstractVector{<:Union{Missing,Multiclass{2}}}
_make_binary(::Type{<:AbstractVector{<:Union{Missing,Finite}}}) =
    AbstractVector{<:Union{Missing,Finite{2}}}

# at level of types:
for New in THRESHOLD_TYPE_EXS
    quote
        MMI.target_scitype(::Type{<:$New{M}}) where M =
            _make_binary(MMI.target_scitype(M))
    end |> eval
end

# at level of instances:
MMI.target_scitype(model::ThresholdUnion) =
    _make_binary(MMI.target_scitype(model.model))


# ## Iteration parameter

# at level of types:
for New in THRESHOLD_TYPE_EXS
    quote
        MMI.iteration_parameter(::Type{<:$New{M}}) where M =
            MLJModels.prepend(:model, MMI.iteration_parameter(M))
    end |> eval
end

# at level of instances:
MMI.iteration_parameter(model::ThresholdUnion) =
    MLJModels.prepend(:model, MMI.iteration_parameter(model.model))


# ## TRAINING LOSSES SUPPORT

MMI.training_losses(thresholder::ThresholdUnion, thresholder_report) =
    MMI.training_losses(thresholder.model, thresholder_report.model_report)
