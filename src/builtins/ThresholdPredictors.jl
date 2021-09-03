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
                                  model=nothing,
                                  threshold=0.5)
    length(args) < 2 || throw(ArgumentError(
        "At most one non-keyword argument allowed. "))
    if length(args) === 1
        atom = only(args)
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

function clean!(model::ThresholdUnion)
    if !(AbstractVector{Multiclass{2}} <: target_scitype(model.model) ||
        AbstractVector{OrderedFactor{2}} <: target_scitype(model.model))
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
    if model isa Probabilistic
        scitype(args[2]) <: AbstractVector{Multiclass{2}} && begin
            first_class, second_class = levels(args[2])
            @warn "Taking positive class as `$(second_class)` and negative class as
            `$(first_class)`.
            Coerce target to `OrderedFactor{2}` to suppress this warning."
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
   length(classes(yhat)) == 2 || begin
       # Due to resampling it's possible for Predicted
       #`AbstractVector{<:UnivariateFinite}`
       # to contain one class. Hence the need for the following warning
       @warn "Predicted `AbstractVector{<:UnivariateFinite}`"*
           " contains only 1 class. Hence predictions will only "*
           "contain this class "*
           "irrrespective of the set `threshold` "
       return mode.(yhat)
   end
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
    max_prob, max_class = findmax([_div(dict[ref], threshold[ref]) for ref in keys(dict)])
    return yhat.decoder(max_class)
end

function _predict_threshold(yhat::AbstractArray{<:UnivariateFinite}, threshold)
    return _predict_threshold.(yhat, (threshold,))
end

function _predict_threshold(yhat::UnivariateFiniteArray{S,V,R,P,N}, threshold) where{S,V,R,P,N}
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

## METADATA

# Note: input traits are inherited from the wrapped model


MMI.package_name(::Type{<:ThresholdUnion}) = "MLJModels"
MMI.package_uuid(::Type{<:ThresholdUnion}) = ""
MMI.is_wrapper(::Type{<:ThresholdUnion}) = true
MMI.package_url(::Type{<:ThresholdUnion}) =
    "https://github.com/alan-turing-institute/MLJModels.jl"

for New in THRESHOLD_TYPE_EXS
    New_str = string(New)
    quote
        MMI.is_pure_julia(::Type{<:$New{M}}) where M =
            MMI.is_pure_julia(M)
        MMI.supports_weights(::Type{<:$New{M}}) where M =
            MMI.supports_weights(M)
        MMI.supports_class_weights(::Type{<:$New{M}}) where M =
            MMI.supports_weights(M)
        MMI.load_path(::Type{<:$New}) =
            "MLJModels.$($New_str)"
        MMI.input_scitype(::Type{<:$New{M}}) where M =
            MMI.input_scitype(M)
        function MMI.target_scitype(::Type{<:$New{M}}) where M
            T = MMI.target_scitype(M)
            if T <: AbstractVector{<:OrderedFactor}
                return AbstractVector{<:OrderedFactor{2}}
            elseif T <: AbstractVector{<:Multiclass}
                return AbstractVector{<:Multiclass{2}}
            elseif T <: AbstractVector{<:Finite}
                return AbstractVector{<:Finite{2}}
            end
            return T
        end
    end |> eval
end
