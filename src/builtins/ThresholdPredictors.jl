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
const err_unsupported_model_type(T) = ArgumentError(
    "`BinaryThresholdPredictor` does not support atomic models with supertype `$T`. "*
    "Supported supertypes are: `$(keys(_type_given_atom))`. "
)

for (atom, type) in type_given_atom
    @eval threshold_constructor(::Type{$atom}) = $type
end
threshold_constructor(A) = throw(err_unsupported_model_type(A))

"""
    BinaryThresholdPredictor(model; threshold=0.5)

Wrap the `Probabilistic` model, `model`, assumed to support binary classification, as a
`Deterministic` model, by applying the specified `threshold` to the positive class
probability. In addition to conventional supervised classifiers, it can also be applied to
outlier detection models that predict normalized scores - in the form of appropriate
`UnivariateFinite` distributions - that is, models that subtype
`AbstractProbabilisticUnsupervisedDetector` or `AbstractProbabilisticSupervisedDetector`.

By convention the positive class is the second class returned by
`levels(y)`, where `y` is the target.

If `threshold=0.5` then calling `predict` on the wrapped model is
equivalent to calling `predict_mode` on the atomic model.

# Example

Below is an application to the well-known Pima Indian diabetes dataset, including
optimization of the `threshold` parameter, with a high balanced accuracy the
objective. The target class distribution is 500 positives to 268 negatives.

Loading the data:

```julia
using MLJ, Random
rng = Xoshiro(123)

diabetes = OpenML.load(43582)
outcome, X = unpack(diabetes, ==(:Outcome), rng=rng);
y = coerce(Int.(outcome), OrderedFactor);
```

Choosing a probabilistic classifier:

```julia
EvoTreesClassifier = @load EvoTreesClassifier
prob_predictor = EvoTreesClassifier()
```

Wrapping in `TunedModel` to get a deterministic classifier with `threshold` as a new
hyperparameter:

```julia
point_predictor = BinaryThresholdPredictor(prob_predictor, threshold=0.6)
Xnew, _ = make_moons(3, rng=rng)
mach = machine(point_predictor, X, y) |> fit!
predict(mach, X)[1:3] # [0, 0, 0]
```

Estimating performance:

```julia
balanced = BalancedAccuracy(adjusted=true)
e = evaluate!(mach, resampling=CV(nfolds=6), measures=[balanced, accuracy])
e.measurement[1] # 0.405 ± 0.089
```

Wrapping in tuning strategy to learn `threshold` that maximizes balanced accuracy:

```julia
r = range(point_predictor, :threshold, lower=0.1, upper=0.9)
tuned_point_predictor = TunedModel(
    point_predictor,
    tuning=RandomSearch(rng=rng),
    resampling=CV(nfolds=6),
    range = r,
    measure=balanced,
    n=30,
)
mach2 = machine(tuned_point_predictor, X, y) |> fit!
optimized_point_predictor = report(mach2).best_model
optimized_point_predictor.threshold # 0.260
predict(mach2, X)[1:3] # [1, 1, 0]
```

Estimating the performance of the auto-thresholding model (nested resampling here):

```julia
e = evaluate!(mach2, resampling=CV(nfolds=6), measure=[balanced, accuracy])
e.measurement[1] # 0.477 ± 0.110
```

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

    A = MMI.abstract_type(atom)
    T = threshold_constructor(A)
    metamodel = T(atom, Float64(threshold))
    message = clean!(metamodel)
    isempty(message) || @warn message
    return metamodel
end

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
        !(get_scitype(args[2]) <: AbstractVector{<:Union{Missing, OrderedFactor{2}}})
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
        model.model, verbosity-1, map(unwrap, args)...
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
        model.model,
        verbosity-1,
        old_fitresult[1],
        old_cache[1],
        map(unwrap, args)...
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


## SERIALIZATION

function MMI.save(model::ThresholdUnion, fitresult)
    atomic_fitresult, threshold = fitresult
    atom = model.model
    return MMI.save(atom, atomic_fitresult), threshold
end
function MMI.restore(model::ThresholdUnion, serializable_fitresult)
    atomic_serializable_fitresult, threshold = serializable_fitresult
    atom = model.model
    return MMI.restore(atom, atomic_serializable_fitresult), threshold
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

## Data Front-end
"""
    ReformattedTarget(y, levels, scitype)

Intenal Use Only.
Wrapper containing a **model specific** target, `y`, its scientific type,
`scitype`(the scientific representation of the **user supplied** target),
and associated levels (the levels of the **user supplied** target), `levels`.
"""
struct ReformattedTarget{T, L, S<:Type}
    y::T
    levels::L
    scitype::S
end

function Base.:(==)(x1::ReformattedTarget, x2::ReformattedTarget)
    return x1.y == x2.y && x1.levels == x2.levels && x1.scitype == x2.scitype
end

unwrap(x) = x
unwrap(x::ReformattedTarget) = getfield(x, :y)
CategoricalArrays.levels(x::ReformattedTarget) = getfield(x, :levels)

function ScientificTypesBase.scitype(
    x::ReformattedTarget, ::ScientificTypes.DefaultConvention
)
    return getfield(x, :scitype)
end

# Faster acesss to scitype for `ReformattedTarget`
get_scitype(x::ReformattedTarget) = x.scitype

function MMI.reformat(model::ThresholdUnion, args...)
    atom = model.model
    reformatted_args = reformat(atom, args...)
    if length(args) > 1
        reformatted_X, reformatted_target, other_reformatted_args... = reformatted_args
        target = args[2]
        wrapped_reformatted_target = ReformattedTarget(
            reformatted_target, levels(target), scitype(target)
        )
        reformatted_args_with_wrapped_target = (
            reformatted_X, wrapped_reformatted_target, other_reformatted_args...
        )
        return reformatted_args_with_wrapped_target
    end
    return reformatted_args
end

function MMI.selectrows(model::ThresholdUnion, I, reformatted_args_with_wrapped_target...)
    atom = model.model
    reformatted_args_rows = selectrows(
        atom, I, map(unwrap, reformatted_args_with_wrapped_target)...
    )

    if length(reformatted_args_with_wrapped_target) > 1
        reformatted_X_rows, reformatted_target_rows, other_reformatted_args_rows... =
            reformatted_args_rows
        reformatted_target = reformatted_args_with_wrapped_target[2]
        wrapped_reformatted_target_rows = ReformattedTarget(
            reformatted_target_rows,
            levels(reformatted_target),
            get_scitype(reformatted_target)
        )

        reformatted_args_rows_with_wrapped_target = (
            reformatted_X_rows,
            wrapped_reformatted_target_rows,
            other_reformatted_args_rows...
        )
        return reformatted_args_rows_with_wrapped_target
    end

    return reformatted_args_rows
end
