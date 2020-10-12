##
###  BinaryThresholdPredictor
##
"""
    BinaryThresholdPredictor(wrapped_model=ConstantClassifier(), threshold=0.5)
    Wraps a `Probabilistic` model(`wrapped_model`) supporting binary classification in a 
`BinaryThresholdPredictor` model. Training the `BinaryThresholdPredictor` model results in 
training the underlying `wrapped_model`. 
`BinaryThresholdPredictor` model predictions are deterministic and depend on the specified 
probability `threshold`(which lies in [0, 1) ). The positive class is predicted if the 
probability value exceeds the threshold, where the postive class is the second class 
returned by levels(y), if y is the target.

Note:
- The default `threshold`(0.5) predicts the modal class.
"""
mutable struct BinaryThresholdPredictor{M <: Probabilistic} <: Deterministic
    wrapped_model::M
    threshold::Float64
end

function clean!(model::BinaryThresholdPredictor)
    if !(AbstractVector{Multiclass{2}} <: target_scitype(model.wrapped_model) || 
        AbstractVector{OrderedFactor{2}} <: target_scitype(model.wrapped_model))
        throw(ArgumentError("`model` has unsupported target_scitype "*
              "`$(target_scitype(model.wrapped_model))`. "))
    end
    message = ""
    if model.threshold >= 1 || model.threshold < 0
        message = message*"`threshold` should be "*
        "in the range [0, 1). Resetting to 0.5. "
        model.threshold = 0.5
    end
    return message
end

function BinaryThresholdPredictor(;wrapped_model=ConstantClassifier(), threshold=0.5)
    model = BinaryThresholdPredictor(wrapped_model, Float64(threshold))
    message = clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.fit(model::BinaryThresholdPredictor, verbosity::Int, args...)
    scitype(args[2]) <: AbstractVector{Multiclass{2}} && begin
        first_class, second_class = levels(args[2])
        @warn "Taking positive class as `$(second_class)` and negative class as
        `$(first_class)`.
        Coerce target to `OrderedFactor{2}` to suppress this warning."
    end
    model_fitresult, model_cache, model_report = MLJBase.fit(
        model.wrapped_model, verbosity-1, args...
    )
    cache = (wrapped_model_cache = model_cache,)
    report = (wrapped_model_report = model_report,)
    fitresult = (model_fitresult, model.threshold)
    return fitresult, cache, report
end

function MLJBase.update(
    model::BinaryThresholdPredictor, verbosity::Int, old_fitresult, old_cache, args...
)
    model_fitresult, model_cache, model_report = MLJBase.update(
        model.wrapped_model, verbosity-1, old_fitresult[1], old_cache[1], args...
    )
    cache = (wrapped_model_cache = model_cache,)
    report = (wrapped_model_report = model_report,)
    fitresult = (model_fitresult, model.threshold)
    return fitresult, cache, report
end

function MLJBase.fitted_params(model::BinaryThresholdPredictor, fitresult)
    return (
        threshold= fitresult[2],
        wrapped_model_fitted_params = MLJBase.fitted_params(
            model.wrapped_model, fitresult[1]
        )
    )
end

function MLJBase.predict(model::BinaryThresholdPredictor, fitresult, X)
   yhat = MLJBase.predict(model.wrapped_model, fitresult[1], X)
   length(yhat.prob_given_ref) == 2 || begin
   # Due to resampling it's possible for Predicted `AbstractVector{<:UnivariateFinite}`
   # to contain one class. Hence the need for the following warning
   @warn "Predicted `AbstractVector{<:UnivariateFinite}`"*
       " contains only 1 class. Hence predictions will only contain this class "*
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

MLJBase.supports_weights(::Type{<:BinaryThresholdPredictor{M}}) where M =
    MLJBase.supports_weights(M)
MLJBase.load_path(::Type{<:BinaryThresholdPredictor}) =
    "MLJModels.BinaryThresholdPredictor"
MLJBase.package_name(::Type{<:BinaryThresholdPredictor}) = "MLJModels"
MLJBase.package_uuid(::Type{<:BinaryThresholdPredictor}) = ""
MLJBase.is_wrapper(::Type{<:BinaryThresholdPredictor}) = true
MLJBase.package_url(::Type{<:BinaryThresholdPredictor}) =
    "https://github.com/alan-turing-institute/MLJModels.jl"
MLJBase.is_pure_julia(::Type{<:BinaryThresholdPredictor{M}}) where M =
    MLJBase.is_pure_julia(M)
MLJBase.input_scitype(::Type{<:BinaryThresholdPredictor{M}}) where M =
    MLJBase.input_scitype(M)
MLJBase.target_scitype(::Type{<:BinaryThresholdPredictor}) =
    AbstractVector{<:Binary}

