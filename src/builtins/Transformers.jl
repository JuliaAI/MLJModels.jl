# Note that doc-strings appear at the end


# # IMPUTER

round_median(v::AbstractVector) = v -> round(eltype(v), median(v))

_median(e)       = skipmissing(e) |> median
_round_median(e) = skipmissing(e) |> (f -> round(eltype(f), median(f)))
_mode(e)         = skipmissing(e) |> mode

@with_kw_noshow mutable struct UnivariateFillImputer <: Unsupervised
    continuous_fill::Function = _median
    count_fill::Function      = _round_median
    finite_fill::Function     = _mode
end

function MMI.fit(transformer::UnivariateFillImputer,
                      verbosity::Integer,
                      v)

    filler(v, ::Type) = throw(ArgumentError(
        "Imputation is not supported for vectors "*
        "of elscitype $(elscitype(v))."))
    filler(v, ::Type{<:Union{Continuous,Missing}}) =
        transformer.continuous_fill(v)
    filler(v, ::Type{<:Union{Count,Missing}}) =
        transformer.count_fill(v)
    filler(v, ::Type{<:Union{Finite,Missing}}) =
        transformer.finite_fill(v)

    fitresult = (filler=filler(v, elscitype(v)),)
    cache = nothing
    report = NamedTuple()

    return fitresult, cache, report

end

function replace_missing(::Type{<:Finite}, vnew, filler)
   all(in(levels(filler)), levels(vnew)) ||
        error(ArgumentError("The `column::AbstractVector{<:Finite}`"*
                            " to be transformed must contain the same levels"*
                            " as the categorical value to be imputed"))
   replace(vnew, missing => filler)

end

function replace_missing(::Type, vnew, filler)
   T = promote_type(nonmissing(eltype(vnew)), typeof(filler))
   w_tight = similar(vnew, T)
   @inbounds for i in eachindex(vnew)
        if ismissing(vnew[i])
           w_tight[i] = filler
        else
           w_tight[i] = vnew[i]
        end
   end
   return w_tight
end

function MMI.transform(transformer::UnivariateFillImputer,
                           fitresult,
                           vnew)

    filler = fitresult.filler

    scitype(filler) <: elscitype(vnew) ||
    error("Attempting to impute a value of scitype $(scitype(filler)) "*
    "into a vector of incompatible elscitype, namely $(elscitype(vnew)). ")

    if elscitype(vnew) >: Missing
        w_tight = replace_missing(nonmissing(elscitype(vnew)), vnew, filler)
    else
        w_tight = vnew
    end

    return w_tight
end

MMI.fitted_params(::UnivariateFillImputer, fitresult) = fitresult

@with_kw_noshow mutable struct FillImputer <: Unsupervised
    features::Vector{Symbol}  = Symbol[]
    continuous_fill::Function = _median
    count_fill::Function      = _round_median
    finite_fill::Function     = _mode
end

function MMI.fit(transformer::FillImputer, verbosity::Int, X)

    s = schema(X)
    features_seen = s.names |> collect # "seen" = "seen in fit"
    scitypes_seen = s.scitypes |> collect

    features = isempty(transformer.features) ? features_seen :
        transformer.features

    issubset(features, features_seen) || throw(ArgumentError(
    "Some features specified do not exist in the supplied table. "))

    # get corresponding scitypes:
    mask = map(features_seen) do ftr
        ftr in features
    end
    features = @view features_seen[mask] # `features` re-ordered
    scitypes = @view scitypes_seen[mask]
    features_and_scitypes = zip(features, scitypes) #|> collect

    # now keep those features that are imputable:
    function isimputable(ftr, T::Type)
        if verbosity > 0 && !isempty(transformer.features)
            @info "Feature $ftr will not be imputed "*
            "(imputation for $T not supported). "
        end
        return false
    end
    isimputable(ftr, ::Type{<:Union{Continuous,Missing}}) = true
    isimputable(ftr, ::Type{<:Union{Count,Missing}}) = true
    isimputable(ftr, ::Type{<:Union{Finite,Missing}}) = true

    mask = map(features_and_scitypes) do tup
        isimputable(tup...)
    end
    features_to_be_imputed = @view features[mask]

    univariate_transformer =
        UnivariateFillImputer(continuous_fill=transformer.continuous_fill,
                              count_fill=transformer.count_fill,
                              finite_fill=transformer.finite_fill)
    univariate_fitresult(ftr) = MMI.fit(univariate_transformer,
                                            verbosity - 1,
                                            selectcols(X, ftr))[1]

    fitresult_given_feature =
        Dict(ftr=> univariate_fitresult(ftr) for ftr in features_to_be_imputed)

    fitresult = (features_seen=features_seen,
                 univariate_transformer=univariate_transformer,
                 fitresult_given_feature=fitresult_given_feature)
    report    = NamedTuple()
    cache     = nothing

    return fitresult, cache, report
end

function MMI.transform(transformer::FillImputer, fitresult, X)

    features_seen = fitresult.features_seen # seen in fit
    univariate_transformer = fitresult.univariate_transformer
    fitresult_given_feature = fitresult.fitresult_given_feature

    all_features = Tables.schema(X).names

    # check that no new features have appeared:
    all(e -> e in features_seen, all_features) || throw(ArgumentError(
        "Attempting to transform table with "*
        "feature labels not seen in fit.\n"*
        "Features seen in fit = $features_seen.\n"*
        "Current features = $([all_features...]). "))

    features = keys(fitresult_given_feature)

    cols = map(all_features) do ftr
        col = MMI.selectcols(X, ftr)
        if ftr in features
            fr = fitresult_given_feature[ftr]
            return transform(univariate_transformer, fr, col)
        end
        return col
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))
    return MMI.table(named_cols, prototype=X)

end

function MMI.fitted_params(::FillImputer, fr)
    dict = fr.fitresult_given_feature
    filler_given_feature = Dict(ftr=>dict[ftr].filler for ftr in keys(dict))
    return (features_seen_in_fit=fr.features_seen,
            univariate_transformer=fr.univariate_transformer,
            filler_given_feature=filler_given_feature)
end


# # FOR FEATURE (COLUMN) SELECTION

mutable struct FeatureSelector <: Unsupervised
    # features to be selected; empty means all
    features::Union{Vector{Symbol}, Function}
    ignore::Bool # features to be ignored
end

# keyword constructor
function FeatureSelector(
    ;
    features::Union{AbstractVector{Symbol}, Function}=Symbol[],
    ignore::Bool=false
)
    transformer = FeatureSelector(features, ignore)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::FeatureSelector)
    err = ""
    if (
        typeof(transformer.features) <: AbstractVector{Symbol} &&
        isempty(transformer.features) &&
        transformer.ignore
    )
        err *= "Features to be ignored must be specified in features field."
    end
    return err
end

function MMI.fit(transformer::FeatureSelector, verbosity::Int, X)
    all_features = Tables.schema(X).names

    if transformer.features isa AbstractVector{Symbol}
        if isempty(transformer.features)
           features = collect(all_features)
        else
            features = if transformer.ignore
                !issubset(transformer.features, all_features) && verbosity > -1 &&
                @warn("Excluding non-existent feature(s).")
                filter!(all_features |> collect) do ftr
                   !(ftr in transformer.features)
                end
            else
                issubset(transformer.features, all_features) ||
                throw(ArgumentError("Attempting to select non-existent feature(s)."))
                transformer.features |> collect
            end
        end
    else
        features = if transformer.ignore
            filter!(all_features |> collect) do ftr
                !(transformer.features(ftr))
            end
        else
            filter!(all_features |> collect) do ftr
                transformer.features(ftr)
            end
        end
        isempty(features) && throw(
            ArgumentError("No feature(s) selected.\n The specified Bool-valued"*
              " callable with the `ignore` option set to `$(transformer.ignore)` "*
              "resulted in an empty feature set for selection")
         )
    end

    fitresult = features
    report = NamedTuple()
    return fitresult, nothing, report
end

MMI.fitted_params(::FeatureSelector, fitresult) = (features_to_keep=fitresult,)

function MMI.transform(::FeatureSelector, features, X)
    all(e -> e in Tables.schema(X).names, features) ||
        throw(ArgumentError("Supplied frame does not admit previously selected features."))
    return MMI.selectcols(X, features)
end


# # UNIVARIATE DISCRETIZER

# helper function:
reftype(::CategoricalArray{<:Any,<:Any,R}) where R = R

@with_kw_noshow mutable struct UnivariateDiscretizer <:Unsupervised
    n_classes::Int = 512
end

struct UnivariateDiscretizerResult{C}
    odd_quantiles::Vector{Float64}
    even_quantiles::Vector{Float64}
    element::C
end

function MMI.fit(transformer::UnivariateDiscretizer, verbosity::Int, X)
    n_classes = transformer.n_classes
    quantiles = quantile(X, Array(range(0, stop=1, length=2*n_classes+1)))
    clipped_quantiles = quantiles[2:2*n_classes] # drop 0% and 100% quantiles

    # odd_quantiles for transforming, even_quantiles used for
    # inverse_transforming:
    odd_quantiles = clipped_quantiles[2:2:(2*n_classes-2)]
    even_quantiles = clipped_quantiles[1:2:(2*n_classes-1)]

    # determine optimal reference type for encoding as categorical:
    R = reftype(categorical(1:n_classes, compress=true))
    output_prototype = categorical(R(1):R(n_classes), compress=true, ordered=true)
    element = output_prototype[1]

    cache  = nothing
    report = NamedTuple()

    res = UnivariateDiscretizerResult(odd_quantiles, even_quantiles, element)
    return res, cache, report
end

# acts on scalars:
function transform_to_int(
            result::UnivariateDiscretizerResult{<:CategoricalValue{R}},
            r::Real) where R
    k = oneR = R(1)
    @inbounds for q in result.odd_quantiles
        if r > q
            k += oneR
        end
    end
    return k
end

# transforming scalars:
MMI.transform(::UnivariateDiscretizer, result, r::Real) =
    transform(result.element, transform_to_int(result, r))

# transforming vectors:
function MMI.transform(::UnivariateDiscretizer, result, v)
   w = [transform_to_int(result, r) for r in v]
   return transform(result.element, w)
end

# inverse_transforming raw scalars:
function MMI.inverse_transform(
    transformer::UnivariateDiscretizer, result , k::Integer)
    k <= transformer.n_classes && k > 0 ||
        error("Cannot transform an integer outside the range "*
              "`[1, n_classes]`, where `n_classes = $(transformer.n_classes)`")
    return result.even_quantiles[k]
end

# inverse transforming a categorical value:
function MMI.inverse_transform(
    transformer::UnivariateDiscretizer, result, e::CategoricalValue)
    k = CategoricalArrays.DataAPI.unwrap(e)
    return inverse_transform(transformer, result, k)
end

# inverse transforming raw vectors:
MMI.inverse_transform(transformer::UnivariateDiscretizer, result,
                          w::AbstractVector{<:Integer}) =
      [inverse_transform(transformer, result, k) for k in w]

# inverse transforming vectors of categorical elements:
function MMI.inverse_transform(transformer::UnivariateDiscretizer, result,
                          wcat::AbstractVector{<:CategoricalValue})
    w = MMI.int(wcat)
    return [inverse_transform(transformer, result, k) for k in w]
end

MMI.fitted_params(::UnivariateDiscretizer, fitresult) = (
    odd_quantiles=fitresult.odd_quantiles,
    even_quantiles=fitresult.even_quantiles
)


# # CONTINUOUS TRANSFORM OF TIME TYPE FEATURES

mutable struct UnivariateTimeTypeToContinuous <: Unsupervised
    zero_time::Union{Nothing, TimeType}
    step::Period
end

function UnivariateTimeTypeToContinuous(;
    zero_time=nothing, step=Dates.Hour(24))
    model = UnivariateTimeTypeToContinuous(zero_time, step)
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(model::UnivariateTimeTypeToContinuous)
    # Step must be able to be added to zero_time if provided.
    msg = ""
    if model.zero_time !== nothing
        try
            tmp = model.zero_time + model.step
        catch err
            if err isa MethodError
                model.zero_time, model.step, status, msg = _fix_zero_time_step(
                    model.zero_time, model.step)
                if status === :error
                    # Unable to resolve, rethrow original error.
                    throw(err)
                end
            else
                throw(err)
            end
        end
    end
    return msg
end

function _fix_zero_time_step(zero_time, step)
    # Cannot add time parts to dates nor date parts to times.
    # If a mismatch is encountered. Conversion from date parts to time parts
    # is possible, but not from time parts to date parts because we cannot
    # represent fractional date parts.
    msg = ""
    if zero_time isa Dates.Date && step isa Dates.TimePeriod
        # Convert zero_time to a DateTime to resolve conflict.
        if step % Hour(24) === Hour(0)
            # We can convert step to Day safely
            msg = "Cannot add `TimePeriod` `step` to `Date` `zero_time`. Converting `step` to `Day`."
            step = convert(Day, step)
        else
            # We need datetime to be compatible with the step.
            msg = "Cannot add `TimePeriod` `step` to `Date` `zero_time`. Converting `zero_time` to `DateTime`."
            zero_time = convert(DateTime, zero_time)
        end
        return zero_time, step, :success, msg
    elseif zero_time isa Dates.Time && step isa Dates.DatePeriod
        # Convert step to Hour if possible. This will fail for
        # isa(step, Month)
        msg = "Cannot add `DatePeriod` `step` to `Time` `zero_time`. Converting `step` to `Hour`."
        step = convert(Hour, step)
        return zero_time, step, :success, msg
    else
        return zero_time, step, :error, msg
    end
end

function MMI.fit(model::UnivariateTimeTypeToContinuous, verbosity::Int, X)
    if model.zero_time !== nothing
        min_dt = model.zero_time
        step = model.step
        # Check zero_time is compatible with X
        example = first(X)
        try
            X - min_dt
        catch err
            if err isa MethodError
                @warn "`$(typeof(min_dt))` `zero_time` is not compatible with `$(eltype(X))` vector. Attempting to convert `zero_time`."
                min_dt = convert(eltype(X), min_dt)
            else
                throw(err)
            end
        end
    else
        min_dt = minimum(X)
        step = model.step
        message = ""
        try
            min_dt + step
        catch err
            if err isa MethodError
                min_dt, step, status, message = _fix_zero_time_step(min_dt, step)
                if status === :error
                    # Unable to resolve, rethrow original error.
                    throw(err)
                end
            else
                throw(err)
            end
        end
        isempty(message) || @warn message
    end
    cache = nothing
    report = NamedTuple()
    fitresult = (min_dt, step)
    return fitresult, cache, report
end

function MMI.transform(model::UnivariateTimeTypeToContinuous, fitresult, X)
    min_dt, step = fitresult
    if typeof(min_dt) ≠ eltype(X)
        # Cannot run if eltype in transform differs from zero_time from fit.
        throw(ArgumentError("Different `TimeType` encountered during `transform` than expected from `fit`. Found `$(eltype(X))`, expected `$(typeof(min_dt))`"))
    end
    # Set the size of a single step.
    next_time = min_dt + step
    if next_time == min_dt
        # Time type loops if step is a multiple of Hour(24), so calculate the
        # number of multiples, then re-scale to Hour(12) and adjust delta to match original.
        m = step / Dates.Hour(12)
        delta = m * (
            Float64(Dates.value(min_dt + Dates.Hour(12)) - Dates.value(min_dt)))
    else
        delta = Float64(Dates.value(min_dt + step) - Dates.value(min_dt))
    end
    return @. Float64(Dates.value(X - min_dt)) / delta
end


# # UNIVARIATE STANDARDIZATION

"""
    UnivariateStandardizer()

Transformer type for standardizing (whitening) single variable data.

This model may be deprecated in the future. Consider using
[`Standardizer`](@ref), which handles both tabular *and* univariate data.

"""
mutable struct UnivariateStandardizer <: Unsupervised end

function MMI.fit(transformer::UnivariateStandardizer, verbosity::Int,
             v::AbstractVector{T}) where T<:Real
    std(v) > eps(Float64) ||
        @warn "Extremely small standard deviation encountered in standardization."
    fitresult = (mean(v), std(v))
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end

MMI.fitted_params(::UnivariateStandardizer, fitresult) =
    (mean=fitresult[1], std=fitresult[2])


# for transforming single value:
function MMI.transform(transformer::UnivariateStandardizer, fitresult, x::Real)
    mu, sigma = fitresult
    return (x - mu)/sigma
end

# for transforming vector:
MMI.transform(transformer::UnivariateStandardizer, fitresult, v) =
              [transform(transformer, fitresult, x) for x in v]

# for single values:
function MMI.inverse_transform(transformer::UnivariateStandardizer, fitresult, y::Real)
    mu, sigma = fitresult
    return mu + y*sigma
end

# for vectors:
MMI.inverse_transform(transformer::UnivariateStandardizer, fitresult, w) =
    [inverse_transform(transformer, fitresult, y) for y in w]


# # STANDARDIZATION OF ORDINAL FEATURES OF TABULAR DATA

mutable struct Standardizer <: Unsupervised
    # features to be standardized; empty means all
    features::Union{AbstractVector{Symbol}, Function}
    ignore::Bool # features to be ignored
    ordered_factor::Bool
    count::Bool
end

# keyword constructor
function Standardizer(
    ;
    features::Union{AbstractVector{Symbol}, Function}=Symbol[],
    ignore::Bool=false,
    ordered_factor::Bool=false,
    count::Bool=false
)
    transformer = Standardizer(features, ignore, ordered_factor, count)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::Standardizer)
    err = ""
    if (
        typeof(transformer.features) <: AbstractVector{Symbol} &&
        isempty(transformer.features) &&
        transformer.ignore
    )
        err *= "Features to be ignored must be specified in features field."
    end
    return err
end

function MMI.fit(transformer::Standardizer, verbosity::Int, X)

    # if not a table, it must be an abstract vector, eltpye AbstractFloat:
    is_univariate = !Tables.istable(X)

    # are we attempting to standardize Count or OrderedFactor?
    is_invertible = !transformer.count && !transformer.ordered_factor

    # initialize fitresult:
    fitresult_given_feature = LittleDict{Symbol,Tuple{AbstractFloat,AbstractFloat}}()

    # special univariate case:
    if is_univariate
        fitresult_given_feature[:unnamed] =
            MMI.fit(UnivariateStandardizer(), verbosity - 1, X)[1]
        return (is_univariate=true,
                is_invertible=true,
                fitresult_given_feature=fitresult_given_feature),
        nothing, nothing
    end

    all_features = Tables.schema(X).names
    feature_scitypes =
        collect(elscitype(selectcols(X, c)) for c in all_features)
    scitypes = Vector{Type}([Continuous])
    transformer.ordered_factor && push!(scitypes, OrderedFactor)
    transformer.count && push!(scitypes, Count)
    AllowedScitype = Union{scitypes...}

    # determine indices of all_features to be transformed
    if transformer.features isa AbstractVector{Symbol}
        if isempty(transformer.features)
            cols_to_fit = filter!(eachindex(all_features) |> collect) do j
                feature_scitypes[j] <: AllowedScitype
            end
        else
            !issubset(transformer.features, all_features) && verbosity > -1 &&
                @warn "Some specified features not present in table to be fit. "
            cols_to_fit = filter!(eachindex(all_features) |> collect) do j
                ifelse(
                    transformer.ignore,
                    !(all_features[j] in transformer.features) &&
                        feature_scitypes[j] <: AllowedScitype,
                    (all_features[j] in transformer.features) &&
                        feature_scitypes[j] <: AllowedScitype
                )
            end
        end
    else
        cols_to_fit = filter!(eachindex(all_features) |> collect) do j
            ifelse(
                transformer.ignore,
                !(transformer.features(all_features[j])) &&
                    feature_scitypes[j] <: AllowedScitype,
                (transformer.features(all_features[j])) &&
                    feature_scitypes[j] <: AllowedScitype
            )
        end
    end

    isempty(cols_to_fit) && verbosity > -1 &&
        @warn "No features to standarize."

    # fit each feature and add result to above dict
    verbosity > 1 && @info "Features standarized: "
    for j in cols_to_fit
        col_data = if (feature_scitypes[j] <: OrderedFactor)
            coerce(selectcols(X, j), Continuous)
        else
            selectcols(X, j)
        end
        col_fitresult, _, _ =
            MMI.fit(UnivariateStandardizer(), verbosity - 1, col_data)
        fitresult_given_feature[all_features[j]] = col_fitresult
        verbosity > 1 &&
            @info "  :$(all_features[j])    mu=$(col_fitresult[1])  "*
            "sigma=$(col_fitresult[2])"
    end

    fitresult = (is_univariate=false, is_invertible=is_invertible,
                 fitresult_given_feature=fitresult_given_feature)
    cache = nothing
    report = (features_fit=keys(fitresult_given_feature),)

    return fitresult, cache, report
end

function MMI.fitted_params(::Standardizer, fitresult)
    is_univariate, _, dic = fitresult
    is_univariate &&
        return fitted_params(UnivariateStandardizer(), dic[:unnamed])
    features_fit = keys(dic) |> collect
    zipped = map(ftr->dic[ftr], features_fit)
    means, stds = zip(zipped...) |> collect
    return (; features_fit, means, stds)
end

MMI.transform(::Standardizer, fitresult, X) =
    _standardize(transform, fitresult, X)

function MMI.inverse_transform(::Standardizer, fitresult, X)
    fitresult.is_invertible ||
        error("Inverse standardization is not supported when `count=true` "*
              "or `ordered_factor=true` during fit. ")
    return _standardize(inverse_transform, fitresult, X)
end

function _standardize(operation, fitresult, X)

    # `fitresult` is dict of column fitresults, keyed on feature names
    is_univariate, _, fitresult_given_feature = fitresult

    if is_univariate
        univariate_fitresult = fitresult_given_feature[:unnamed]
        return operation(UnivariateStandardizer(), univariate_fitresult, X)
    end

    features_to_be_transformed = keys(fitresult_given_feature)

    all_features = Tables.schema(X).names

    all(e -> e in all_features, features_to_be_transformed) ||
        error("Attempting to transform data with incompatible feature labels.")

    col_transformer = UnivariateStandardizer()

    cols = map(all_features) do ftr
        ftr_data = selectcols(X, ftr)
        if ftr in features_to_be_transformed
            col_to_transform = coerce(ftr_data, Continuous)
            operation(col_transformer,
                      fitresult_given_feature[ftr],
                      col_to_transform)
        else
            ftr_data
        end
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))

    return MMI.table(named_cols, prototype=X)
end


# # UNIVARIATE BOX-COX TRANSFORMATIONS

function standardize(v)
    map(v) do x
        (x - mean(v))/std(v)
    end
end

function midpoints(v::AbstractVector{T}) where T <: Real
    return [0.5*(v[i] + v[i + 1]) for i in 1:(length(v) -1)]
end

function normality(v)
    n  = length(v)
    v  = standardize(convert(Vector{Float64}, v))
    # sort and replace with midpoints
    v = midpoints(sort!(v))
    # find the (approximate) expected value of the size (n-1)-ordered statistics for
    # standard normal:
    d = Distributions.Normal(0,1)
    w = map(collect(1:(n-1))/n) do x
        quantile(d, x)
    end
    return cor(v, w)
end

function boxcox(lambda, c, x::Real)
    c + x >= 0 || throw(DomainError)
    if lambda == 0.0
        c + x > 0 || throw(DomainError)
        return log(c + x)
    end
    return ((c + x)^lambda - 1)/lambda
end

boxcox(lambda, c, v::AbstractVector{T}) where T <: Real =
    [boxcox(lambda, c, x) for x in v]

@with_kw_noshow mutable struct UnivariateBoxCoxTransformer <: Unsupervised
    n::Int      = 171   # nbr values tried in optimizing exponent lambda
    shift::Bool = false # whether to shift data away from zero
end

function MMI.fit(transformer::UnivariateBoxCoxTransformer, verbosity::Int,
             v::AbstractVector{T}) where T <: Real

    m = minimum(v)
    m >= 0 || error("Cannot perform a Box-Cox transformation on negative data.")

    c = 0.0 # default
    if transformer.shift
        if m == 0
            c = 0.2*mean(v)
        end
    else
        m != 0 || error("Zero value encountered in data being Box-Cox transformed.\n"*
                        "Consider calling `fit!` with `shift=true`.")
    end

    lambdas = range(-0.4, stop=3, length=transformer.n)
    scores = Float64[normality(boxcox(l, c, v)) for l in lambdas]
    lambda = lambdas[argmax(scores)]

    return  (lambda, c), nothing, NamedTuple()
end

MMI.fitted_params(::UnivariateBoxCoxTransformer, fitresult) =
    (λ=fitresult[1], c=fitresult[2])

# for X scalar or vector:
MMI.transform(transformer::UnivariateBoxCoxTransformer, fitresult, X) =
    boxcox(fitresult..., X)

# scalar case:
function MMI.inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           fitresult, x::Real)
    lambda, c = fitresult
    if lambda == 0
        return exp(x) - c
    else
        return (lambda*x + 1)^(1/lambda) - c
    end
end

# vector case:
function MMI.inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           fitresult, w::AbstractVector{T}) where T <: Real
    return [inverse_transform(transformer, fitresult, y) for y in w]
end


# # ONE HOT ENCODING

@with_kw_noshow mutable struct OneHotEncoder <: Unsupervised
    features::Vector{Symbol}   = Symbol[]
    drop_last::Bool            = false
    ordered_factor::Bool       = true
    ignore::Bool               = false
end

# we store the categorical refs for each feature to be encoded and the
# corresponing feature labels generated (called
# "names"). `all_features` is stored to ensure no new features appear
# in new input data, causing potential name clashes.
struct OneHotEncoderResult <: MMI.MLJType
    all_features::Vector{Symbol} # all feature labels
    ref_name_pairs_given_feature::Dict{Symbol,Vector{Union{Pair{<:Unsigned,Symbol}, Pair{Missing, Symbol}}}}
    fitted_levels_given_feature::Dict{Symbol, CategoricalArray}
end

# join feature and level into new label without clashing with anything
# in all_features:
function compound_label(all_features, feature, level)
    label = Symbol(string(feature, "__", level))
    # in the (rare) case subft is not a new feature label:
    while label in all_features
        label = Symbol(string(label,"_"))
    end
    return label
end

function MMI.fit(transformer::OneHotEncoder, verbosity::Int, X)

    all_features = Tables.schema(X).names # a tuple not vector

    if isempty(transformer.features)
        specified_features = collect(all_features)
    else
        if transformer.ignore
            specified_features = filter(all_features |> collect) do ftr
                !(ftr in transformer.features)
            end
        else
            specified_features = transformer.features
        end
    end


    ref_name_pairs_given_feature = Dict{Symbol,Vector{Pair{<:Unsigned,Symbol}}}()

    allowed_scitypes = ifelse(
        transformer.ordered_factor,
        Union{Missing, Finite},
        Union{Missing, Multiclass}
    )
    fitted_levels_given_feature = Dict{Symbol, CategoricalArray}()
    col_scitypes = schema(X).scitypes
    # apply on each feature
    for j in eachindex(all_features)
        ftr = all_features[j]
        col = MMI.selectcols(X,j)
        T = col_scitypes[j]
        if T <: allowed_scitypes && ftr in specified_features
            ref_name_pairs_given_feature[ftr] = Pair{<:Unsigned,Symbol}[]
            shift = transformer.drop_last ? 1 : 0
            levels = classes(col)
            fitted_levels_given_feature[ftr] = levels
            if verbosity > 0
                @info "Spawning $(length(levels)-shift) sub-features "*
                "to one-hot encode feature :$ftr."
            end
            for level in levels[1:end-shift]
                ref = MMI.int(level)
                name = compound_label(all_features, ftr, level)
                push!(ref_name_pairs_given_feature[ftr], ref => name)
            end
        end
    end

    fitresult = OneHotEncoderResult(collect(all_features),
                                    ref_name_pairs_given_feature,
                                    fitted_levels_given_feature)

    # get new feature names
    d = ref_name_pairs_given_feature
    new_features = Symbol[]
    features_to_be_transformed = keys(d)
    for ftr in all_features
        if ftr in features_to_be_transformed
            append!(new_features, last.(d[ftr]))
        else
            push!(new_features, ftr)
        end
    end

    report = (features_to_be_encoded=
              collect(keys(ref_name_pairs_given_feature)),
              new_features=new_features)
    cache = nothing

    return fitresult, cache, report
end

MMI.fitted_params(::OneHotEncoder, fitresult) = (
    all_features = fitresult.all_features,
    fitted_levels_given_feature = fitresult.fitted_levels_given_feature,
    ref_name_pairs_given_feature = fitresult.ref_name_pairs_given_feature,
)

# If v=categorical('a', 'a', 'b', 'a', 'c') and MMI.int(v[1]) = ref
# then `_hot(v, ref) = [true, true, false, true, false]`
hot(v::AbstractVector{<:CategoricalValue}, ref) = map(v) do c
    MMI.int(c) == ref
end

function hot(col::AbstractVector{<:Union{Missing, CategoricalValue}}, ref) map(col) do c
    if ismissing(ref)
        missing
    else
        MMI.int(c) == ref
    end
end
end

function MMI.transform(transformer::OneHotEncoder, fitresult, X)
    features = Tables.schema(X).names     # tuple not vector

    d = fitresult.ref_name_pairs_given_feature

    # check the features match the fit result
    all(e -> e in fitresult.all_features, features) ||
        error("Attempting to transform table with feature "*
              "names not seen in fit. ")
    new_features = Symbol[]
    new_cols = [] # not Vector[] !!
    features_to_be_transformed = keys(d)
    for ftr in features
        col = MMI.selectcols(X, ftr)
        if ftr in features_to_be_transformed
            Set(fitresult.fitted_levels_given_feature[ftr]) ==
                Set(classes(col)) ||
            error("Found category level mismatch in feature `$(ftr)`. "*
            "Consider using `levels!` to ensure fitted and transforming "*
            "features have the same category levels.")
            append!(new_features, last.(d[ftr]))
            pairs = d[ftr]
            refs = first.(pairs)
            names = last.(pairs)
            cols_to_add = map(refs) do ref
                if ismissing(ref) missing
                else float.(hot(col, ref))
                end
            end
            append!(new_cols, cols_to_add)
        else
            push!(new_features, ftr)
            push!(new_cols, col)
        end
    end
    named_cols = NamedTuple{tuple(new_features...)}(tuple(new_cols)...)
    return MMI.table(named_cols, prototype=X)
end


# # CONTINUOUS_ENCODING

@with_kw_noshow mutable struct ContinuousEncoder <: Unsupervised
    drop_last::Bool                = false
    one_hot_ordered_factors::Bool  = false
end

function MMI.fit(transformer::ContinuousEncoder, verbosity::Int, X)

    # what features can be converted and therefore kept?
    s = schema(X)
    features = s.names
    scitypes = s.scitypes
    Convertible = Union{Continuous, Finite, Count}
    feature_scitype_tuples = zip(features, scitypes) |> collect
    features_to_keep  =
        first.(filter(t -> last(t) <: Convertible, feature_scitype_tuples))
    features_to_be_dropped = setdiff(collect(features), features_to_keep)

    if verbosity > 0
        if !isempty(features_to_be_dropped)
            @info "Some features cannot be replaced with "*
            "`Continuous` features and will be dropped: "*
            "$features_to_be_dropped. "
        end
    end

    # fit the one-hot encoder:
    hot_encoder =
        OneHotEncoder(ordered_factor=transformer.one_hot_ordered_factors,
                      drop_last=transformer.drop_last)
    hot_fitresult, _, hot_report = MMI.fit(hot_encoder, verbosity - 1, X)

    new_features = setdiff(hot_report.new_features, features_to_be_dropped)

    fitresult = (features_to_keep=features_to_keep,
                 one_hot_encoder=hot_encoder,
                 one_hot_encoder_fitresult=hot_fitresult)

    # generate the report:
    report = (features_to_keep=features_to_keep,
              new_features=new_features)

    cache = nothing

    return fitresult, cache, report

end

MMI.fitted_params(::ContinuousEncoder, fitresult) = fitresult

function MMI.transform(transformer::ContinuousEncoder, fitresult, X)

    features_to_keep, hot_encoder, hot_fitresult = values(fitresult)

    # dump unseen or untransformable features:
    selector = FeatureSelector(features=features_to_keep)
    selector_fitresult, _, _ = MMI.fit(selector, 0, X)
    X0 = transform(selector, selector_fitresult, X)

    # one-hot encode:
    X1 = transform(hot_encoder, hot_fitresult, X0)

    # convert remaining to continuous:
    return coerce(X1, Count=>Continuous, OrderedFactor=>Continuous)

end


# # INTERACTION TRANSFORMER

@mlj_model mutable struct InteractionTransformer <: Static
    order::Int                                          = 2::(_ > 1)
    features::Union{Nothing, Vector{Symbol}}            = nothing::(_ !== nothing ? length(_) > 1 : true)
end

infinite_scitype(col) = eltype(scitype(col)) <: Infinite

actualfeatures(features::Nothing, table) =
    filter(feature -> infinite_scitype(Tables.getcolumn(table, feature)), Tables.columnnames(table))

function actualfeatures(features::Vector{Symbol}, table)
    diff = setdiff(features, Tables.columnnames(table))
    diff != [] && throw(ArgumentError(string("Column(s) ", join([x for x in diff], ", "), " are not in the dataset.")))

    for feature in features
        infinite_scitype(Tables.getcolumn(table, feature)) || throw(ArgumentError("Column $feature's scitype is not Infinite."))
    end
    return Tuple(features)
end

interactions(columns, order::Int) =
    collect(Iterators.flatten(combinations(columns, i) for i in 2:order))

interactions(columns, variables...) =
    .*((Tables.getcolumn(columns, var) for var in variables)...)

function MMI.transform(model::InteractionTransformer, _, X)
    features = actualfeatures(model.features, X)
    interactions_ = interactions(features, model.order)
    interaction_features = Tuple(Symbol(join(inter, "_")) for inter in interactions_)
    columns = Tables.Columns(X)
    interaction_table = NamedTuple{interaction_features}([interactions(columns, inter...) for inter in interactions_])
    return merge(Tables.columntable(X), interaction_table)
end

# # METADATA FOR ALL BUILT-IN TRANSFORMERS

metadata_pkg.(
    (FeatureSelector, UnivariateStandardizer,
     UnivariateDiscretizer, Standardizer,
     UnivariateBoxCoxTransformer, UnivariateFillImputer,
     OneHotEncoder, FillImputer, ContinuousEncoder,
     UnivariateTimeTypeToContinuous, InteractionTransformer),
    package_name       = "MLJModels",
    package_uuid       = "d491faf4-2d78-11e9-2867-c94bc002c0b7",
    package_url        = "https://github.com/alan-turing-institute/MLJModels.jl",
    is_pure_julia      = true,
    package_license    = "MIT")

metadata_model(UnivariateFillImputer,
    input_scitype = Union{AbstractVector{<:Union{Continuous,Missing}},
                  AbstractVector{<:Union{Count,Missing}},
                  AbstractVector{<:Union{Finite,Missing}}},
    output_scitype= Union{AbstractVector{<:Continuous},
                  AbstractVector{<:Count},
                  AbstractVector{<:Finite}},
    human_name = "single variable fill imputer",
    load_path  = "MLJModels.UnivariateFillImputer")

metadata_model(FillImputer,
    input_scitype   = Table,
    output_scitype = Table,
    load_path    = "MLJModels.FillImputer")

metadata_model(FeatureSelector,
    input_scitype   = Table,
    output_scitype = Table,
    load_path    = "MLJModels.FeatureSelector")

metadata_model(UnivariateDiscretizer,
    input_scitype   = AbstractVector{<:Continuous},
    output_scitype = AbstractVector{<:OrderedFactor},
    human_name = "single variable discretizer",
    load_path    = "MLJModels.UnivariateDiscretizer")

metadata_model(UnivariateStandardizer,
    input_scitype   = AbstractVector{<:Infinite},
    output_scitype = AbstractVector{Continuous},
    human_name = "single variable discretizer",
    load_path    = "MLJModels.UnivariateStandardizer")

metadata_model(Standardizer,
    input_scitype   = Union{Table, AbstractVector{<:Continuous}},
    output_scitype = Union{Table, AbstractVector{<:Continuous}},
    load_path    = "MLJModels.Standardizer")

metadata_model(UnivariateBoxCoxTransformer,
    input_scitype   = AbstractVector{Continuous},
    output_scitype = AbstractVector{Continuous},
    human_name = "single variable Box-Cox transformer",
    load_path    = "MLJModels.UnivariateBoxCoxTransformer")

metadata_model(OneHotEncoder,
    input_scitype   = Table,
    output_scitype = Table,
    human_name = "one-hot encoder",
    load_path    = "MLJModels.OneHotEncoder")

metadata_model(ContinuousEncoder,
    input_scitype   = Table,
    output_scitype = Table(Continuous),
    load_path    = "MLJModels.ContinuousEncoder")

metadata_model(UnivariateTimeTypeToContinuous,
    input_scitype   = AbstractVector{<:ScientificTimeType},
    output_scitype = AbstractVector{Continuous},
    human_name ="single variable transformer that creates "*
         "continuous representations of temporally typed data",
    load_path    = "MLJModels.UnivariateTimeTypeToContinuous")

metadata_model(InteractionTransformer,
    input_scitype   = Tuple{Table},
    output_scitype = Table,
    human_name = "interaction transformer",
    load_path    = "MLJModels.InteractionTransformer")

# # DOC STRINGS

# The following document strings comply with the MLJ standard.

"""
$(MLJModelInterface.doc_header(UnivariateFillImputer))

Use this model to imputing `missing` values in a vector with a fixed
value learned from the non-missing values of training vector.

For imputing missing values in tabular data, use [`FillImputer`](@ref)
instead.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, x)

where

- `x`: any abstract vector with element scitype `Union{Missing, T}`
  where `T` is a subtype of `Continuous`, `Multiclass`,
  `OrderedFactor` or `Count`; check scitype using `scitype(x)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `continuous_fill`: function or other callable to determine value to
  be imputed in the case of `Continuous` (abstract float) data;
  default is to apply `median` after skipping `missing` values

- `count_fill`: function or other callable to determine value to be
  imputed in the case of `Count` (integer) data; default is to apply
  rounded `median` after skipping `missing` values

- `finite_fill`: function or other callable to determine value to be
  imputed in the case of `Multiclass` or `OrderedFactor` data
  (categorical vectors); default is to apply `mode` after skipping
  `missing` values


# Operations

- `transform(mach, xnew)`: return `xnew` with missing values imputed
  with the fill values learned when fitting `mach`


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `filler`: the fill value to be imputed in all new data


# Examples

```
using MLJ
imputer = UnivariateFillImputer()

x_continuous = [1.0, 2.0, missing, 3.0]
x_multiclass = coerce(["y", "n", "y", missing, "y"], Multiclass)
x_count = [1, 1, 1, 2, missing, 3, 3]

mach = machine(imputer, x_continuous)
fit!(mach)

julia> fitted_params(mach)
(filler = 2.0,)

julia> transform(mach, [missing, missing, 101.0])
3-element Vector{Float64}:
 2.0
 2.0
 101.0

mach2 = machine(imputer, x_multiclass) |> fit!

julia> transform(mach2, x_multiclass)
5-element CategoricalArray{String,1,UInt32}:
 "y"
 "n"
 "y"
 "y"
 "y"

mach3 = machine(imputer, x_count) |> fit!

julia> transform(mach3, [missing, missing, 5])
3-element Vector{Int64}:
 2
 2
 5
```

For imputing tabular data, use [`FillImputer`](@ref).

"""
UnivariateFillImputer

"""
$(MLJModelInterface.doc_header(FillImputer))

Use this model to impute `missing` values in tabular data. A fixed
"filler" value is learned from the training data, one for each column
of the table.

For imputing missing values in a vector, use
[`UnivariateFillImputer`](@ref) instead.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have element scitypes `Union{Missing, T}`, where `T` is a
  subtype of `Continuous`, `Multiclass`, `OrderedFactor` or
  `Count`. Check scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `features`: a vector of names of features (symbols) for which
  imputation is to be attempted; default is empty, which is
  interpreted as "impute all".

- `continuous_fill`: function or other callable to determine value to
  be imputed in the case of `Continuous` (abstract float) data; default is to apply
  `median` after skipping `missing` values

- `count_fill`: function or other callable to determine value to
  be imputed in the case of `Count` (integer) data; default is to apply
  rounded `median` after skipping `missing` values

- `finite_fill`: function or other callable to determine value to be
  imputed in the case of `Multiclass` or `OrderedFactor` data
  (categorical vectors); default is to apply `mode` after skipping `missing` values


# Operations

- `transform(mach, Xnew)`: return `Xnew` with missing values imputed with
  the fill values learned when fitting `mach`


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features_seen_in_fit`: the names of features (columns) encountered
  during training

- `univariate_transformer`: the univariate model applied to determine
    the fillers (it's fields contain the functions defining the filler computations)

- `filler_given_feature`: dictionary of filler values, keyed on
  feature (column) names


# Examples

```
using MLJ
imputer = FillImputer()

X = (a = [1.0, 2.0, missing, 3.0, missing],
     b = coerce(["y", "n", "y", missing, "y"], Multiclass),
     c = [1, 1, 2, missing, 3])

schema(X)
julia> schema(X)
┌───────┬───────────────────────────────┐
│ names │ scitypes                      │
├───────┼───────────────────────────────┤
│ a     │ Union{Missing, Continuous}    │
│ b     │ Union{Missing, Multiclass{2}} │
│ c     │ Union{Missing, Count}         │
└───────┴───────────────────────────────┘

mach = machine(imputer, X)
fit!(mach)

julia> fitted_params(mach).filler_given_feature
(filler = 2.0,)

julia> fitted_params(mach).filler_given_feature
Dict{Symbol, Any} with 3 entries:
  :a => 2.0
  :b => "y"
  :c => 2

julia> transform(mach, X)
(a = [1.0, 2.0, 2.0, 3.0, 2.0],
 b = CategoricalValue{String, UInt32}["y", "n", "y", "y", "y"],
 c = [1, 1, 2, 2, 3],)
```

See also [`UnivariateFillImputer`](@ref).

"""
FillImputer

"""
$(MLJModelInterface.doc_header(FeatureSelector))

Use this model to select features (columns) of a table, usually as
part of a model `Pipeline`.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any table of input features, where "table" is in the sense of Tables.jl

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `features`: one of the following, with the behavior indicated:

  - `[]` (empty, the default): filter out all features (columns) which
    were not encountered in training

  - non-empty vector of feature names (symbols): keep only the
    specified features (`ignore=false`) or keep only unspecified
    features (`ignore=true`)

  - function or other callable: keep a feature if the callable returns
    `true` on its name. For example, specifying
    `FeatureSelector(features = name -> name in [:x1, :x3], ignore =
    true)` has the same effect as `FeatureSelector(features = [:x1,
    :x3], ignore = true)`, namely to select all features, with the
    exception of `:x1` and `:x3`.

- `ignore`: whether to ignore or keep specified `features`, as
  explained above


# Operations

- `transform(mach, Xnew)`: select features from the table `Xnew` as
  specified by the model, taking features seen during training into
  account, if relevant


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features_to_keep`: the features that will be selected


# Example

```
using MLJ

X = (ordinal1 = [1, 2, 3],
     ordinal2 = coerce(["x", "y", "x"], OrderedFactor),
     ordinal3 = [10.0, 20.0, 30.0],
     ordinal4 = [-20.0, -30.0, -40.0],
     nominal = coerce(["Your father", "he", "is"], Multiclass));

selector = FeatureSelector(features=[:ordinal3, ], ignore=true);

julia> transform(fit!(machine(selector, X)), X)
(ordinal1 = [1, 2, 3],
 ordinal2 = CategoricalValue{Symbol,UInt32}["x", "y", "x"],
 ordinal4 = [-20.0, -30.0, -40.0],
 nominal = CategoricalValue{String,UInt32}["Your father", "he", "is"],)

```
"""
FeatureSelector


"""
$(MLJModelInterface.doc_header(Standardizer))

Use this model to standardize (whiten) a `Continuous` vector, or
relevant columns of a table. The rescalings applied by this
transformer to new data are always those learned during the training
phase, which are generally different from what would actually
standardize the new data.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any Tables.jl compatible table or any abstract vector with
  `Continuous` element scitype (any abstract float vector). Only
  features in a table with `Continuous` scitype can be standardized;
  check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `features`: one of the following, with the behavior indicated below:

  - `[]` (empty, the default): standardize all features (columns)
    having `Continuous` element scitype

  - non-empty vector of feature names (symbols): standardize only the
    `Continuous` features in the vector (if `ignore=false`) or
    `Continuous` features *not* named in the vector (`ignore=true`).

  - function or other callable: standardize a feature if the callable
    returns `true` on its name. For example, `Standardizer(features =
    name -> name in [:x1, :x3], ignore = true, count=true)` has the
    same effect as `Standardizer(features = [:x1, :x3], ignore = true,
    count=true)`, namely to standardize all `Continuous` and `Count`
    features, with the exception of `:x1` and `:x3`.

  Note this behavior is further modified if the `ordered_factor` or `count` flags
  are set to `true`; see below

- `ignore=false`: whether to ignore or standardize specified `features`, as
  explained above

- `ordered_factor=false`: if `true`, standardize any `OrderedFactor`
  feature wherever a `Continuous` feature would be standardized, as
  described above

- `count=false`: if `true`, standardize any `Count` feature wherever a
  `Continuous` feature would be standardized, as described above


# Operations

- `transform(mach, Xnew)`: return `Xnew` with relevant features
  standardized according to the rescalings learned during fitting of
  `mach`.

- `inverse_transform(mach, Z)`: apply the inverse transformation to
  `Z`, so that `inverse_transform(mach, transform(mach, Xnew))` is
  approximately the same as `Xnew`; unavailable if `ordered_factor` or
  `count` flags were set to `true`.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features_fit` - the names of features that will be standardized

- `means` - the corresponding untransformed mean values

- `stds` - the corresponding untransformed standard deviations


# Report

The fields of `report(mach)` are:

- `features_fit`: the names of features that will be standardized


# Examples

```
using MLJ

X = (ordinal1 = [1, 2, 3],
     ordinal2 = coerce([:x, :y, :x], OrderedFactor),
     ordinal3 = [10.0, 20.0, 30.0],
     ordinal4 = [-20.0, -30.0, -40.0],
     nominal = coerce(["Your father", "he", "is"], Multiclass));

julia> schema(X)
┌──────────┬──────────────────┐
│ names    │ scitypes         │
├──────────┼──────────────────┤
│ ordinal1 │ Count            │
│ ordinal2 │ OrderedFactor{2} │
│ ordinal3 │ Continuous       │
│ ordinal4 │ Continuous       │
│ nominal  │ Multiclass{3}    │
└──────────┴──────────────────┘

stand1 = Standardizer();

julia> transform(fit!(machine(stand1, X)), X)
(ordinal1 = [1, 2, 3],
 ordinal2 = CategoricalValue{Symbol,UInt32}[:x, :y, :x],
 ordinal3 = [-1.0, 0.0, 1.0],
 ordinal4 = [1.0, 0.0, -1.0],
 nominal = CategoricalValue{String,UInt32}["Your father", "he", "is"],)

stand2 = Standardizer(features=[:ordinal3, ], ignore=true, count=true);

julia> transform(fit!(machine(stand2, X)), X)
(ordinal1 = [-1.0, 0.0, 1.0],
 ordinal2 = CategoricalValue{Symbol,UInt32}[:x, :y, :x],
 ordinal3 = [10.0, 20.0, 30.0],
 ordinal4 = [1.0, 0.0, -1.0],
 nominal = CategoricalValue{String,UInt32}["Your father", "he", "is"],)
```

See also [`OneHotEncoder`](@ref), [`ContinuousEncoder`](@ref).
"""
Standardizer


"""
$(MLJModelInterface.doc_header(UnivariateDiscretizer))

Discretization converts a `Continuous` vector into an `OrderedFactor`
vector. In particular, the output is a `CategoricalVector` (whose
reference type is optimized).

The transformation is chosen so that the vector on which the
transformer is fit has, in transformed form, an approximately uniform
distribution of values. Specifically, if `n_classes` is the level of
discretization, then `2*n_classes - 1` ordered quantiles are computed,
the odd quantiles being used for transforming (discretization) and the
even quantiles for inverse transforming.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, x)

where

- `x`: any abstract vector with `Continuous` element scitype; check
  scitype with `scitype(x)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `n_classes`: number of discrete classes in the output


# Operations

- `transform(mach, xnew)`: discretize `xnew` according to the
  discretization learned when fitting `mach`

- `inverse_transform(mach, z)`: attempt to reconstruct from `z` a
  vector that transforms to give `z`


# Fitted parameters

The fields of `fitted_params(mach).fitesult` include:

- `odd_quantiles`: quantiles used for transforming (length is `n_classes - 1`)

- `even_quantiles`: quantiles used for inverse transforming (length is `n_classes`)


# Example

```
using MLJ
using Random
Random.seed!(123)

discretizer = UnivariateDiscretizer(n_classes=100)
mach = machine(discretizer, randn(1000))
fit!(mach)

julia> x = rand(5)
5-element Vector{Float64}:
 0.8585244609846809
 0.37541692370451396
 0.6767070590395461
 0.9208844241267105
 0.7064611415680901

julia> z = transform(mach, x)
5-element CategoricalArrays.CategoricalArray{UInt8,1,UInt8}:
 0x52
 0x42
 0x4d
 0x54
 0x4e

x_approx = inverse_transform(mach, z)
julia> x - x_approx
5-element Vector{Float64}:
 0.008224506144777322
 0.012731354778359405
 0.0056265330571125816
 0.005738175684445124
 0.006835652575801987
```

"""
UnivariateDiscretizer


"""
$(MLJModelInterface.doc_header(UnivariateBoxCoxTransformer))

Box-Cox transformations attempt to make data look more normally
distributed. This can improve performance and assist in the
interpretation of models which suppose that data is
generated by a normal distribution.

A Box-Cox transformation (with shift) is of the form

    x -> ((x + c)^λ - 1)/λ

for some constant `c` and real `λ`, unless `λ = 0`, in which
case the above is replaced with

    x -> log(x + c)

Given user-specified hyper-parameters `n::Integer` and `shift::Bool`,
the present implementation learns the parameters `c` and `λ` from the
training data as follows: If `shift=true` and zeros are encountered in
the data, then `c` is set to `0.2` times the data mean.  If there are
no zeros, then no shift is applied. Finally, `n` different values of `λ`
between `-0.4` and `3` are considered, with `λ` fixed to the value
maximizing normality of the transformed data.

*Reference:* [Wikipedia entry for power
 transform](https://en.wikipedia.org/wiki/Power_transform).


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, x)

where

- `x`: any abstract vector with element scitype `Continuous`; check
  the scitype with `scitype(x)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `n=171`: number of values of the exponent `λ` to try

- `shift=false`: whether to include a preliminary constant translation
  in transformations, in the presence of zeros


# Operations

- `transform(mach, xnew)`: apply the Box-Cox transformation learned when fitting `mach`

- `inverse_transform(mach, z)`: reconstruct the vector `z` whose
  transformation learned by `mach` is `z`


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `λ`: the learned Box-Cox exponent

- `c`: the learned shift


# Examples

```
using MLJ
using UnicodePlots
using Random
Random.seed!(123)

transf = UnivariateBoxCoxTransformer()

x = randn(1000).^2

mach = machine(transf, x)
fit!(mach)

z = transform(mach, x)

julia> histogram(x)
                ┌                                        ┐
   [ 0.0,  2.0) ┤███████████████████████████████████  848
   [ 2.0,  4.0) ┤████▌ 109
   [ 4.0,  6.0) ┤█▍ 33
   [ 6.0,  8.0) ┤▍ 7
   [ 8.0, 10.0) ┤▏ 2
   [10.0, 12.0) ┤  0
   [12.0, 14.0) ┤▏ 1
                └                                        ┘
                                 Frequency

julia> histogram(z)
                ┌                                        ┐
   [-5.0, -4.0) ┤█▎ 8
   [-4.0, -3.0) ┤████████▊ 64
   [-3.0, -2.0) ┤█████████████████████▊ 159
   [-2.0, -1.0) ┤█████████████████████████████▊ 216
   [-1.0,  0.0) ┤███████████████████████████████████  254
   [ 0.0,  1.0) ┤█████████████████████████▊ 188
   [ 1.0,  2.0) ┤████████████▍ 90
   [ 2.0,  3.0) ┤██▊ 20
   [ 3.0,  4.0) ┤▎ 1
                └                                        ┘
                                 Frequency

```

"""
UnivariateBoxCoxTransformer


"""
$(MLJModelInterface.doc_header(OneHotEncoder))

Use this model to one-hot encode the `Multiclass` and `OrderedFactor`
features (columns) of some table, leaving other columns unchanged.

New data to be transformed may lack features present in the fit data,
but no *new* features can be present.

**Warning:** This transformer assumes that `levels(col)` for any
`Multiclass` or `OrderedFactor` column, `col`, is the same for
training data and new data to be transformed.

To ensure *all* features are transformed into `Continuous` features, or
dropped, use [`ContinuousEncoder`](@ref) instead.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any Tables.jl compatible table. Columns can be of mixed type
  but only those with element scitype `Multiclass` or `OrderedFactor`
  can be encoded. Check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `features`: a vector of symbols (column names). If empty (default)
  then all `Multiclass` and `OrderedFactor` features are
  encoded. Otherwise, encoding is further restricted to the specified
  features (`ignore=false`) or the unspecified features
  (`ignore=true`). This default behavior can be modified by the
  `ordered_factor` flag.

- `ordered_factor=false`: when `true`, `OrderedFactor` features are
  universally excluded

- `drop_last=true`: whether to drop the column corresponding to the
  final class of encoded features. For example, a three-class feature
  is spawned into three new features if `drop_last=false`, but just
  two features otherwise.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `all_features`: names of all features encountered in training

- `fitted_levels_given_feature`: dictionary of the levels associated
  with each feature encoded, keyed on the feature name

- `ref_name_pairs_given_feature`: dictionary of pairs `r => ftr` (such
  as `0x00000001 => :grad__A`) where `r` is a CategoricalArrays.jl
  reference integer representing a level, and `ftr` the corresponding
  new feature name; the dictionary is keyed on the names of features that
  are encoded


# Report

The fields of `report(mach)` are:

- `features_to_be_encoded`: names of input features to be encoded

- `new_features`: names of all output features


# Example

```
using MLJ

X = (name=categorical(["Danesh", "Lee", "Mary", "John"]),
     grade=categorical(["A", "B", "A", "C"], ordered=true),
     height=[1.85, 1.67, 1.5, 1.67],
     n_devices=[3, 2, 4, 3])

julia> schema(X)
┌───────────┬──────────────────┐
│ names     │ scitypes         │
├───────────┼──────────────────┤
│ name      │ Multiclass{4}    │
│ grade     │ OrderedFactor{3} │
│ height    │ Continuous       │
│ n_devices │ Count            │
└───────────┴──────────────────┘

hot = OneHotEncoder(drop_last=true)
mach = fit!(machine(hot, X))
W = transform(mach, X)

julia> schema(W)
┌──────────────┬────────────┐
│ names        │ scitypes   │
├──────────────┼────────────┤
│ name__Danesh │ Continuous │
│ name__John   │ Continuous │
│ name__Lee    │ Continuous │
│ grade__A     │ Continuous │
│ grade__B     │ Continuous │
│ height       │ Continuous │
│ n_devices    │ Count      │
└──────────────┴────────────┘
```

See also [`ContinuousEncoder`](@ref).

"""
OneHotEncoder


"""
$(MLJModelInterface.doc_header(ContinuousEncoder))

Use this model to arrange all features (columns) of a table to have
`Continuous` element scitype, by applying the following protocol to
each feature `ftr`:

- If `ftr` is already `Continuous` retain it.

- If `ftr` is `Multiclass`, one-hot encode it.

- If `ftr` is `OrderedFactor`, replace it with `coerce(ftr,
  Continuous)` (vector of floating point integers), unless
  `ordered_factors=false` is specified, in which case one-hot encode
  it.

- If `ftr` is `Count`, replace it with `coerce(ftr, Continuous)`.

- If `ftr` has some other element scitype, or was not observed in
  fitting the encoder, drop it from the table.

**Warning:** This transformer assumes that `levels(col)` for any
`Multiclass` or `OrderedFactor` column, `col`, is the same for
training data and new data to be transformed.

To selectively one-hot-encode categorical features (without dropping
columns) use [`OneHotEncoder`](@ref) instead.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any Tables.jl compatible table. Columns can be of mixed type
  but only those with element scitype `Multiclass` or `OrderedFactor`
  can be encoded. Check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `drop_last=true`: whether to drop the column corresponding to the
  final class of one-hot encoded features. For example, a three-class
  feature is spawned into three new features if `drop_last=false`, but
  two just features otherwise.

- `one_hot_ordered_factors=false`: whether to one-hot any feature
  with `OrderedFactor` element scitype, or to instead coerce it
  directly to a (single) `Continuous` feature using the order


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features_to_keep`: names of features that will not be dropped from
  the table

- `one_hot_encoder`: the `OneHotEncoder` model instance for handling
  the one-hot encoding

- `one_hot_encoder_fitresult`: the fitted parameters of the
  `OneHotEncoder` model


# Report

- `features_to_keep`: names of input features that will not be dropped
  from the table

- `new_features`: names of all output features


# Example

```julia
X = (name=categorical(["Danesh", "Lee", "Mary", "John"]),
     grade=categorical(["A", "B", "A", "C"], ordered=true),
     height=[1.85, 1.67, 1.5, 1.67],
     n_devices=[3, 2, 4, 3],
     comments=["the force", "be", "with you", "too"])

julia> schema(X)
┌───────────┬──────────────────┐
│ names     │ scitypes         │
├───────────┼──────────────────┤
│ name      │ Multiclass{4}    │
│ grade     │ OrderedFactor{3} │
│ height    │ Continuous       │
│ n_devices │ Count            │
│ comments  │ Textual          │
└───────────┴──────────────────┘

encoder = ContinuousEncoder(drop_last=true)
mach = fit!(machine(encoder, X))
W = transform(mach, X)

julia> schema(W)
┌──────────────┬────────────┐
│ names        │ scitypes   │
├──────────────┼────────────┤
│ name__Danesh │ Continuous │
│ name__John   │ Continuous │
│ name__Lee    │ Continuous │
│ grade        │ Continuous │
│ height       │ Continuous │
│ n_devices    │ Continuous │
└──────────────┴────────────┘

julia> setdiff(schema(X).names, report(mach).features_to_keep) # dropped features
1-element Vector{Symbol}:
 :comments

```

See also [`OneHotEncoder`](@ref)
"""
ContinuousEncoder


"""
$(MLJModelInterface.doc_header(UnivariateTimeTypeToContinuous))

Use this model to convert vectors with a `TimeType` element type to
vectors of `Float64` type (`Continuous` element scitype).


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, x)

where

- `x`: any abstract vector whose element type is a subtype of
  `Dates.TimeType`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `zero_time`: the time that is to correspond to 0.0 under
  transformations, with the type coinciding with the training data
  element type. If unspecified, the earliest time encountered in
  training is used.

- `step::Period=Hour(24)`: time interval to correspond to one unit
  under transformation


# Operations

- `transform(mach, xnew)`: apply the encoding inferred when `mach` was fit


# Fitted parameters

`fitted_params(mach).fitresult` is the tuple `(zero_time, step)`
actually used in transformations, which may differ from the
user-specified hyper-parameters.


# Example

```
using MLJ
using Dates

x = [Date(2001, 1, 1) + Day(i) for i in 0:4]

encoder = UnivariateTimeTypeToContinuous(zero_time=Date(2000, 1, 1),
                                         step=Week(1))

mach = machine(encoder, x)
fit!(mach)
julia> transform(mach, x)
5-element Vector{Float64}:
 52.285714285714285
 52.42857142857143
 52.57142857142857
 52.714285714285715
 52.857142
```

"""
UnivariateTimeTypeToContinuous

"""
$(MLJModelInterface.doc_header(InteractionTransformer))

Generates all polynomial interaction terms up to the given order for the subset of chosen
columns.  Any column that contains elements with scitype `<:Infinite` is a valid basis to
generate interactions.  If `features` is not specified, all such columns with scitype
`<:Infinite` in the table are used as a basis.

In MLJ or MLJBase, you can transform features `X` with the single call

    transform(machine(model), X)

See also the example below.


# Hyper-parameters

- `order`: Maximum order of interactions to be generated.
- `features`: Restricts interations generation to those columns

# Operations

- `transform(machine(model), X)`: Generates polynomial interaction terms out of table `X`
  using the hyper-parameters specified in `model`.

# Example

```
using MLJ

X = (
    A = [1, 2, 3],
    B = [4, 5, 6],
    C = [7, 8, 9],
    D = ["x₁", "x₂", "x₃"]
)
it = InteractionTransformer(order=3)
mach = machine(it)

julia> transform(mach, X)
(A = [1, 2, 3],
 B = [4, 5, 6],
 C = [7, 8, 9],
 D = ["x₁", "x₂", "x₃"],
 A_B = [4, 10, 18],
 A_C = [7, 16, 27],
 B_C = [28, 40, 54],
 A_B_C = [28, 80, 162],)

it = InteractionTransformer(order=2, features=[:A, :B])
mach = machine(it)

julia> transform(mach, X)
(A = [1, 2, 3],
 B = [4, 5, 6],
 C = [7, 8, 9],
 D = ["x₁", "x₂", "x₃"],
 A_B = [4, 10, 18],)

```

"""
InteractionTransformer
