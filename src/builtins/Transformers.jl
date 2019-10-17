# NOTE 26/9/2019 (TL): it's very annoying but we **CANNOT** seem to
# be allowed to use @mlj_model, metadata_pkg or metadata_model here
# without it causing some form of "Method definition" warning
# or issue about an eval being called in a closed module.

module Transformers

using ..MLJBase, ..Tables
using StatsBase, Statistics, CategoricalArrays, Distributions

import ..nonmissing
import ..MLJBase: @mlj_model, metadata_pkg, metadata_model


export FeatureSelector,
        StaticTransformer,
        UnivariateDiscretizer,
        UnivariateStandardizer,
        Standardizer,
        UnivariateBoxCoxTransformer,
        OneHotEncoder,
        FillImputer

## CONSTANTS

const N_VALUES_THRESH = 16 # for BoxCoxTransformation
const CategoricalElement = Union{CategoricalValue,CategoricalString}


#### STATIC TRANSFORMERS ####

const STATIC_TRANSFORMER_DESCR = "Applies a given data transformation `f` (either a function or callable)."

"""
StaticTransformer

$STATIC_TRANSFORMER_DESCR

## Field

* `f=identity`: function or callable object to use for the data transformation.
"""
mutable struct StaticTransformer <: MLJBase.Unsupervised
    f
end
StaticTransformer(;f=identity) = StaticTransformer(f)

MLJBase.fitted_params(::StaticTransformer) = NamedTuple()
MLJBase.fit(::StaticTransformer, ::Integer, _) = nothing, nothing, NamedTuple()
MLJBase.transform(model::StaticTransformer, fitresult, Xnew) = (model.f)(Xnew)

# metadata
MLJBase.input_scitype(::Type{<:StaticTransformer})  = MLJBase.Table(MLJBase.Scientific)
MLJBase.output_scitype(::Type{<:StaticTransformer}) = MLJBase.Table(MLJBase.Scientific)
MLJBase.docstring(::Type{<:StaticTransformer})      = STATIC_TRANSFORMER_DESCR
MLJBase.load_path(::Type{<:StaticTransformer})      = "MLJModels.StaticTransformer"


#### IMPUTER ####

const FILL_IMPUTER_DESCR = "Imputes missing data with a fixed value computed on the non-missing values. The way to compute the filler depends on the scitype of the data and can be specified."

round_median(v::AbstractVector) = v -> round(eltype(v), median(v))

_median       = e -> skipmissing(e) |> median
_round_median = e -> skipmissing(e) |> (f -> round(eltype(f), median(f)))
_mode         = e -> skipmissing(e) |> mode

"""
FillImputer

$FILL_IMPUTER_DESCR

## Fields

* `continuous_fill`:  function to use on Continuous data (by default the median)
* `count_fill`:       function to use on Count data (by default the rounded median)
* `categorical_fill`: function to use on Finite data (by default the mode)
"""
mutable struct FillImputer <: MLJBase.Unsupervised
    features::Vector{Symbol}
    continuous_fill::Function
    count_fill::Function
    finite_fill::Function
end
FillImputer(; features=Symbol[], continuous_fill=_median, count_fill=_round_median, finite_fill=_mode) =
    FillImputer(features, continuous_fill, count_fill, finite_fill)

function MLJBase.fit(transformer::FillImputer, verbosity::Int, X)
    if isempty(transformer.features)
        features = Tables.schema(X).names |> collect
    else
        features = transformer.features
    end
    fitresult = features
    report    = nothing
    cache     = nothing
    return fitresult, cache, report
end

function MLJBase.transform(transformer::FillImputer, fitresult, X)
    features = Tables.schema(X).names
    # check that the features match that of the transformer
    all(e -> e in fitresult, features) ||
        error("Attempting to transform table with feature labels not seen in fit. ")

    cols = map(features) do ftr
        col = MLJBase.selectcols(X, ftr)
        if eltype(col) >: Missing
            T    = scitype_union(col)
            if T <: Union{MLJBase.Continuous,Missing}
                filler = transformer.continuous_fill(col)
            elseif T <: Union{MLJBase.Count,Missing}
                filler = transformer.count_fill(col)
            elseif T <: Union{MLJBase.Finite,Missing}
                filler = transformer.finite_fill(col)
            end
            col = copy(col) # carries the same name but not attached to the same memory
            col[ismissing.(col)] .= filler
            col = convert.(nonmissing(eltype(col)), col)
        end
        col
    end
    named_cols = NamedTuple{features}(tuple(cols...))
    return MLJBase.table(named_cols, prototype=X)
end

# metadata
MLJBase.input_scitype(::Type{<:FillImputer})  = MLJBase.Table(MLJBase.Scientific)
MLJBase.output_scitype(::Type{<:FillImputer}) = MLJBase.Table(MLJBase.Scientific)
MLJBase.docstring(::Type{<:FillImputer})      = FILL_IMPUTER_DESCR
MLJBase.load_path(::Type{<:FillImputer})      = "MLJModels.FillImputer"


## FOR FEATURE (COLUMN) SELECTION

"""
FeatureSelector(features=Symbol[])

An unsupervised model for filtering features (columns) of a table.
Only those features encountered during fitting will appear in
transformed tables if `features` is empty (the default).
Alternatively, if a non-empty `features` is specified, then only the
specified features are used. Throws an error if a recorded or
specified feature is not present in the transformation input.

"""
mutable struct FeatureSelector <: MLJBase.Unsupervised
    features::Vector{Symbol}
end
FeatureSelector(; features=Symbol[]) = FeatureSelector(features)

function MLJBase.fit(transformer::FeatureSelector, verbosity::Int, X)
    namesX = collect(Tables.schema(X).names)
    if isempty(transformer.features)
        fitresult = namesX
    else
        all(e -> e in namesX, transformer.features) ||
            throw(error("Attempting to select non-existent feature(s)."))
        fitresult = transformer.features
    end
    report = NamedTuple()
    return fitresult, nothing, report
end

MLJBase.fitted_params(::FeatureSelector, fitresult) = (features_to_keep=fitresult,)

function MLJBase.transform(transformer::FeatureSelector, features, X)
    all(e -> e in Tables.schema(X).names, features) ||
        throw(error("Supplied frame does not admit previously selected features."))
    return MLJBase.selectcols(X, features)
end

# metadata
MLJBase.input_scitype(::Type{<:FeatureSelector})  = MLJBase.Table(MLJBase.Scientific)
MLJBase.output_scitype(::Type{<:FeatureSelector}) = MLJBase.Table(MLJBase.Scientific)
MLJBase.docstring(::Type{<:FeatureSelector})      = "Filter features (columns) of a table by name."
MLJBase.load_path(::Type{<:FeatureSelector})      = "MLJModels.FeatureSelector"

#### UNIVARIATE Discretizer ####
"""

UnivariateDiscretizer(n_classes=512)
Returns a `MLJModel` for for discretising any Continuous vector v
 (scitype(v) <: AbstractVector{Continuous}), where `n_classes` describes the resolution of the
discretization. Transformed vectors are of eltype `Int46`. The
transformation is chosen so that the vector on which the transformer
is fit has, in transformed form, an approximately uniform distribution
of values.
### Example
    using MLJ
    t = UnivariateDiscretizer(n_classes=10)
    v = randn(1000)
    tM = fit(t, v)   # fit the transformer on `v`
    w = transform(tM, v) # transform `v` according to `tM`
"""
mutable struct UnivariateDiscretizer <:MLJBase.Unsupervised
    n_classes::Int
end

# lazy keyword constructor:
UnivariateDiscretizer(; n_classes=512) = UnivariateDiscretizer(n_classes)

struct UnivariateDiscretizerResult
    odd_quantiles::Vector{Float64}
    even_quantiles::Vector{Float64}
end

function MLJBase.fit(transformer::UnivariateDiscretizer, verbosity::Int,X)
    n_classes = transformer.n_classes
    quantiles = quantile(X, Array(range(0, stop=1, length=2*n_classes+1)))
    clipped_quantiles = quantiles[2:2*n_classes] # drop 0% and 100% quantiles

    # odd_quantiles for transforming, even_quantiles used for inverse_transforming:
    odd_quantiles = clipped_quantiles[2:2:(2*n_classes-2)]
    even_quantiles = clipped_quantiles[1:2:(2*n_classes-1)]
    cache = nothing
    report = NamedTuple()
    return UnivariateDiscretizerResult(odd_quantiles, even_quantiles), cache, report
end
# transforming scalars:
function MLJBase.transform(transformer::UnivariateDiscretizer, result, r::Real)
    return k = sum(r .> result.odd_quantiles)
end

#transforming vectors:
function MLJBase.transform(transformer::UnivariateDiscretizer, result,
                  v) where (scitype(v) <: AbstractVector{Continuous})
   w=[transform(transformer, result, r) for r in v]
   return categorical(w, ordered=true)
end

# scalars:
function MLJBase.inverse_transform(transformer::UnivariateDiscretizer, result, k::Integer)
   n_classes = length(result.even_quantiles)
   if k < 1
       return result.even_quantiles[1]
   elseif k > n_classes
       return result.even_quantiles[n_classes]
   end
   return result.even_quantiles[k]
end

# vectors:
function MLJBase.inverse_transform(transformer::UnivariateDiscretizer, result,
                          wcat::CategoricalArray)
    w=MLJBase.int(wcat)
   return [inverse_transform(transformer, result, k) for k in   w]
end


MLJBase.input_scitype(::Type{<:UnivariateDiscretizer})  = AbstractVector{<:MLJBase.Continuous}
MLJBase.output_scitype(::Type{<:UnivariateDiscretizer}) = AbstractVector{<:MLJBase.OrderedFactor}
MLJBase.docstring(::Type{<:UnivariateDiscretizer})      = "Discretise continuous variables via quantiles"
MLJBase.load_path(::Type{<:UnivariateDiscretizer})      = "MLJModels.UnivariateDiscretizer"








## UNIVARIATE STANDARDIZATION

"""
    UnivariateStandardizer()

Unsupervised model for standardizing (whitening) univariate data.

"""
mutable struct UnivariateStandardizer <: MLJBase.Unsupervised end

function MLJBase.fit(transformer::UnivariateStandardizer, verbosity::Int,
             v::AbstractVector{T}) where T<:Real
    std(v) > eps(Float64) ||
        @warn "Extremely small standard deviation encountered in standardization."
    fitresult = (mean(v), std(v))
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end

# for transforming single value:
function MLJBase.transform(transformer::UnivariateStandardizer, fitresult, x::Real)
    mu, sigma = fitresult
    return (x - mu)/sigma
end

# for transforming vector:
MLJBase.transform(transformer::UnivariateStandardizer, fitresult, v) =
              [transform(transformer, fitresult, x) for x in v]

# for single values:
function MLJBase.inverse_transform(transformer::UnivariateStandardizer, fitresult, y::Real)
    mu, sigma = fitresult
    return mu + y*sigma
end

# for vectors:
MLJBase.inverse_transform(transformer::UnivariateStandardizer, fitresult, w) =
    [inverse_transform(transformer, fitresult, y) for y in w]

# metadata
MLJBase.input_scitype(::Type{<:UnivariateStandardizer})  = AbstractVector{<:MLJBase.Infinite}
MLJBase.output_scitype(::Type{<:UnivariateStandardizer}) = AbstractVector{MLJBase.Continuous}
MLJBase.docstring(::Type{<:UnivariateStandardizer})      = "Standardize (whiten) univariate data."
MLJBase.load_path(::Type{<:UnivariateStandardizer})      = "MLJModels.UnivariateStandardizer"

## STANDARDIZATION OF ORDINAL FEATURES OF TABULAR DATA

"""
    Standardizer(; features=Symbol[])

Unsupervised model for standardizing (whitening) the columns of
tabular data. If `features` is empty then all columns `v` for which
all elements have `Continuous` scitypes are standardized. For
different behaviour (e.g. standardizing counts as well), specify the
names of features to be standardized.

    using DataFrames
    X = DataFrame(x1=[0.2, 0.3, 1.0], x2=[4, 2, 3])
    stand_model = Standardizer()
    transform(fit!(machine(stand_model, X)), X)

    3×2 DataFrame
    │ Row │ x1        │ x2    │
    │     │ Float64   │ Int64 │
    ├─────┼───────────┼───────┤
    │ 1   │ -0.688247 │ 4     │
    │ 2   │ -0.458831 │ 2     │
    │ 3   │ 1.14708   │ 3     │

"""
mutable struct Standardizer <: MLJBase.Unsupervised
    features::Vector{Symbol} # features to be standardized; empty means all of
end
Standardizer(; features=Symbol[]) = Standardizer(features)

function MLJBase.fit(transformer::Standardizer, verbosity::Int, X::Any)

    all_features = Tables.schema(X).names
    mach_types   = collect(eltype(selectcols(X, c)) for c in all_features)

    # determine indices of all_features to be transformed
    if isempty(transformer.features)
        cols_to_fit = filter!(eachindex(all_features)|>collect) do j
            mach_types[j] <: AbstractFloat
        end
    else
        cols_to_fit = filter!(eachindex(all_features)|>collect) do j
            all_features[j] in transformer.features && mach_types[j] <: Real
        end
    end

    fitresult_given_feature = Dict{Symbol,Tuple{Float64,Float64}}()

    # fit each feature
    verbosity < 2 || @info "Features standarized: "
    for j in cols_to_fit
        col_fitresult, cache, report =
            fit(UnivariateStandardizer(), verbosity - 1, selectcols(X, j))
        fitresult_given_feature[all_features[j]] = col_fitresult
        verbosity < 2 ||
            @info "  :$(all_features[j])    mu=$(col_fitresult[1])  sigma=$(col_fitresult[2])"
    end

    fitresult = fitresult_given_feature
    cache = nothing
    report = (features_fit=keys(fitresult_given_feature),)

    return fitresult, cache, report
end

MLJBase.fitted_params(::Standardizer, fitresult) = (mean_and_std_given_feature=fitresult,)

function MLJBase.transform(transformer::Standardizer, fitresult, X)

    # `fitresult` is dict of column fitresults, keyed on feature names

    features_to_be_transformed = keys(fitresult)

    all_features = Tables.schema(X).names

    all(e -> e in all_features, features_to_be_transformed) ||
        error("Attempting to transform data with incompatible feature labels.")

    col_transformer = UnivariateStandardizer()

    cols = map(all_features) do ftr
        if ftr in features_to_be_transformed
            transform(col_transformer, fitresult[ftr], selectcols(X, ftr))
        else
            selectcols(X, ftr)
        end
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))

    return MLJBase.table(named_cols, prototype=X)
end

# metadata
MLJBase.input_scitype(::Type{<:Standardizer})  = MLJBase.Table(MLJBase.Scientific)
MLJBase.output_scitype(::Type{<:Standardizer}) = MLJBase.Table(MLJBase.Scientific)
MLJBase.docstring(::Type{<:Standardizer})      = "Standardize (whiten) data."
MLJBase.load_path(::Type{<:Standardizer})      = "MLJModels.Standardizer"

## UNIVARIATE BOX-COX TRANSFORMATIONS

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


"""
    UnivariateBoxCoxTransformer(; n=171, shift=false)

Unsupervised model specifying a univariate Box-Cox
transformation of a single variable taking non-negative values, with a
possible preliminary shift. Such a transformation is of the form

    x -> ((x + c)^λ - 1)/λ for λ not 0
    x -> log(x + c) for λ = 0

On fitting to data `n` different values of the Box-Cox
exponent λ (between `-0.4` and `3`) are searched to fix the value
maximizing normality. If `shift=true` and zero values are encountered
in the data then the transformation sought includes a preliminary
positive shift `c` of `0.2` times the data mean. If there are no zero
values, then no shift is applied.

"""
mutable struct UnivariateBoxCoxTransformer <: MLJBase.Unsupervised
    n::Int      # nbr values tried in optimizing exponent lambda
    shift::Bool # whether to shift data away from zero
end
UnivariateBoxCoxTransformer(; n=171, shift=false) = UnivariateBoxCoxTransformer(n, shift)

function MLJBase.fit(transformer::UnivariateBoxCoxTransformer, verbosity::Int,
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

MLJBase.fitted_params(::UnivariateBoxCoxTransformer, fitresult) =
    (λ=fitresult[1], c=fitresult[2])

# for X scalar or vector:
MLJBase.transform(transformer::UnivariateBoxCoxTransformer, fitresult, X) =
    boxcox(fitresult..., X)

# scalar case:
function MLJBase.inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           fitresult, x::Real)
    lambda, c = fitresult
    if lambda == 0
        return exp(x) - c
    else
        return (lambda*x + 1)^(1/lambda) - c
    end
end

# vector case:
function MLJBase.inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           fitresult, w::AbstractVector{T}) where T <: Real
    return [inverse_transform(transformer, fitresult, y) for y in w]
end

# metadata
MLJBase.input_scitype(::Type{<:UnivariateBoxCoxTransformer})  = AbstractVector{MLJBase.Continuous}
MLJBase.output_scitype(::Type{<:UnivariateBoxCoxTransformer}) = AbstractVector{MLJBase.Continuous}
MLJBase.docstring(::Type{<:UnivariateBoxCoxTransformer})      = "Box-Cox transformation of univariate data."
MLJBase.load_path(::Type{<:UnivariateBoxCoxTransformer})      = "MLJModels.UnivariateBoxCoxTransformer"


## ONE HOT ENCODING

"""
    OneHotEncoder(; features=Symbol[], drop_last=false, ordered_factor=true)

Unsupervised model for one-hot encoding all features of `Finite`
scitype, within some table. If `ordered_factor=false` then
only `Multiclass` features are considered. The features encoded is
further restricted to those in `features`, when specified and
non-empty.

If `drop_last` is true, the column for the last level of each
categorical feature is dropped. New data to be transformed may lack
features present in the fit data, but no new features can be present.

*Warning:* This transformer assumes that the elements of a categorical
 feature in new data to be transformed point to the same
 CategoricalPool object encountered during the fit.

"""
mutable struct OneHotEncoder <: MLJBase.Unsupervised
    features::Vector{Symbol}
    drop_last::Bool
    ordered_factor::Bool
end
OneHotEncoder(; features=Symbol[], drop_last=false, ordered_factor=true) =
    OneHotEncoder(features, drop_last, ordered_factor)

# we store the categorical refs for each feature to be encoded and the
# corresponing feature labels generated (called
# "names"). `all_features` is stored to ensure no new features appear
# in new input data, causing potential name clashes.
struct OneHotEncoderResult <: MLJBase.MLJType
    all_features::Vector{Symbol} # all feature labels
    ref_name_pairs_given_feature::Dict{Symbol,Vector{Pair{<:Unsigned,Symbol}}}
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

function MLJBase.fit(transformer::OneHotEncoder, verbosity::Int, X)

    all_features = Tables.schema(X).names # a tuple not vector
    specified_features =
        isempty(transformer.features) ? collect(all_features) : transformer.features
    #
    ref_name_pairs_given_feature = Dict{Symbol,Vector{Pair{<:Unsigned,Symbol}}}()
    allowed_scitypes = ifelse(transformer.ordered_factor, Finite, Multiclass)
    col_scitypes = schema(X).scitypes
    # apply on each feature
    for j in eachindex(all_features)
        ftr = all_features[j]
        col = MLJBase.selectcols(X,j)
        T = col_scitypes[j]
        if T <: allowed_scitypes && ftr in specified_features
            ref_name_pairs_given_feature[ftr] = Pair{<:Unsigned,Symbol}[]
            shift = transformer.drop_last ? 1 : 0
            levels = MLJBase.classes(first(col))
            if verbosity > 0
                @info "Spawning $(length(levels)-shift) sub-features "*
                "to one-hot encode feature :$ftr."
            end
            for level in levels[1:end-shift]
                ref = MLJBase.int(level)
                name = compound_label(all_features, ftr, level)
                push!(ref_name_pairs_given_feature[ftr], ref => name)
            end
        end
    end
    fitresult = OneHotEncoderResult(collect(all_features), ref_name_pairs_given_feature)
    report = (features_to_be_encoded=collect(keys(ref_name_pairs_given_feature)),)
    cache = nothing
    return fitresult, cache, report
end

# If v=categorical('a', 'a', 'b', 'a', 'c') and MLJBase.int(v[1]) = ref
# then `hot(v, ref) = [true, true, false, true, false]`
hot(v::AbstractVector{<:CategoricalElement}, ref) = map(v) do c
    MLJBase.int(c) == ref
end

function MLJBase.transform(transformer::OneHotEncoder, fitresult, X)
    features = Tables.schema(X).names # tuple not vector
    d = fitresult.ref_name_pairs_given_feature
    # check the features match the fit result
    all(e -> e in fitresult.all_features, features) ||
        error("Attempting to transform table with feature labels not seen in fit. ")
    new_features = Symbol[]
    new_cols = Vector[]
    features_to_be_transformed = keys(d)
    for ftr in features
        col = MLJBase.selectcols(X, ftr)
        if ftr in features_to_be_transformed
            append!(new_features, last.(d[ftr]))
            pairs = d[ftr]
            refs = first.(pairs)
            names = last.(pairs)
            cols_to_add = map(refs) do ref
                float.(hot(col, ref))
            end
            append!(new_cols, cols_to_add)
        else
            push!(new_features, ftr)
            push!(new_cols, col)
        end
    end
    named_cols = NamedTuple{tuple(new_features...)}(tuple(new_cols)...)
    return MLJBase.table(named_cols, prototype=X)
end

# metadata
MLJBase.input_scitype(::Type{<:OneHotEncoder})  = MLJBase.Table(MLJBase.Scientific)
MLJBase.output_scitype(::Type{<:OneHotEncoder}) = MLJBase.Table(MLJBase.Scientific)
MLJBase.docstring(::Type{<:OneHotEncoder})      = "One-Hot-Encoding of the data."
MLJBase.load_path(::Type{<:OneHotEncoder})      = "MLJModels.OneHotEncoder"

#### Metadata for all built-in transformers

const BUILTIN_TRANSFORMERS = Union{
    Type{<:FeatureSelector},
    Type{<:StaticTransformer},
    Type{<:UnivariateStandardizer},
    Type{<:Standardizer},
    Type{<:UnivariateBoxCoxTransformer},
    Type{<:OneHotEncoder},
    Type{<:FillImputer}
    }

MLJBase.package_license(::BUILTIN_TRANSFORMERS) = "MIT"
MLJBase.package_name(::BUILTIN_TRANSFORMERS) = "MLJModels"
MLJBase.package_uuid(::BUILTIN_TRANSFORMERS) = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
MLJBase.package_url(::BUILTIN_TRANSFORMERS)  = "https://github.com/alan-turing-institute/MLJModels.jl"
MLJBase.is_pure_julia(::BUILTIN_TRANSFORMERS) = true
MLJBase.is_wrapper(::BUILTIN_TRANSFORMERS)    = false

end # module

using .Transformers
