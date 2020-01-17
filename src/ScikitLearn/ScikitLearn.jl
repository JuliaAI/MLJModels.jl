module ScikitLearn_

import MLJBase
import MLJBase: @mlj_model, metadata_model,
                _process_model_def, _model_constructor, _model_cleaner
using ScientificTypes, Tables, CategoricalArrays

# NOTE: legacy code for SVM models does not use the @sk_model macro.

#> import package:
import ..ScikitLearn: @sk_import
import ..ScikitLearn
@sk_import svm: SVC
@sk_import svm: NuSVC
@sk_import svm: LinearSVC
@sk_import svm: SVR
@sk_import svm: NuSVR
@sk_import svm: LinearSVR

include("svm.jl")

const SK = ScikitLearn

const Option{T} = Union{Nothing, T}

const SKLM = ((SK.Skcore).pyimport("sklearn.linear_model"))
const SKGP = ((SK.Skcore).pyimport("sklearn.gaussian_process"))
const SKEN = ((SK.Skcore).pyimport("sklearn.ensemble"))
const SKDU = ((SK.Skcore).pyimport("sklearn.dummy"))
const SKNB = ((SK.Skcore).pyimport("sklearn.naive_bayes"))
const SKNE = ((SK.Skcore).pyimport("sklearn.neighbors"))
const SKDA = ((SK.Skcore).pyimport("sklearn.discriminant_analysis"))
# const SKNN = ((SK.Skcore).pyimport("sklearn.neural_network"))

const SKCL = ((SK.Skcore).pyimport("sklearn.cluster"))

# ============================================================================

"""
macro sk_reg(ex)

Helper macro for defining interfaces of ScikitLearn regression models.
Struct fields require a type annotation and a default value as in the example
below. Constraints for parameters (fields) are introduced as for field3 below.
The constraint must refer to the parameter as `arg`. If the used parameter does
not meet a constraint the default value is used.

@sk_reg mutable struct SomeRegression <: MLJBase.Deterministic
    field1::Int = 1
    field2::Any = nothing
    field3::Float64 = 0.5::(0 < arg < 0.8)
end

MLJBase.fit and MLJBase.predict methods are also produced. See also [`@sk_clf`](@ref)
"""
macro sk_reg(ex)
    modelname, params, clean_ex, ex = _sk_constructor(ex)
    fit_ex = _skmodel_fit_reg(modelname, params)
    _sk_finalize(modelname, clean_ex, fit_ex, ex)
end

"""
macro sk_clf(ex)

Same as [`@sk_reg`](@ref) but for classifiers.
"""
macro sk_clf(ex)
    modelname, params, clean_ex, ex = _sk_constructor(ex)
    fit_ex = _skmodel_fit_clf(modelname, params)
    _sk_finalize(modelname, clean_ex, fit_ex, ex)
end

"""
macro sk_uns(ex)

Same as [`@sk_reg`](@ref) but for unsupervised models.
"""
macro sk_uns(ex)
    modelname, params, clean_ex, ex = _sk_constructor(ex)
    fit_ex = _skmodel_fit_uns(modelname, params)
    _sk_finalize(modelname, clean_ex, fit_ex, ex)
end

# ============================================================================

function _sk_constructor(ex)
    # similar to @mlj_model
    ex, modelname, params, defaults, constraints = _process_model_def(ex)
    # keyword constructor
    const_ex = _model_constructor(modelname, params, defaults)
    # associate the constructor with the definition of the struct
    push!(ex.args[3].args, const_ex)
    # cleaner
    clean_ex = _model_cleaner(modelname, defaults, constraints)
    # return
    return modelname, params, clean_ex, ex
end

function _sk_finalize(m, clean_ex, fit_ex, ex)
    # call a different predict based on whether probabilistic, deteterministic
    # or unsupervised
    if ex.args[2].args[2] == :(MLJBase.Deterministic)
        predict_ex = _skmodel_predict(m)
    elseif ex.args[2].args[2] == :(MLJBase.Probabilistic)
        predict_ex = _skmodel_predict_prob(m)
    else
        predict_ex = nothing
    end
    esc(
        quote
            export $m
            $ex
            $fit_ex
            $clean_ex
            $predict_ex
            MLJBase.load_path(::Type{<:$m})       = "MLJModels.ScikitLearn_.$(MLJBase.name($m))"
            MLJBase.package_name(::Type{<:$m})    = "ScikitLearn"
            MLJBase.package_uuid(::Type{<:$m})    = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
            MLJBase.is_pure_julia(::Type{<:$m})   = false
            MLJBase.package_url(::Type{<:$m})     = "https://github.com/cstjean/ScikitLearn.jl"
            MLJBase.package_license(::Type{<:$m}) = "BSD"
        end
    )
end

# ============================================================================
# Specifics for SUPERVISED MODELS
# ============================================================================

"""
_skmodel_fit_reg

Called as part of [`@sk_reg`](@ref), returns the expression corresponing to the
`fit` method for a ScikitLearn regression model.
"""
function _skmodel_fit_reg(modelname, params)
    quote
        function MLJBase.fit(model::$modelname, verbosity::Int, X, y)
            # set X and y into a format that can be processed by sklearn
            Xmatrix   = MLJBase.matrix(X)
            yplain    = y
            targnames = nothing
            # check if it's a multi-target regression case, in that case keep
            # track of the names of the target columns so that the prediction
            # can be named accordingly
            if Tables.istable(y)
               yplain    = MLJBase.matrix(y)
               targnames = Tables.schema(y).names
            end
            # Call the parent constructor from Sklearn.jl named $modelname_
            skmodel = $(Symbol(modelname, "_"))($((Expr(:kw, p, :(model.$p)) for p in params)...))
            fitres  = SK.fit!(skmodel, Xmatrix, yplain)
            # TODO: we may want to use the report later on
            report  = NamedTuple()
            # the first nothing is so that we can use the same predict for reg and clf
            return ((fitres, nothing, targnames), nothing, report)
        end
    end
end

"""
_skmodel_fit_clf

Called as part of [`@sk_clf`](@ref), returns the expression corresponing to the
`fit` method for a ScikitLearn classifier model.
"""
function _skmodel_fit_clf(modelname, params)
    quote
        function MLJBase.fit(model::$modelname, verbosity::Int, X, y)
            Xmatrix = MLJBase.matrix(X)
            yplain  = MLJBase.int(y)
            skmodel = $(Symbol(modelname, "_"))($((Expr(:kw, p, :(model.$p)) for p in params)...))
            fitres  = SK.fit!(skmodel, Xmatrix, yplain)
            # TODO: we may want to use the report later on
            report  = NamedTuple()
            # pass y[1] for decoding in predict method, first nothing is targnames
            return ((fitres, y[1], nothing), nothing, report)
        end
    end
end

"""
_skmodel_predict

Called as part of [`@sk_model`](@ref), returns the expression corresponing to
the `predict` method for the ScikitLearn model (for a deterministic model).
"""
function _skmodel_predict(modelname)
    quote
        function MLJBase.predict(model::$modelname, (fitresult, y1, targnames), Xnew)
            Xmatrix = MLJBase.matrix(Xnew)
            preds   = SK.predict(fitresult, Xmatrix)
            if isa(preds, Matrix)
                # only regressors are possibly multitarget;
                # build a table with the appropriate column names
                return MLJBase.table(preds, names=targnames)
            end
            if y1 !== nothing
                # if it's a classifier)
                return preds |> MLJBase.decoder(y1)
            end
            return preds
        end
    end
end

"""
_skmodel_predict_prob

Same as `_skmodel_predict` but with probabilities. Note that only classifiers
are probabilistic in sklearn so that we always decode.
"""
function _skmodel_predict_prob(modelname)
    quote
        # there are no multi-task classifiers in sklearn
        function MLJBase.predict(model::$modelname, (fitresult, y1, _), Xnew)
            Xmatrix = MLJBase.matrix(Xnew)
            # this is an array of size n x c with rows that sum to 1
            preds   = SK.predict_proba(fitresult, Xmatrix)
            classes = MLJBase.classes(y1)
            return [MLJBase.UnivariateFinite(classes, preds[i, :]) for i in 1:size(Xmatrix,1)]
        end
    end
end

# ============================================================================
# Specifics for UNSUPERVISED MODELS
# ============================================================================
# Depending on the model there may be
# * a transform
# * a inverse_transform
# * a predict

"""
_skmodel_fit_uns

Called as part of [`@sk_uns`](@ref), returns the expression corresponing to the
`fit` method for a ScikitLearn unsupervised model.
"""
function _skmodel_fit_uns(modelname, params)
    quote
        function MLJBase.fit(model::$modelname, verbosity::Int, X)
            Xmatrix = MLJBase.matrix(X)
            skmodel = $(Symbol(modelname, "_"))($((Expr(:kw, p, :(model.$p)) for p in params)...))
            fitres  = SK.fit!(skmodel, Xmatrix)

            sm = $(Symbol(modelname))
            # TODO: we may want to use the report later on
            report = NamedTuple()
            return (fitres, nothing, report)
        end
    end
end

"""
macro sku_tranfsorm

Adds a `transform` method to a declared scikit unsupervised model if
there is one supported.
"""
macro sku_transform(modelname)
    esc(
        quote
            function MLJBase.transform(::$modelname, fitres, X)
                X = SK.transform(fitres, MLJBase.matrix(X))
                MLJBase.table(X)
            end
        end
    )
end

"""
macro sku_inverse_tranfsorm

Adds an `inverse_transform` method to a declared scikit unsupervised model if
there is one supported.
"""
macro sku_inverse_transform(modelname)
    esc(
        quote
            function MLJBase.inverse_transform(::$modelname, fitres, X)
                X = SK.inverse_transform(fitres, MLJBase.matrix(X))
                MLJBase.table(X)
            end
        end
    )
end

"""
macro sku_predict

Adds a `predict` method to a declared scikit unsupervised model if
there is one supported. Returns a categorical vector.
Only `AffinityPropagation`, `Birch`, `KMeans`, `MiniBatchKMeans` and
`MeanShift` support a `predict` method.
Note: for models that offer a `fit_predict`, the encoding is done in the
`fitted_params`.
"""
macro sku_predict(modelname)
    esc(
        quote
            function MLJBase.predict(m::$modelname, fitres, X)
                # this is due to the fact that we have nested modules
                # so potentially have to extract the leaf node...
                sm = Symbol($modelname)
                ss = string(sm)
                sm = Symbol(split(ss, ".")[end])
                if sm in (:Birch, :KMeans, :MiniBatchKMeans)
                    catv = categorical(1:m.n_clusters)
                elseif sm == :AffinityPropagation
                    nc   = length(fitres.cluster_centers_indices_)
                    catv = categorical(1:nc)
                elseif sm == :MeanShift
                    nc   = size(fitres.cluster_centers_, 1)
                    catv = categorical(1:nc)
                else
                    throw(ArgumentError("Model $sm does not support `predict`."))
                end
                preds  = SK.predict(fitres, MLJBase.matrix(X)) .+ 1
                return catv[preds]
            end
        end
    )
end

# ============================================================================

include("linear-regressors.jl")
include("linear-classifiers.jl")
include("gaussian-process.jl")
include("ensemble.jl")
include("discriminant-analysis.jl")
include("misc.jl")

include("clustering.jl")

end # module
