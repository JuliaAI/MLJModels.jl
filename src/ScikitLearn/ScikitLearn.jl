module ScikitLearn_

#> for all Supervised models:
import MLJBase
import MLJBase: @mlj_model, metadata_model,
    _process_model_def, _model_constructor, _model_cleaner
using ScientificTypes
using Tables

#> for all classifiers:
using CategoricalArrays
using Tables

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

const Option{T} = Union{Nothing, T}

const SKLM = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model"))
const SKGP = ((ScikitLearn.Skcore).pyimport("sklearn.gaussian_process"))
const SKEN = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble"))
const SKDU = ((ScikitLearn.Skcore).pyimport("sklearn.dummy"))
const SKNB = ((ScikitLearn.Skcore).pyimport("sklearn.naive_bayes"))
const SKNE = ((ScikitLearn.Skcore).pyimport("sklearn.neighbors"))
const SKNN = ((ScikitLearn.Skcore).pyimport("sklearn.neural_network"))

"""
_skmodel_fit_reg

Called as part of [`@sk_reg`](@ref), returns the expression corresponing to the `fit` method
for the ScikitLearn regression model.
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
			fitres  = ScikitLearn.fit!(skmodel, Xmatrix, yplain)
			# TODO: we may want to use the report later on
			report  = NamedTuple()
			# the first nothing is so that we can use the same predict for reg and clf
			return ((fitres, nothing, targnames), nothing, report)
		end
	end
end

"""
_skmodel_fit_clf

Called as part of [`@sk_clf`](@ref), returns the expression corresponing to the `fit` method
for the ScikitLearn classifier model.
"""
function _skmodel_fit_clf(modelname, params)
	quote
		function MLJBase.fit(model::$modelname, verbosity::Int, X, y)
			Xmatrix = MLJBase.matrix(X)
			yplain  = MLJBase.int(y)
			skmodel = $(Symbol(modelname, "_"))($((Expr(:kw, p, :(model.$p)) for p in params)...))
			fitres  = ScikitLearn.fit!(skmodel, Xmatrix, yplain)
			# TODO: we may want to use the report later on
			report  = NamedTuple()
			# pass y[1] for decoding in predict method, first nothing is targnames
			return ((fitres, y[1], nothing), nothing, report)
		end
	end
end


"""
_skmodel_predict

Called as part of [`@sk_model`](@ref), returns the expression corresponing to the `predict` method
for the ScikitLearn model (for a deterministic model)
"""
function _skmodel_predict(modelname)
	quote
		function MLJBase.predict(model::$modelname, (fitresult, y1, targnames), Xnew)
			Xmatrix = MLJBase.matrix(Xnew)
			preds   = ScikitLearn.predict(fitresult, Xmatrix)
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

Same as `_skmodel_predict` but with probabilities. Note that only classifiers are probabilistic
in sklearn so that we always decode.
"""
function _skmodel_predict_prob(modelname)
	quote
		# there are no multi-task classifiers in sklearn
		function MLJBase.predict(model::$modelname, (fitresult, y1, _), Xnew)
			Xmatrix = MLJBase.matrix(Xnew)
			# this is an array of size n x c with rows that sum to 1
			preds   = ScikitLearn.predict_proba(fitresult, Xmatrix)
			classes = MLJBase.classes(y1)
			return [MLJBase.UnivariateFinite(classes, preds[i, :]) for i in 1:size(Xmatrix,1)]
		end
	end
end

# --------------------------------------------------------
# functions to help have a different macro for regressors
# and classifiers, these functions are just there to avoid
# duplication of code
# --------------------------------------------------------
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
	# call a different predict based on whether probabilistic or deteterministic
	if ex.args[2].args[2] == :(MLJBase.Probabilistic)
		predict_ex = _skmodel_predict_prob(m)
	else
		predict_ex = _skmodel_predict(m)
	end
    esc(
		quote
			# Base.@__doc__ $ex
	        export $m
	        $ex
	        $fit_ex
	        $clean_ex
	        $predict_ex
	        MLJBase.load_path(::Type{<:$m}) 	  = "MLJModels.ScikitLearn_.$($m)"
	        MLJBase.package_name(::Type{<:$m})    = "ScikitLearn"
	        MLJBase.package_uuid(::Type{<:$m})    = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
	        MLJBase.is_pure_julia(::Type{<:$m})   = false
	        MLJBase.package_url(::Type{<:$m})     = "https://github.com/cstjean/ScikitLearn.jl"
	        MLJBase.package_license(::Type{<:$m}) = "BSD"
	    end
	)
end


"""
macro sk_reg(ex)

Helper macro for defining interfaces of ScikitLearn regression models. Struct fields require a type
annotation and a default value as in the example below. Constraints for parameters (fields) are
introduced as for field3 below. The constraint must refer to the parameter as `arg`. If the used
parameter does not meet a constraint the default value is used.

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


include("linear-regressors.jl")
include("linear-classifiers.jl")
include("gaussian-process.jl")
include("ensemble.jl")
include("misc.jl")

end # module
