module ScikitLearn_

#> for all Supervised models:
import MLJBase
using ScientificTypes
using Tables

#> for all classifiers:
using CategoricalArrays

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

import .._process_model_def, .._model_constructor, .._model_cleaner
import  ..metadata_model # metadata_pkg is handled by @sk_model

const Option{T} = Union{Nothing, T}

const SKLM = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model"))
const SKGP = ((ScikitLearn.Skcore).pyimport("sklearn.gaussian_process"))
const SKEN = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble"))
const SKDU = ((ScikitLearn.Skcore).pyimport("sklearn.dummy"))
const SKNB = ((ScikitLearn.Skcore).pyimport("sklearn.naive_bayes"))
const SKNE = ((ScikitLearn.Skcore).pyimport("sklearn.neighbors"))
const SKNN = ((ScikitLearn.Skcore).pyimport("sklearn.neural_network"))

"""
_skmodel_fit

Called as part of [`@sk_model`](@ref), returns the expression corresponing to the `fit` method
for the ScikitLearn model.
"""
function _skmodel_fit(modelname, params)
	quote
		function MLJBase.fit(model::$modelname, verbosity::Int, X, y)
			# body of the function
			Xmatrix   = MLJBase.matrix(X)
			yplain    = y
			targnames = nothing
			decode    = nothing
			y1 		  = nothing
			# in multi-target regression case
			if Tables.istable(y)
			   yplain    = MLJBase.matrix(y)
			   targnames = MLJBase.schema(y).names
			end
			if eltype(y) <: CategoricalString
				yplain = MLJBase.int(y)
				y1     = y[1]
			end
			# Call the parent constructor from Sklearn.jl named Model_
			skmodel = $(Symbol(modelname, "_"))($((Expr(:kw, p, :(model.$p)) for p in params)...))
			fitres  = ScikitLearn.fit!(skmodel, Xmatrix, yplain)
			# TODO: we may want to use the report later on
			report  = NamedTuple()
			# passing y[1] is useful in the case of a classifier (decoding)
			return ((fitres, y1, targnames), nothing, report)
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
				# if it's a classifier
				return preds |> MLJBase.decoder(y1)
			end
			return preds
		end
	end
end


"""
_skmodel_predict_prob

Same as `_skmodel_predict` but with probabilities.
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


"""
    macro sk_model(ex)

Helper macro for defining interfaces ot ScikitLearn models. Struct fields require a type annotation and a default value as in the example below. Constraints for parameters (fields) are introduced as for field3 below. The constraint must refer to the parameter as `arg`. If the used parameter does not meet a constraint the default value is used.

@sk_model mutable struct SomeModel <: MLJBase.Deterministic
    field1::Int = 1
    field2::Any = nothing
    field3::Float64 = 0.5::(0 < arg < 0.8)
end

MLJBase.fit and MLJBase.predict methods are also produced.
"""
macro sk_model(ex)
	# similar to @mlj_model
    ex, modelname, params, defaults, constraints = _process_model_def(ex)
	# keyword constructor
    const_ex = _model_constructor(modelname, params, defaults)
	# associate the constructor with the definition of the struct
    push!(ex.args[3].args, const_ex)
	# cleaner
    clean_ex = _model_cleaner(modelname, defaults, constraints)

	# here starts the differences with the `@mlj_model` macro: addition of an
	# automatically defined `fit` and `predict` method
	fit_ex 	   = _skmodel_fit(modelname, params)

	if ex.args[2].args[2] == :(MLJBase.Probabilistic)
		predict_ex = _skmodel_predict_prob(modelname)
	else
		predict_ex = _skmodel_predict(modelname)
	end

    esc(
		quote
			# Base.@__doc__ $ex
	        export $modelname
	        $ex
	        $fit_ex
	        $clean_ex
	        $predict_ex
	        MLJBase.load_path(::Type{<:$modelname}) 	  = "MLJModels.ScikitLearn_.$($modelname)"
	        MLJBase.package_name(::Type{<:$modelname})    = "ScikitLearn"
	        MLJBase.package_uuid(::Type{<:$modelname})    = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
	        MLJBase.is_pure_julia(::Type{<:$modelname})   = false
	        MLJBase.package_url(::Type{<:$modelname})     = "https://github.com/cstjean/ScikitLearn.jl"
	        MLJBase.package_license(::Type{<:$modelname}) = "BSD"
	    end
	)
end

include("linear-regressors.jl")
include("linear-classifiers.jl")

include("gaussian-process.jl")
include("ensemble.jl")

include("misc.jl")

end # module
