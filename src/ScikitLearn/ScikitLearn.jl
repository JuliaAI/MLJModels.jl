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
			# in multi-target case
			if Tables.istable(y)
			   yplain    = MLJBase.matrix(y)
			   targnames = MLJBase.schema(y).names
			end
			# Call the parent constructor from Sklearn.jl named Model_
			skmodel = $(Symbol(modelname, "_"))($((Expr(:kw, p, :(model.$p)) for p in params)...))
			fitres  = ScikitLearn.fit!(skmodel, Xmatrix, yplain)
			# TODO: we may want to use the report later on
			report  = NamedTuple()
			return ((fitres, targnames), nothing, report)
		end
	end
end


"""
_skmodel_predict

Called as part of [`@sk_model`](@ref), returns the expression corresponing to the `predict` method
for the ScikitLearn model.
"""
function _skmodel_predict(modelname)
	quote
		function MLJBase.predict(model::$modelname, (fitresult, targnames), Xnew)
			Xmatrix = MLJBase.matrix(Xnew)
			preds   = ScikitLearn.predict(fitresult, Xmatrix)
			if isa(preds, Matrix)
				# build a table with the appropriate column names
				preds = MLJBase.table(preds, names=targnames)
			end
			return preds
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
	predict_ex = _skmodel_predict(modelname)

	mdl = modelname
    esc(
		quote
			# Base.@__doc__ $ex
	        export $modelname
	        $ex
	        $fit_ex
	        $clean_ex
	        $predict_ex
	        MLJBase.load_path(::Type{<:$mdl}) 		= "MLJModels.ScikitLearn_.$mdl"
	        MLJBase.package_name(::Type{<:$mdl})    = "ScikitLearn"
	        MLJBase.package_uuid(::Type{<:$mdl})    = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
	        MLJBase.is_pure_julia(::Type{<:$mdl})   = false
	        MLJBase.package_url(::Type{<:$mdl})     = "https://github.com/cstjean/ScikitLearn.jl"
	        MLJBase.package_license(::Type{<:$mdl}) = "BSD"
	    end
	)
end

include("linear-regressors.jl")
#include("linear-classifiers.jl")

include("gaussian-process.jl")
include("ensemble.jl")


end # module
