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

import .._unpack!


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
    # pull out defaults and constraints
    defaults 	= Dict()
    constraints = Dict()
    Model 		= ex.args[2] isa Symbol ? ex.args[2] : ex.args[2].args[1]
    fnames 		= Symbol[]

    for i = 1:length(ex.args[3].args)
        f = ex.args[3].args[i]
        f isa LineNumberNode && continue

        fname, ftype = f.args[1] isa Symbol ?
                            (f.args[1], :Any) :
                            (f.args[1].args[1], f.args[1].args[2])
        push!(fnames, fname)

        if f.head == :(=)
            default = f.args[2]
            if default isa Expr
                constraints[fname] = default.args[2]
                default = default.args[1]
            end
            defaults[fname] = default
            ex.args[3].args[i] = f.args[1]
        end
    end

    # make kw constructor which calls the clean! function
    const_ex = Expr(:function, Expr(:call, Model, Expr(:parameters,
     	                  [Expr(:kw, fname, defaults[fname]) for fname in fnames]...)),
                    # body of the function
                    Expr(:block,
                         Expr(:(=), :model, Expr(:call, :new, [fname for fname in fnames]...)),
                         :(message = MLJBase.clean!(model)),
            			 :(isempty(message) || @warn message),
            			 :(return model)
        				 )
    			 	)
    push!(ex.args[3].args, const_ex)

    # add fit function
    fit_ex = :(function MLJBase.fit(model::$Model, verbosity::Int, X, y)
                   # body of the function
                   Xmatrix    = MLJBase.matrix(X)
                   yplain     = y
                   targ_names = nothing
                   # in multi-target case
                   if Tables.istable(y)
                       yplain     = MLJBase.matrix(y)
                       targ_names = MLJBase.schema(y).names
                   end
                   cache     = $(Symbol(Model, "_"))($([Expr(:kw, fname, :(model.$fname))
                                                            for fname in fnames]...))
                   result    = ScikitLearn.fit!(cache, Xmatrix, yplain)
                   fitresult = result
                   # TODO: we may want to use the report later on
                   report    = NamedTuple()
                   return ((fitresult, targ_names), nothing, report)
               end)

    # clean function
    clean_ex = Expr(:function, :(MLJBase.clean!(model::$Model)),
                    # body of the function
                    Expr(:block,
                         :(warning = ""),
                         # condition and action for each constraint
                         # each parameter is given as field::Type = default::constraint
                         # here we recuperate the constraint and express it as an if statement
                         # for instance if we had
                         #     alpha::Real = 0.0::(arg > 0.0)
                         # this would become
                         #     if !(alpha > 0.0)
        				 [Expr(:if, Expr(:call, :!, _unpack!(constr, :(model.$param))),
                               # action of the constraint is violated:
                               # add a message and use default for the parameter
        				       Expr(:block,
                                    :(warning *= $("constraint ($constr) failed; using default: $param=$(defaults[param]).\n")),
                                    :(model.$param = $(defaults[param]))
                                    )
                               ) for (param, constr) in constraints]...,
                         # return full message
        				 :(return warning)
                        )
                    )
    # predict function
    predict_ex = Expr(:function, :(MLJBase.predict(model::$Model, (fitresult, targ_names), Xnew)),
                    # body of the predict function
        			Expr(:block,
                         :(xnew  = MLJBase.matrix(Xnew)),
                         :(preds = ScikitLearn.predict(fitresult, xnew)),
                         :(isa(preds, Matrix) && (preds = MLJBase.table(preds, names=targ_names))),
                         :(return preds)
                         ) )

    # model metadata note that it does not assign scitypes etc, these have
    # to be added manually model by model.
    # --> input_scitype
    # --> target_scitype
    Model_str = string(Model)
    esc(
        quote
        export $Model
        $ex
        $fit_ex
        $clean_ex
        $predict_ex
        MLJBase.load_path(::Type{<:$Model})       = string("MLJModels.ScikitLearn_.", $Model_str)
        MLJBase.package_name(::Type{<:$Model})    = "ScikitLearn"
        MLJBase.package_uuid(::Type{<:$Model})    = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        MLJBase.is_pure_julia(::Type{<:$Model})   = false
        MLJBase.package_url(::Type{<:$Model})     = "https://github.com/cstjean/ScikitLearn.jl"
        MLJBase.package_license(::Type{<:$Model}) = "BSD"
        end
    )
end

include("linear-regressors.jl")
#include("linear-classifiers.jl")

include("gaussian-process.jl")
include("ensemble.jl")


end # module
