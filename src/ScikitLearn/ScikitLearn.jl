module ScikitLearn_

#> for all Supervised models:
import MLJBase
using ScientificTypes

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

# NOTE: The rest of the module uses the @sk_model macro
# For each model, a struct needs to be given along with
# a specific metadata for target and input scitype
# and possibly an adapted clean! method

# This is what allows to read the constraint declared in @sk_model structs and
# transform it in an executable condition
# For instance if we had
#   alpha::Int = 0.5::(arg > 0.0)
# Then it would transform the `(arg > 0.0)` in `(alpha > 0.0)`
function _replace_expr!(ex, rep)
    if ex isa Expr
        for i in eachindex(ex.args)
            if ex.args[i] == :arg
                ex.args[i] = rep
            end
            _replace_expr!(ex.args[i], rep)
        end
    end
    return ex
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
    # pull out defaults and constraints
    defaults 	= Dict()
    constraints = Dict()
    stname 		= ex.args[2] isa Symbol ? ex.args[2] : ex.args[2].args[1]
    fnames 		= Symbol[]

    for i = 1:length(ex.args[3].args)
        f = ex.args[3].args[i]
        f isa LineNumberNode && continue

        fname, ftype = f.args[1] isa Symbol ?
                            (ff.args[1], :Any) :
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
    const_ex = Expr(:function, Expr(:call, stname, Expr(:parameters,
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
    fit_ex = :(function MLJBase.fit(model::$stname, verbosity::Int, X, y)
                   # body of the function
                   Xmatrix   = MLJBase.matrix(X)
                   cache     = $(Symbol(stname, "_"))($([Expr(:kw, fname, :(model.$fname))
                                                            for fname in fnames]...))
                   result    = ScikitLearn.fit!(cache, Xmatrix, y)
                   fitresult = result
                   # TODO: we may want to use the report later on
                   report    = NamedTuple()
                   return (fitresult, nothing, report)
               end)

    # clean function
    clean_ex = Expr(:function, :(MLJBase.clean!(model::$stname)),
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
        				 [Expr(:if, Expr(:call, :!, _replace_expr!(constr, :(model.$field))),
                               # action of the constraint is violated:
                               # add a message and use default for the parameter
        				       Expr(:block,
                                    :(warning *= $("constraint ($constr) failed for field $field; using default: $(defaults[field]).\n")),
                                    :(model.$field = $(defaults[field]))
                                    )
                               ) for (field, constr) in constraints]...,
                         # return full message
        				 :(return warning)
                        )
                    )
    # predict function
    predict_ex = Expr(:function, :(MLJBase.predict(model::$stname, fitresult, Xnew)),
                    # body of the predict function
        			Expr(:block,
                         :(xnew 	     = MLJBase.matrix(Xnew)),
                         :(prediction = ScikitLearn.predict(fitresult, xnew)),
                         :(return prediction)
                         )
                      )

    # model metadata note that it does not assign scitypes etc, these have
    # to be added manually model by model.
    # --> input_scitype
    # --> target_scitype
    esc(
        quote
        export $stname
        $ex
        $fit_ex
        $clean_ex
        $predict_ex
        MLJBase.load_path(::Type{<:$stname})       = string("MLJModels.ScikitLearn_.", :($stname))
        MLJBase.package_name(::Type{<:$stname})    = "ScikitLearn"
        MLJBase.package_uuid(::Type{<:$stname})    = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        MLJBase.is_pure_julia(::Type{<:$stname})   = false
        MLJBase.package_url(::Type{<:$stname})     = "https://github.com/cstjean/ScikitLearn.jl"
        MLJBase.package_license(::Type{<:$stname}) = "BSD"
        end
    )
end

include("linear-regressors.jl")
#include("linear-classifiers.jl")

include("gaussian-process.jl")
include("ensemble.jl")


end # module
