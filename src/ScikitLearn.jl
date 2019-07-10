module ScikitLearn_

#> export the new models you're going to define (and nothing else):
export SVMClassifier, SVMRegressor
export SVMNuClassifier, SVMNuRegressor
export SVMLClassifier, SVMLRegressor
export ElasticNet, ElasticNetCV

#> for all Supervised models:
import MLJBase

#> for all classifiers:
using CategoricalArrays


function _replace_expr!(ex, rep)
    if ex isa Expr
        for i = 1:length(ex.args)
            if ex.args[i] == :arg
                ex.args[i] = rep
            end
            _replace_expr!(ex.args[i], rep)
        end
    end
    return ex
end

"""
    macro sklmodel(ex)

Helper macro for defining interfaces ot ScikitLearn models. Struct fields require a type annotation and a default value as in the example below. Constraints for parameters (fields) are introduced as for field3 below. The constraint must refer to the parameter as `arg`. If the used parameter does not meet the constraint the default value is used.

@sklmodel mutable struct SomeModel <: MLJBase.Deterministic
    field1::Int = 1
    field2::Any = nothing
    field3::Float64 = 0.5::(0 < arg < 0.8)
end

MLJBase.fit and MLJBase.predict methods are also produced.
"""
macro sklmodel(ex)
    # pull out defaults and constraints
    defaults = Dict()
    constraints = Dict()
    stname = ex.args[2] isa Symbol ? ex.args[2] : ex.args[2].args[1]
    fnames = Symbol[]
    for i = 1:length(ex.args[3].args)
        f = ex.args[3].args[i]
        f isa LineNumberNode && continue
        fname, ftype = f.args[1] isa Symbol ? (ff.args[1], :Any) : (f.args[1].args[1], f.args[1].args[2])
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
    
    # make kw constructor
    const_ex = Expr(:function, 
        Expr(:call, stname, Expr(:parameters, [Expr(:kw, fname, defaults[fname]) for fname in fnames]...)),
        Expr(:block,
            Expr(:(=), :model, Expr(:call, :new, [fname for fname in fnames]...)),
            :(message = MLJBase.clean!(model)),
            :(isempty(message) || @warn message),
            :(return model)
        )
    )
    push!(ex.args[3].args, const_ex)
    
    # add fit method
    fit_ex = :(function MLJBase.fit(model::$stname, verbosity::Int, X, y)
        Xmatrix = MLJBase.matrix(X)
        cache = $(Symbol(stname, "_"))($([Expr(:kw, fname, :(model.$fname)) for fname in fnames]...))
        result = ScikitLearn.fit!(cache, Xmatrix, y)
        fitresult = result
        report = NamedTuple{}()
        return (fitresult, nothing, report)
    end)

    clean_ex = Expr(:function,:(MLJBase.clean!(model::$stname)),
    Expr(:block,
        :(warning = ""),
        [Expr(:if, Expr(:call, :!, _replace_expr!(c, :(model.$f))), 
        Expr(:block, 
            :(warning *= $("constraint ($c) failed for $f, using default: $(defaults[f])\n")),
            :(model.$f = $(defaults[f]))
            )) for (f,c) in constraints]...,
        :(return warning)
        )
    )

    predict_ex = Expr(:function, 
        :(MLJBase.predict(model::$stname, fitresult, Xnew)),
        Expr(:block,
            :(xnew = MLJBase.matrix(Xnew)),
            :(prediction = ScikitLearn.predict(fitresult, xnew)),
            :(return prediction)
        )
    )
    
    quote
        $(esc(ex))
        $(esc(fit_ex))
        $(esc(clean_ex))
        $(esc(predict_ex))
        MLJBase.load_path(::Type{<:$(esc(stname))}) = string($(stname))
        MLJBase.package_name(::Type{<:$(esc(stname))}) = "ScikitLearn"
        MLJBase.package_uuid(::Type{<:$(esc(stname))}) = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        MLJBase.is_pure_julia(::Type{<:$(esc(stname))}) = false
        MLJBase.package_url(::Type{<:$(esc(stname))}) = "https://github.com/cstjean/ScikitLearn.jl"
    end
end

#> import package:
import ..ScikitLearn: @sk_import
import ..ScikitLearn
@sk_import svm: SVC
@sk_import svm: NuSVC
@sk_import svm: LinearSVC
@sk_import svm: SVR
@sk_import svm: NuSVR
@sk_import svm: LinearSVR



"""
    SVMClassifier(; kwargs...)
C-Support Vector classifier from
[https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC). Implemented hyperparameters as per
package documentation cited above.
See also, SVMNuClassifier, SVMLClassifier, SVMRegressor
"""
mutable struct SVMClassifier <: MLJBase.Deterministic
    C::Float64
    kernel::Union{String,Function}
    degree::Int
    gamma::Union{Float64,String}
    coef0::Float64
    shrinking::Bool
    tol::Float64
    cache_size::Float64
    max_iter::Int
    decision_function_shape::String
    random_state
end

# constructor:
#> all arguments are kwargs with a default value
function SVMClassifier(
    ;C=1.0
    ,kernel="rbf"
    ,degree=3
    ,gamma="auto"
    ,coef0=0.0
    ,shrinking=true
    ,tol=1e-3
    ,cache_size=200
    ,max_iter=-1
    ,decision_function_shape="ovr"
    ,random_state=nothing)

    model = SVMClassifier(
        C
        , kernel
        , degree
        , gamma
        , coef0
        , shrinking
        , tol
        , cache_size
        , max_iter
        , decision_function_shape
        , random_state
        )

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end


function MLJBase.clean!(model::SVMClassifier)
    warning = ""
    if(typeof(model.kernel)==String && (!(model.kernel  in
            ["linear","poly","rbf","sigmoid","precomputed"])))
            warning *="kernel parameter is not valid, setting to default=\"rbf\" \n"
	    model.kernel="rbf"
    end
    if(typeof(model.gamma)==String && (!(model.gamma  in
            ["auto","scale"])))
            warning *="gamma parameter is not valid, setting to default=\"auto\" \n"
	    model.gamma="auto"
    end
    if(!(model.decision_function_shape in ["ovo","ovr"]))
            warning *="decision_function_shape parameter is not valid, setting to default=\"ovr\" \n"
	    model.decision_function_shape="ovr"
    end
    return warning
end

"""
    SVMNuClassifier(; kwargs...)
NU-Support Vector classifier from
[https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC). Implemented hyperparameters as per
package documentation cited above.
See also, SVMClassifier, SVMLClassifier, SVMNuRegressor
"""
mutable struct SVMNuClassifier <: MLJBase.Deterministic
    nu::Float64
    kernel::Union{String,Function}
    degree::Int
    gamma::Union{Float64,String}
    coef0::Float64
    shrinking::Bool
    tol::Float64
    cache_size::Float64
    max_iter::Int
    decision_function_shape::String
    random_state
end

# constructor:
#> all arguments are kwargs with a default value
function SVMNuClassifier(
    ;nu=0.5
    ,kernel="rbf"
    ,degree=3
    ,gamma="auto"
    ,coef0=0.0
    ,shrinking=true
    ,tol=1e-3
    ,cache_size=200
    ,max_iter=-1
    ,decision_function_shape="ovr"
    ,random_state=nothing)

    model = SVMNuClassifier(
        nu
        , kernel
        , degree
        , gamma
        , coef0
        , shrinking
        , tol
        , cache_size
        , max_iter
        , decision_function_shape
        , random_state
        )

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

function MLJBase.clean!(model::SVMNuClassifier)
    warning = ""
    if(typeof(model.kernel)==String && (!(model.kernel  in
            ["linear","poly","rbf","sigmoid","precomputed"])))
            warning *="kernel parameter is not valid, setting to default=\"rbf\" \n"
	    model.kernel="rbf"
    end
    if(typeof(model.gamma)==String && (!(model.gamma  in
            ["auto","scale"])))
            warning *="gamma parameter is not valid, setting to default=\"auto\" \n"
	    model.gamma="auto"
    end
    if(!(model.decision_function_shape in ["ovo","ovr"]))
            warning *="decision_function_shape parameter is not valid, setting to default=\"ovr\" \n"
	    model.decision_function_shape="ovr"
    end
    return warning
end

"""
    SVMLClassifier(; kwargs...)
Linear-support Vector classifier from
[https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC). Implemented hyperparameters as per
package documentation cited above.
See also, SVMClassifier, SVMNuClassifier, SVMLRegressor
"""

mutable struct SVMLClassifier <: MLJBase.Deterministic
    C::Float64
    loss::String
    dual::Bool
    penalty::String
    tol::Float64
    max_iter::Int
    intercept_scaling::Float64
    random_state
end

# constructor:
#> all arguments are kwargs with a default value


function SVMLClassifier(
    ;C=1.0
    ,loss="squared_hinge"
    ,dual=true
    ,penalty="l2"
    ,tol=1e-3
    ,max_iter=-1
    ,intercept_scaling=1.
    ,random_state=nothing)

    model = SVMLClassifier(
        C
        , loss
	, dual
	, penalty
        , tol
        , max_iter
	, intercept_scaling
        , random_state
        )

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

function MLJBase.clean!(model::SVMLClassifier)
    warning = ""
    if(!(model.loss in ["hinge","squared_hinge"]))
            warning *="loss parameter is not valid, setting to default=\"squared_hinge\" \n"
	    model.loss="squared_hinge"
    end
    if(!(model.penalty in ["l1","l2"]))
            warning *="penalty parameter is not valid, setting to default=\"l2\" \n"
	    model.penalty="l2"
    end
    return warning
end

"""
    SVMRegressor(; kwargs...)
Epsilon-Support Vector Regression from
[https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR). Implemented hyperparameters as per
package documentation cited above.
See also, SVMClassifier, SVMNuRegressor, SVMLRegressor
"""
mutable struct SVMRegressor <: MLJBase.Deterministic
    C::Float64
    kernel::Union{String,Function}
    degree::Int
    gamma::Union{Float64,String}
    coef0::Float64
    shrinking::Bool
    tol::Float64
    cache_size::Float64
    max_iter::Int
    epsilon::Float64
end

# constructor:
#> all arguments are kwargs with a default value
function SVMRegressor(
    ;C=1.0
    ,kernel="rbf"
    ,degree=3
    ,gamma="auto"
    ,coef0=0.0
    ,shrinking=true
    ,tol=1e-3
    ,cache_size=200
    ,max_iter=-1
    ,epsilon=0.1)

    model = SVMRegressor(
        C
        , kernel
        , degree
        , gamma
        , coef0
        , shrinking
        , tol
        , cache_size
        , max_iter
        , epsilon)

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

function MLJBase.clean!(model::SVMRegressor)
    warning = ""
    if(typeof(model.kernel)==String && (!(model.kernel  in
            ["linear","poly","rbf","sigmoid","precomputed"])))
            warning *="kernel parameter is not valid, setting to default=\"rbf\" \n"
	    model.kernel="rbf"
    end
    if(typeof(model.gamma)==String && (!(model.gamma  in
            ["auto","scale"])))
            warning *="gamma parameter is not valid, setting to default=\"auto\" \n"
	    model.gamma="auto"
    end
    return warning
end

"""
    SVMNuRegressor(; kwargs...)
Nu Support Vector Regression from
[https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR). Implemented hyperparameters as per
package documentation cited above.
See also, SVMNuClassifier, SVMRegressor, SVMLRegressor
"""

mutable struct SVMNuRegressor <: MLJBase.Deterministic
    nu::Float64
    C::Float64
    kernel::Union{String,Function}
    degree::Int
    gamma::Union{Float64,String}
    coef0::Float64
    shrinking::Bool
    tol::Float64
    cache_size::Float64
    max_iter::Int
end

# constructor:
#> all arguments are kwargs with a default value
function SVMNuRegressor(
    ;nu=0.5
    ,C=1.0
    ,kernel="rbf"
    ,degree=3
    ,gamma="auto"
    ,coef0=0.0
    ,shrinking=true
    ,tol=1e-3
    ,cache_size=200
    ,max_iter=-1)

    model = SVMNuRegressor(
        nu
	, C
        , kernel
        , degree
        , gamma
        , coef0
        , shrinking
        , tol
        , cache_size
        , max_iter)

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

function MLJBase.clean!(model::SVMNuRegressor)
    warning = ""
    if(typeof(model.kernel)==String && (!(model.kernel  in
            ["linear","poly","rbf","sigmoid","precomputed"])))
            warning *="kernel parameter is not valid, setting to default=\"rbf\" \n"
	    model.kernel="rbf"
    end
    if(typeof(model.gamma)==String && (!(model.gamma  in
            ["auto","scale"])))
            warning *="gamma parameter is not valid, setting to default=\"auto\" \n"
	    model.gamma="auto"
    end
    return warning
end

"""
    SVMLRegressor(; kwargs...)
Linear Support Vector Regression from
[https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR). Implemented hyperparameters as per
package documentation cited above.
See also, SVMRegressor, SVMNuRegressor, SVMLClassifier
"""

mutable struct SVMLRegressor <: MLJBase.Deterministic
    C::Float64
    loss::String
    fit_intercept::Bool
    dual::Bool
    tol::Float64
    max_iter::Int
    epsilon::Float64
end

# constructor:
#> all arguments are kwargs with a default value
function SVMLRegressor(
    ;C=1.0
    ,loss="epsilon_insensitive"
    ,fit_intercept=true
    ,dual=true
    ,tol=1e-3
    ,max_iter=-1
    ,epsilon=0.1)

    model = SVMLRegressor(
        C
	, loss
	, fit_intercept
	, dual
        , tol
        , max_iter
        , epsilon)

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

function MLJBase.clean!(model::SVMLRegressor)
    warning = ""
    if(!(model.loss in ["epsilon_insensitive","squared_epsilon_insensitive"]))
            warning *="loss parameter is not valid, setting to default=\"epsilon_insensitive\" \n"
	    model.loss="epsilon_insensitive"
    end
    return warning
end


function MLJBase.fit(model::SVMClassifier
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)

    y_plain = MLJBase.int(y)
    decode  = MLJBase.decoder(y[1]) # for predict method

    cache = SVC(C=model.C,
            kernel=model.kernel,
            degree=model.degree,
            coef0=model.coef0,
            shrinking=model.shrinking,
            gamma=model.gamma,
            tol=model.tol,
            cache_size=model.cache_size,
            max_iter=model.max_iter,
            decision_function_shape=model.decision_function_shape,
            random_state=model.random_state
    )

    result = ScikitLearn.fit!(cache, Xmatrix, y_plain)
    fitresult = (result, decode)
    report = NamedTuple()

    return fitresult, nothing, report

end

function MLJBase.fit(model::SVMNuClassifier
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)

    y_plain = MLJBase.int(y)
    decode  = MLJBase.decoder(y[1]) # for predict method

    cache = NuSVC(nu=model.nu,
            kernel=model.kernel,
            degree=model.degree,
            coef0=model.coef0,
            shrinking=model.shrinking,
	    gamma=model.gamma,
            tol=model.tol,
            cache_size=model.cache_size,
            max_iter=model.max_iter,
            decision_function_shape=model.decision_function_shape,
            random_state=model.random_state
    )

    result = ScikitLearn.fit!(cache,Xmatrix,y_plain)
    fitresult = (result, decode)
    report = NamedTuple()

    return fitresult, nothing, report

end

function MLJBase.fit(model::SVMLClassifier
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)

    y_plain = MLJBase.int(y)
    decode  = MLJBase.decoder(y[1]) # for predict method

    cache = LinearSVC(C=model.C,
	    loss = model.loss,
            dual=model.dual,
            penalty=model.penalty,
            intercept_scaling=model.intercept_scaling,
            tol=model.tol,
            max_iter=model.max_iter,
            random_state=model.random_state
    )

    result = ScikitLearn.fit!(cache,Xmatrix,y_plain)
    fitresult = (result, decode)
    report = NamedTuple()

    return fitresult, nothing, report

end

function MLJBase.fit(model::SVMRegressor
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)

    cache = SVR(C=model.C,
            kernel=model.kernel,
            degree=model.degree,
            coef0=model.coef0,
            shrinking=model.shrinking,
	    gamma=model.gamma,
            tol=model.tol,
            cache_size=model.cache_size,
            max_iter=model.max_iter,
            epsilon=model.epsilon)

    fitresult = ScikitLearn.fit!(cache,Xmatrix,y)
    report = NamedTuple()

    return fitresult, nothing, report
end

function MLJBase.fit(model::SVMNuRegressor
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)

    cache = NuSVR(nu=model.nu,
            C=model.C,
            kernel=model.kernel,
            degree=model.degree,
            coef0=model.coef0,
            shrinking=model.shrinking,
	    gamma=model.gamma,
            tol=model.tol,
            cache_size=model.cache_size,
            max_iter=model.max_iter)

    fitresult = ScikitLearn.fit!(cache,Xmatrix,y)
    report = NamedTuple()

    return fitresult, nothing, report
end

function MLJBase.fit(model::SVMLRegressor
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)

    cache = LinearSVR(C=model.C,
            loss=model.loss,
	    fit_intercept=model.fit_intercept,
	    dual=model.dual,
	    tol=model.tol,
	    max_iter=model.max_iter,
	    epsilon=model.epsilon)

    fitresult = ScikitLearn.fit!(cache,Xmatrix,y)
    report = NamedTuple()

    return fitresult, nothing, report
end


# placeholder types for predict dispatching
SVMC = Union{SVMClassifier, SVMNuClassifier, SVMLClassifier}
SVMR = Union{SVMRegressor, SVMNuRegressor, SVMLRegressor}
SVM = Union{SVMC, SVMR}

function MLJBase.predict(model::SVMC
                         , fitresult
                         , Xnew)

    xnew = MLJBase.matrix(Xnew)
    result, decode = fitresult
    prediction = ScikitLearn.predict(result, xnew)
    return decode(prediction)
end

function MLJBase.predict(model::SVMR
                         , fitresult
                         , Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult,xnew)
    return prediction
end


## METADATA

MLJBase.load_path(::Type{<:SVMClassifier}) = "MLJModels.ScikitLearn_.SVMClassifier"
MLJBase.load_path(::Type{<:SVMNuClassifier}) = "MLJModels.ScikitLearn_.SVMNuClassifier"
MLJBase.load_path(::Type{<:SVMLClassifier}) = "MLJModels.ScikitLearn_.SVMLClassifier"
MLJBase.load_path(::Type{<:SVMRegressor}) = "MLJModels.ScikitLearn_.SVMRegressor"
MLJBase.load_path(::Type{<:SVMNuRegressor}) = "MLJModels.ScikitLearn_.SVMNuRegressor"
MLJBase.load_path(::Type{<:SVMRegressor}) = "MLJModels.ScikitLearn_.SVMRegressor"
MLJBase.load_path(::Type{<:SVMLRegressor}) = "MLJModels.ScikitLearn_.SVMLRegressor"

MLJBase.package_name(::Type{<:SVM}) = "ScikitLearn"
MLJBase.package_uuid(::Type{<:SVM}) = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
MLJBase.is_pure_julia(::Type{<:SVM}) = false
MLJBase.package_url(::Type{<:SVM}) = "https://github.com/cstjean/ScikitLearn.jl"
MLJBase.input_scitype_union(::Type{<:SVM}) = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:SVM}) = true
MLJBase.target_scitype_union(::Type{<:SVMC}) = MLJBase.Finite
MLJBase.target_scitype_union(::Type{<:SVMR}) = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:SVM}) = true


################
# LINEAR MODEL #
################



ARDRegression_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).ARDRegression
@sklmodel mutable struct ARDRegression <: MLJBase.Deterministic
    n_iter::Int = 300::(arg>0)
    tol::Float64 = 0.001
    alpha_1::Float64 = 1e-6
    alpha_2::Float64 = 1e-6
    lambda_1::Float64 = 1e-6
    lambda_2::Float64 = 1e-6
    compute_score::Bool = false
    threshold_lambda::Float64 = 1.0e4
    fit_intercept::Bool = true
    normalize::Bool = false
    copy_X::Bool = true
    verbose::Bool = false
end

BayesianRidge_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).BayesianRidge
@sklmodel mutable struct BayesianRidge <: MLJBase.Deterministic
    n_iter::Int = 300::(arg>0)
    tol::Float64 = 0.001
    alpha_1::Float64 = 1e-6
    alpha_2::Float64 = 1e-6
    lambda_1::Float64 = 1e-6
    lambda_2::Float64 = 1e-6
    compute_score::Bool = false
    fit_intercept::Bool = true
    normalize::Bool = false
    copy_X::Bool = true
    verbose::Bool = false
end

ElasticNet_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).ElasticNet
@sklmodel mutable struct ElasticNet <: MLJBase.Deterministic
    alpha::Float64 = 1.0
    l1_ratio::Float64 = 0.5
    fit_intercept::Bool = true
    normalize::Bool = false
    precompute::Any = false
    max_iter::Int = 1000
    copy_X::Bool = true
    tol::Float64 = 0.0001
    warm_start::Bool = false
    positive::Bool = false
    random_state::Any = nothing
    selection::String = "cyclic"
end

ElasticNetCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).ElasticNetCV
@sklmodel mutable struct ElasticNetCV <: MLJBase.Deterministic
    l1_ratio::Union{Float64, Any} = 0.5
    eps::Float64 = 0.001
    n_alphas::Int = 100
    alphas::Any = nothing
    fit_intercept::Bool = true
    normalize::Bool = false
    precompute::Any = "auto"
    max_iter::Int = 1000
    tol::Float64 = 0.0001
    cv::Any = 5
    copy_X::Bool = true
    verbose::Union{Bool, Int} = 0
    n_jobs::Union{Int, Any} = nothing
    positive::Bool = false
    random_state::Union{Int, Nothing} = nothing
    selection::String = "cyclic"
end

HuberRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).HuberRegressor
@sklmodel mutable struct HuberRegressor <: MLJBase.Deterministic
    epsilon::Float64 = 1.35::(arg>1.0)
    max_iter::Int = 1000
    alpha::Float64 = 0.0001
    warm_start::Bool = false
    fit_intercept::Bool = true
    tol::Float64 = 1e-5
end

Lars_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).Lars
@sklmodel mutable struct Lars <: MLJBase.Deterministic
    fit_intercept::Bool = true
    verbose::Union{Bool,Int} = 0
    normalize::Bool = true
    eps::Float64 = 1e-8
    copy_X::Bool = true
    fit_path::Bool = true
    positive::Bool = false
end

LarsCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LarsCV
@sklmodel mutable struct LarsCV <: MLJBase.Deterministic
    fit_intercept::Bool = true
    verbose::Union{Bool,Int} = 0
    max_iter::Int = 500
    normalize::Bool = true
    precompute::Any = "auto"
    cv::Any = 5
    max_n_alphas::Int = 1000
    n_jobs::Union{Nothing,Int} = nothing
    eps::Float64 = 1e-8
    copy_X::Bool = true
    positive::Bool = false
end

Lasso_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).Lasso
@sklmodel mutable struct Lasso <: MLJBase.Deterministic
    alpha::Float64 = 1.0
    fit_intercept::Bool = true
    normalize::Bool = false
    precompute::Any = false
    copy_X::Bool = true
    max_iter::Int = 1000
    tol::Float64 = 0.0001
    warm_start::Bool = false
    positive::Bool = false
    random_state::Any = nothing
    selection::String = "cyclic"
end

LassoCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LassoCV
@sklmodel mutable struct LassoCV <: MLJBase.Deterministic
    eps::Float64 = 0.001
    n_alphas::Int = 100
    alphas::Any = nothing
    fit_intercept::Bool = true
    normalize::Bool = false
    precompute::Any = "auto"
    max_iter::Int = 1000
    tol::Float64 = 0.0001
    copy_X::Bool = true
    cv::Any = 5
    verbose::Union{Bool, Int} = 0
    n_jobs::Union{Int, Any} = nothing
    positive::Bool = false
    random_state::Int = nothing
    selection::String = "cyclic"
end

LassoLars_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LassoLars
@sklmodel mutable struct LassoLars <: MLJBase.Deterministic
    alpha::Float64 = 1.0
    fit_intercept::Bool = true
    verbose::Union{Bool, Int} = false
    normalize::Bool = true
    precompute::Any = "auto"
    max_iter::Int = 500
    eps::Float64 = 2.220446049250313e-16::(arg>0.0)
    copy_X::Bool = true
    fit_path::Bool = true
    positive::Any = false
end

LassoLarsCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LassoLarsCV
@sklmodel mutable struct LassoLarsCV <: MLJBase.Deterministic
    fit_intercept::Bool = true
    verbose::Union{Bool, Int} = false
    max_iter::Int = 500
    normalize::Bool = true
    precompute::Any = "auto"
    cv::Any = 5
    max_n_alphas::Int = 1000
    n_jobs::Union{Nothing,Int} = nothing
    eps::Float64 = 2.220446049250313e-16::(arg>0.0)
    copy_X::Bool = true
    positive::Any = false
end

LinearRegression_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LinearRegression
@sklmodel mutable struct LinearRegression <: MLJBase.Deterministic
    fit_intercept::Bool = true
    normalize::Bool = false
    copy_X::Bool = true
    n_jobs::Union{Int, Any} = nothing
end

MultiTaskElasticNet_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).MultiTaskElasticNet
@sklmodel mutable struct MultiTaskElasticNet <: MLJBase.Deterministic
    alpha::Float64 = 1.0
    l1_ratio::Union{Float64, Vector{Float64}} = 0.5::(0<=arg<=1)
    fit_intercept::Bool = true
    normalize::Bool = true
    copy_X::Bool = true
    max_iter::Int = 1000
    tol::Float64 = 0.0001
    warm_start::Bool = false
    random_state::Any = nothing
    selection::String = "cyclic"
end

MultiTaskElasticNetCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).MultiTaskElasticNetCV
@sklmodel mutable struct MultiTaskElasticNetCV <: MLJBase.Deterministic
    l1_ratio::Union{Float64, Vector{Float64}} = 0.5
    eps::Float64 = 1e-3
    n_alphas::Int = 100
    alphas::Any = nothing
    fit_intercept::Bool = true
    normalize::Bool = false
    max_iter::Int = 1000
    tol::Float64 = 0.0001
    cv::Int = 5
    copy_X::Bool = true
    verbose::Union{Bool, Int} = 0
    n_jobs::Union{Int, Any} = nothing
    random_state::Any = nothing
    selection::String = "cyclic"
end

MultiTaskLassoCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).MultiTaskLassoCV
@sklmodel mutable struct MultiTaskLassoCV <: MLJBase.Deterministic
    eps::Float64 = 1e-3
    n_alphas::Int = 100
    alphas::Any = nothing
    fit_intercept::Bool = true
    normalize::Bool = false
    max_iter::Int = 300
    tol::Float64 = 1e-5
    copy_X::Bool = true
    cv::Any = 5
    verbose::Union{Bool, Int} = false
    n_jobs::Union{Int, Any} = 1
    random_state::Any = nothing
    selection::String = "cyclic"
end

OrthogonalMatchingPursuit_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).OrthogonalMatchingPursuit
@sklmodel mutable struct OrthogonalMatchingPursuit <: MLJBase.Deterministic
    n_nonzero_coefs::Union{Nothing,Int} = nothing
    tol::Union{Nothing,Float64} = nothing
    fit_intercept::Bool = true
    normalize::Bool = false
    precompute::Any = "auto"
end

OrthogonalMatchingPursuitCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).OrthogonalMatchingPursuitCV
@sklmodel mutable struct OrthogonalMatchingPursuitCV <: MLJBase.Deterministic
    copy::Bool = true
    fit_intercept::Bool = true
    normalize::Bool = false
    max_iter::Union{Nothing,Int} = nothing
    cv::Any = 5
    n_jobs::Int = 1
    verbose::Union{Bool, Int} = false
end

Ridge_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).Ridge
@sklmodel mutable struct Ridge <: MLJBase.Deterministic
    alpha::Union{Float64,Vector{Float64}} = 1.0
    fit_intercept::Bool = true
    normalize::Bool = false
    copy_X::Bool = true
    max_iter::Int = 1000
    tol::Float64 = 1e-4
    solver::Any = "auto"
    random_state::Any = nothing
end

RidgeClassifier_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RidgeClassifier
@sklmodel mutable struct RidgeClassifier <: MLJBase.Deterministic
    alpha::Float64 = 1.0
    fit_intercept::Bool = true
    normalize::Bool = false
    copy_X::Bool = true
    max_iter::Int = 300
    tol::Float64 = 1e-6
    class_weight::Union{Any, Any} = nothing
    solver::String = "auto"::(arg in ("auto","svg","cholesky","lsqr","sparse_cg","sag","saga"))
    random_state::Any = nothing
end

RidgeCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RidgeCV
@sklmodel mutable struct RidgeCV <: MLJBase.Deterministic
    alphas::Any = nothing
    fit_intercept::Bool = true
    normalize::Bool = false
    scoring::Any = Nothing
    cv::Any = 5
    gcv_mode::Any = "auto"
    store_cv_values::Bool = false
end

RidgeClassifierCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RidgeClassifierCV
@sklmodel mutable struct RidgeClassifierCV <: MLJBase.Deterministic
    alphas::Any = nothing
    fit_intercept::Bool = true
    normalize::Bool = false
    scoring::Union{Nothing,String} = nothing
    cv::Any = nothing
    class_weight::Any = nothing
    store_cv_values::Bool = false
end

TheilSenRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).TheilSenRegressor
@sklmodel mutable struct TheilSenRegressor <: MLJBase.Deterministic
    fit_intercept::Bool = true
    copy_X::Bool = true
    max_subpopulation::Int = 1000
    n_subsamples::Union{Nothing,Int} = nothing
    max_iter::Int = 300
    tol::Float64 = 1e-3
    random_state::Any = nothing
    n_jobs::Int= 1
    verbose::Bool = false
end

## hand crafted methods
function MLJBase.clean!(model::T) where T <: Union{ElasticNet,ElasticNetCV}
    warning = ""
	if(model.alpha<0.0)
		warning *="alpha must be stricly positive, set to 1"
		model.alpha=1
	end
	if(model.l1_ratio!=nothing)
		for (iter,val) in enumerate(model.l1_ratio)
			if(!(0<val<=1))
				warning *="l1 must be in (0,1], set to 1"
				model.l1_ratio[iter]=1
			end
		end
	end
	return warning
end


function MLJBase.fitted_params(model::ElasticNet, fitresult)
	 return NamedTuple{(:intercept,:coef)}((fitresult.intercept_,fitresult.coef_))
end

function MLJBase.fitted_params(model::ElasticNetCV, fitresult)
	 return NamedTuple{(:intercept,:coef)}((fitresult.intercept_,fitresult.coef_))
end




GaussianProcessRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.gaussian_process")).GaussianProcessRegressor
@sklmodel mutable struct GaussianProcessRegressor <: MLJBase.Deterministic
    kernel::Any = nothing
    alpha::Union{Float64, Any} = 1.0e-10
    optimizer::Union{String, Any} = "fmin_l_bfgs_b"
    n_restarts_optimizer::Int = 0
    normalize_y::Bool = false
    copy_X_train::Bool = true
    random_state::Any = nothing
end

#### Ensemble

AdaBoostRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).AdaBoostRegressor
@sklmodel mutable struct AdaBoostRegressor <: MLJBase.Deterministic
    base_estimator::Any = nothing
    n_estimators::Int = 50
    learning_rate::Float64 = 1.0
    loss::Any = "linear"::(arg in ("linear","square","exponential"))
    random_state::Union{Nothing,Int} = nothing
end

BaggingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).BaggingRegressor
@sklmodel mutable struct BaggingRegressor <: MLJBase.Deterministic
    base_estimator::Any = nothing
    n_estimators::Int = 10
    max_samples::Union{Int, Float64} = 1.0
    max_features::Union{Int, Float64} = 1.0
    bootstrap::Bool = true
    bootstrap_features::Bool = false
    oob_score::Bool = false
    warm_start::Bool = false
    n_jobs::Union{Int, Any} = nothing
    random_state::Any = nothing
    verbose::Int = 0
end

GradientBoostingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).GradientBoostingRegressor
@sklmodel mutable struct GradientBoostingRegressor <: MLJBase.Deterministic
    loss::String = "ls"::(arg in ("ls","lad","huber","quantile"))
    learning_rate::Float64 = 0.1
    n_estimators::Int = 100
    subsample::Float64 = 1.0
    criterion::String = "friedman_mse"
    min_samples_split::Union{Int, Float64} = 2
    min_samples_leaf::Union{Int, Float64} = 1
    min_weight_fraction_leaf::Float64 = 0.0
    max_depth::Int = 3
    min_impurity_decrease::Float64 = 0.0
    min_impurity_split::Float64 = 1e-7
    init::Union{Any, Any} = nothing
    random_state::Any = nothing
    max_features::Any = Nothing
    alpha::Float64 = 0.9
    verbose::Int =0
    max_leaf_nodes::Union{Int, Any} = nothing
    warm_start::Bool = false
    presort::Union{Bool, Any} = "auto"
    validation_fraction::Float64 = 0.1
    n_iter_no_change::Union{Nothing,Int} = nothing
    tol::Float64 = 1e-4
end

# HistGradientBoostingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).HistGradientBoostingRegressor
# @sklmodel mutable struct HistGradientBoostingRegressor <: MLJBase.Deterministic
#     loss::Any = "least_squares"
#     learning_rate::Float64 = 0.1
#     max_iter::Int = 100
#     max_leaf_nodes::Int = 31
#     max_depth::Union{Int, Nothing} = nothing
#     min_samples_leaf::Int = 20
#     l2_regularization::Float64 = 0.0
#     max_bins::Int = 256
#     scoring::Any = nothing
#     validation_fraction::Union{Int, Float64, Nothing} = 0.1
#     n_iter_no_change::Union{Int, Nothing} = nothing
#     tol::Union{Float64, Any} = 1e-7
#     random_state::Any = nothing
# end

RandomForestRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).RandomForestRegressor
@sklmodel mutable struct RandomForestRegressor <: MLJBase.Deterministic
    n_estimators::Int = 10::(arg>0)
    criterion::String = "mse"
    max_depth::Union{Int, Nothing} = nothing
    min_samples_split::Union{Int, Float64} = 2
    min_samples_leaf::Union{Int, Float64} = 1
    min_weight_fraction_leaf::Float64 = 0.0
    max_features::Union{Int, Float64, String, Nothing} = "auto"
    max_leaf_nodes::Union{Int, Nothing} = nothing
    min_impurity_decrease::Float64 = 0.0
    min_impurity_split::Float64 = 1e-7
    bootstrap::Bool = true
    oob_score::Bool = false
    n_jobs::Union{Int, Nothing} = nothing
    random_state::Any = nothing
    verbose::Int = 0
    warm_start::Bool = false
end

VotingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).VotingRegressor
@sklmodel mutable struct VotingRegressor <: MLJBase.Deterministic
    estimators::Any = nothing
    weights::Any = nothing
    n_jobs::Union{Int, Any} = nothing
end





GraphicalLassoCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.covariance")).GraphicalLassoCV
@sklmodel mutable struct GraphicalLassoCV <: MLJBase.Deterministic
    alphas::Union{Nothing,Int,Vector{Int}} = nothing
    n_refinements::Int = 5::(arg>0)
    cv::Any = nothing
    tol::Union{Nothing,Float64} = nothing
    enet_tol::Union{Nothing,Float64} = nothing
    max_iter::Union{Nothing,Int} = nothing
    mode::String = "cd"::(arg in ("cd","lars"))
    n_jobs::Union{Nothing,Int} = nothing
    verbose::Bool = false
    assume_centered::Bool = false
end



# needs sense check on defaults


# RANSACRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RANSACRegressor
# mutable struct RANSACRegressor <: MLJBase.Deterministic
#     base_estimator::Any = nothing
#     min_samples::Union{Int,Float64} = 5::(arg isa Int ? arg <= 1 : !(0.0 <= arg <= 1.0))
#     residual_threshold::Union{Nothing,Float64} = nothing
#     is_data_valid::Any = nothing
#     is_model_valid::Any = nothing
#     max_trials::Union{Nothing, Int} = nothing
#     max_skips::Union{Nothing, Int} = nothing
#     stop_n_inliers::Union{Nothing, Int} = nothing
#     stop_score::Union{Nothing, Float64} = nothing
#     stop_probability::Float64 = 0.99::(0.0<=arg<=1.0)
#     loss::String = "absolute_loss"::(arg in ("absolute_loss","squared_loss"))
#     random_state::Any = nothing
# end


## METATDATA

MLJBase.load_path(::Type{<:ElasticNet}) = "MLJModels.ScikitLearn_.ElasticNet"
MLJBase.package_name(::Type{<:ElasticNet}) = "ScikitLearn"
MLJBase.package_uuid(::Type{<:ElasticNet}) = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
MLJBase.is_pure_julia(::Type{<:ElasticNet}) = false
MLJBase.package_url(::Type{<:ElasticNet}) = "https://github.com/cstjean/ScikitLearn.jl"
MLJBase.input_scitype_union(::Type{<:ElasticNet}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:ElasticNet}) = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:ElasticNet}) = true

MLJBase.load_path(::Type{<:ElasticNetCV}) = "MLJModels.ScikitLearn_.ElasticNetCV"
MLJBase.package_name(::Type{<:ElasticNetCV}) = "ScikitLearn"
MLJBase.package_uuid(::Type{<:ElasticNetCV}) = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
MLJBase.is_pure_julia(::Type{<:ElasticNetCV}) = false
MLJBase.package_url(::Type{<:ElasticNetCV}) = "https://github.com/cstjean/ScikitLearn.jl"
MLJBase.input_scitype_union(::Type{<:ElasticNetCV}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:ElasticNetCV}) = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:ElasticNetCV}) = true


end # module
