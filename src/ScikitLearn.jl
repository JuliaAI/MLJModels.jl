module ScikitLearn_

#> export the new models you're going to define (and nothing else):
export SVMClassifier, SVMRegressor
export SVMNuClassifier, SVMNuRegressor
export SVMLClassifier, SVMLRegressor
export SCElasticNet, SCElasticNetCV

#> for all Supervised models:
import MLJBase

#> for all classifiers:
using CategoricalArrays

#> import package:
import ..ScikitLearn: @sk_import
import ..ScikitLearn
@sk_import svm: SVC
@sk_import svm: NuSVC
@sk_import svm: LinearSVC

@sk_import svm: SVR
@sk_import svm: NuSVR
@sk_import svm: LinearSVR
@sk_import linear_model : ElasticNet
@sk_import linear_model : ElasticNetCV



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


"""
   SCElasticNet(; kwargs...)
ElasticNet from
[https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html). Implemented hyperparameters as per
package documentation cited above.

"""
mutable struct SCElasticNet <: MLJBase.Deterministic
    alpha::Float64
    l1_ratio::Float64
    fit_intercept::Bool
    normalize::Bool
    precompute::Bool
    max_iter::Int
    copy_X::Bool
    tol::Float64
    warm_start::Bool
    positive::Bool
    selection::String
end

# constructor:
#> all arguments are kwargs with a default value
function SCElasticNet(
    ;alpha=1.0
    ,l1_ratio = 0.5
    ,fit_intercept = true
    ,normalize=false
    ,precompute=false
    ,max_iter=1000
    ,copy_X=true
    ,tol=0.0001
    ,warm_start=false
    ,positive=false
    ,selection="cyclic")

    model = SCElasticNet(
        alpha
        , l1_ratio
        , fit_intercept
        , normalize
        , precompute
        , max_iter
        , copy_X
        , tol
        , warm_start
        , positive
        , selection
        )

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end


function MLJBase.clean!(model::SCElasticNet)
    warning = ""
    return warning
end

function MLJBase.fit(model::SCElasticNet
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)

    cache = ElasticNet(alpha=model.alpha,
            l1_ratio=model.l1_ratio,
            fit_intercept=model.fit_intercept,
            normalize=model.normalize,
            precompute=model.precompute,
            max_iter=model.max_iter,
	    copy_X=model.copy_X,
            tol=model.tol,
            warm_start=model.warm_start,
            positive=model.positive )

    result = ScikitLearn.fit!(cache,Xmatrix,y)
    fitresult = result
	report = NamedTuple{(:n_iters,:placeholder)}((fitresult.n_iter_,nothing))
    return fitresult, nothing, report

end

function MLJBase.fitted_params(model::SCElasticNet, fitresult)
	 return NamedTuple{(:intercept,:coef)}((fitresult.intercept_,fitresult.coef_))
end


function MLJBase.predict(model::SCElasticNet
                         , fitresult
                         , Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult,xnew)
    return prediction
end


"""
   SCElasticNetCV(; kwargs...)
ElasticNetCV from
[https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html). Implemented hyperparameters as per
package documentation cited above.

"""
mutable struct SCElasticNetCV <: MLJBase.Deterministic
    l1_ratio
	eps::Float64
	n_alphas::Int
	alphas
	fit_intercept::Bool
	normalize::Bool
	precompute
	max_iter::Int
	tol::Float64
	cv
	copy_X::Bool
	positive::Bool
	selection::String
end

# constructor:
#> all arguments are kwargs with a default value
function SCElasticNetCV(
    ;l1_ratio = 0.5
	,eps=0.001
	,n_alphas=100
	,alphas=nothing
    ,fit_intercept = true
    ,normalize=false
    ,precompute=false
    ,max_iter=1000
    ,tol=0.0001
	,cv=nothing
	,copy_X=true
    ,positive=false
    ,selection="cyclic")

    model = SCElasticNetCV(
		l1_ratio
		, eps
		, n_alphas
		, alphas
		, fit_intercept
		, normalize
		, precompute
		, max_iter
		, tol
		, cv
		, copy_X
		, positive
		, selection
        )

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end


function MLJBase.clean!(model::SCElasticNetCV)
    warning = ""
    return warning
end

function MLJBase.fit(model::SCElasticNetCV
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)

    cache = ElasticNetCV(l1_ratio=model.l1_ratio
						, eps=model.eps
						, n_alphas = model.n_alphas
						, alphas = model.alphas
						, fit_intercept = model.fit_intercept
						, normalize = model.normalize
						, precompute = model.precompute
						, max_iter = model.max_iter
						, tol = model.tol
						, cv = model.cv
						, copy_X = model.copy_X
						, positive = model.positive
						, selection = model.selection
						)

    result = ScikitLearn.fit!(cache,Xmatrix,y)
    fitresult = result
    report = NamedTuple{(:l1_ratio,:alpha,:n_iters)}((fitresult.l1_ratio_,fitresult.alpha_,fitresult.n_iter_))

    return fitresult, nothing, report

end

function MLJBase.fitted_params(model::SCElasticNetCV, fitresult)
	 return NamedTuple{(:intercept,:coef)}((fitresult.intercept_,fitresult.coef_))
end

function MLJBase.predict(model::SCElasticNetCV
                         , fitresult
                         , Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult,xnew)
    return prediction
end

# metadata:
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

MLJBase.load_path(::Type{<:SCElasticNet}) = "MLJModels.ScikitLearn_.SCElasticNet"
MLJBase.package_name(::Type{<:SCElasticNet}) = "ScikitLearn"
MLJBase.package_uuid(::Type{<:SCElasticNet}) = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
MLJBase.is_pure_julia(::Type{<:SCElasticNet}) = false
MLJBase.package_url(::Type{<:SCElasticNet}) = "https://github.com/cstjean/ScikitLearn.jl"
MLJBase.input_scitype_union(::Type{<:SCElasticNet}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:SCElasticNet}) = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:SCElasticNet}) = true

MLJBase.load_path(::Type{<:SCElasticNetCV}) = "MLJModels.ScikitLearn_.SCElasticNet"
MLJBase.package_name(::Type{<:SCElasticNetCV}) = "ScikitLearn"
MLJBase.package_uuid(::Type{<:SCElasticNetCV}) = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
MLJBase.is_pure_julia(::Type{<:SCElasticNetCV}) = false
MLJBase.package_url(::Type{<:SCElasticNetCV}) = "https://github.com/cstjean/ScikitLearn.jl"
MLJBase.input_scitype_union(::Type{<:SCElasticNetCV}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:SCElasticNetCV}) = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:SCElasticNetCV}) = true
end # module
