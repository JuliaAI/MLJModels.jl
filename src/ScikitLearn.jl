module ScikitLearn_

#> export the new models you're going to define (and nothing else):
export SVMClassifier, SVMRegressor
export SVMNuClassifier, SVMNuRegressor
export SVMLClassifier, SVMLRegressor

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


"""
    SVMClassifier(; kwargs...)

C-Support Vector classifier from
[https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC). Implemented hyperparameters as per
package documentation cited above.

See also, SVMNuClassifier, SVMLClassifier, SVMRegressor

"""
mutable struct SVMClassifier{Any} <: MLJBase.Deterministic{Any}
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

    model = SVMClassifier{Any}(
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
mutable struct SVMNuClassifier{Any} <: MLJBase.Deterministic{Any}
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

    model = SVMNuClassifier{Any}(
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

mutable struct SVMLClassifier{Any} <: MLJBase.Deterministic{Any}
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

    model = SVMLClassifier{Any}(
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
mutable struct SVMRegressor{Any} <: MLJBase.Deterministic{Any}
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

    model = SVMRegressor{Any}(
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

mutable struct SVMNuRegressor{Any} <: MLJBase.Deterministic{Any}
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

    model = SVMNuRegressor{Any}(
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

mutable struct SVMLRegressor{Any} <: MLJBase.Deterministic{Any}
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

    model = SVMLRegressor{Any}(
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


function MLJBase.fit(model::SVMClassifier{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)
    
    Xmatrix = MLJBase.matrix(X)
    decoder = MLJBase.CategoricalDecoder(y)
    y_plain = MLJBase.transform(decoder, y)

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
    
    result = ScikitLearn.fit!(cache,Xmatrix,y_plain)
    fitresult = (result, decoder)
    report = nothing
    
    return fitresult, cache, report 

end

function MLJBase.fit(model::SVMNuClassifier{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)
    
    Xmatrix = MLJBase.matrix(X)
    decoder = MLJBase.CategoricalDecoder(y)
    y_plain = MLJBase.transform(decoder, y)

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
    fitresult = (result, decoder)
    report = nothing
    
    return fitresult, cache, report 

end

function MLJBase.fit(model::SVMLClassifier{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)
    
    Xmatrix = MLJBase.matrix(X)
    decoder = MLJBase.CategoricalDecoder(y)
    y_plain = MLJBase.transform(decoder, y)

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
    fitresult = (result, decoder)
    report = nothing
    
    return fitresult, cache, report 

end

function MLJBase.fit(model::SVMRegressor{Any}
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
    report = nothing
    
    return fitresult, cache, report 
end

function MLJBase.fit(model::SVMNuRegressor{Any}
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
    report = nothing
    
    return fitresult, cache, report 
end

function MLJBase.fit(model::SVMLRegressor{Any}
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
    report = nothing
    
    return fitresult, cache, report 
end


#> placeholder types for predict dispatching
SVMC = Union{SVMClassifier, SVMNuClassifier, SVMLClassifier}
SVMR = Union{SVMRegressor, SVMNuRegressor, SVMLRegressor}
SVM = Union{SVMC, SVMR}

function MLJBase.predict(model::SVMC
                     , fitresult::Tuple
                     , Xnew)

    xnew = MLJBase.matrix(Xnew) 
    result, decoder = fitresult
    prediction = ScikitLearn.predict(result, xnew)
    return MLJBase.inverse_transform(decoder,prediction)
end

function MLJBase.predict(model::SVMR
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
MLJBase.input_scitypes(::Type{<:SVM}) = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:SVM}) = true
MLJBase.target_scitype(::Type{<:SVMC}) = MLJBase.Multiclass
MLJBase.target_scitype(::Type{<:SVMR}) = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:SVM}) = true

end # module
