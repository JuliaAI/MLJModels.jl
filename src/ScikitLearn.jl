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


#Use Multiple Dispatch to implement two versions of the fit method?
#Thus allowing use of both categorical vectors and non-categorical vectors?
#This can be reverted if necessary by simply deleting the code which handles
#the separate case.

function MLJBase.fit(model::SVMClassifier{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)

    Xmatrix = MLJBase.matrix(X)
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
    
    fitresult = ScikitLearn.fit!(cache,Xmatrix,y)

    report = nothing
    
    return fitresult, cache, report 

end

function MLJBase.fit(model::SVMClassifier{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y::CategoricalVector)
    
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
    
    fitresult = ScikitLearn.fit!(cache,Xmatrix,y)

    report = nothing
    
    return fitresult, cache, report 

end

function MLJBase.fit(model::SVMNuClassifier{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y::CategoricalVector)
    
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
    cache = LinearSVC(C=model.C,
	    loss = model.loss,
            dual=model.dual,
            penalty=model.penalty,
            intercept_scaling=model.intercept_scaling, 
            tol=model.tol,
            max_iter=model.max_iter,
            random_state=model.random_state
    )
    
    fitresult = ScikitLearn.fit!(cache,Xmatrix,y)

    report = nothing
    
    return fitresult, cache, report 

end

function MLJBase.fit(model::SVMLClassifier{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y::CategoricalVector)
    
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
    
    #Xmatrix = MLJBase.matrix(X)
    
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
    
    fitresult = ScikitLearn.fit!(cache,X,y)
    report = nothing
    
    return fitresult, cache, report 
end


function MLJBase.fit(model::SVMRegressor{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y::CategoricalVector)
    
    Xmatrix = MLJBase.matrix(X)
    decoder = MLJBase.CategoricalDecoder(y)
    y_plain = MLJBase.transform(decoder, y)
    
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
    
    result = ScikitLearn.fit!(cache,Xmatrix,y_plain)
    fitresult = (result, decoder)
    report = nothing
    
    return fitresult, cache, report 
end

function MLJBase.fit(model::SVMNuRegressor{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)
    
    #Xmatrix = MLJBase.matrix(X)
    
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
    
    fitresult = ScikitLearn.fit!(cache,X,y)
    report = nothing
    
    return fitresult, cache, report 
end


function MLJBase.fit(model::SVMNuRegressor{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y::CategoricalVector)
    
    Xmatrix = MLJBase.matrix(X)
    decoder = MLJBase.CategoricalDecoder(y)
    y_plain = MLJBase.transform(decoder, y)
    
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
    
    result = ScikitLearn.fit!(cache,Xmatrix,y_plain)
    fitresult = (result, decoder)
    report = nothing
    
    return fitresult, cache, report 
end

function MLJBase.fit(model::SVMLRegressor{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y)
    
    #Xmatrix = MLJBase.matrix(X)
    
    cache = LinearSVR(C=model.C,
            loss=model.loss,
	    fit_intercept=model.fit_intercept,
	    dual=model.dual,
	    tol=model.tol,
	    max_iter=model.max_iter,
	    epsilon=model.epsilon)
    
    fitresult = ScikitLearn.fit!(cache,X,y)
    report = nothing
    
    return fitresult, cache, report 
end


function MLJBase.fit(model::SVMLRegressor{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X
             , y::CategoricalVector)
    
    Xmatrix = MLJBase.matrix(X)
    decoder = MLJBase.CategoricalDecoder(y)
    y_plain = MLJBase.transform(decoder, y)
    
    cache = LinearSVR(C=model.C,
            loss=model.loss,
	    fit_intercept=model.fit_intercept,
	    dual=model.dual,
	    tol=model.tol,
	    max_iter=model.max_iter,
	    epsilon=model.epsilon)
    
    result = ScikitLearn.fit!(cache,Xmatrix,y_plain)
    fitresult = (result, decoder)
    report = nothing
    
    return fitresult, cache, report 
end




#> placeholder types for predict dispatching
SVMC = Union{SVMClassifier{Any}, SVMNuClassifier{Any}, SVMLClassifier{Any}}
SVMR = Union{SVMRegressor{Any}, SVMNuRegressor{Any}, SVMLRegressor{Any}}

function MLJBase.predict(model::Union{SVMC,SVMR}
                     , fitresult
                     , Xnew) 
    prediction = ScikitLearn.predict(fitresult,Xnew)
    return prediction
end

function MLJBase.predict(model::Union{SVMC,SVMR}
                     , fitresult::Tuple
                     , Xnew)

    xnew = MLJBase.matrix(Xnew) 
    plain, decoder = fitresult
    prediction = ScikitLearn.predict(plain,xnew)
    return MLJBase.inverse_transform(decoder,prediction)
end


# metadata:
MLJBase.package_name(::Type{<:SVMC}) = "ScikitLearn"
MLJBase.package_uuid(::Type{<:SVMC}) = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
MLJBase.is_pure_julia(::Type{<:SVMC}) = :no
#MLJBase.inputs_can_be(::Type{<:SVMClassifier}) = [:numeric, ]
#MLJBase.target_kind(::Type{<:SVMClassifier}) = :
#MLJBase.target_quantity(::Type{<:SVMClassifier}) = :

end # module
## EXPOSE THE INTERFACE

using .ScikitLearn_
export SVMClassifier, SVMNuClassifier, SVMLClassifier, SVMRegressor, SVMNuRegressor, SVMLRegressor    
