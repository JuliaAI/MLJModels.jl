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
MLJBase.is_pure_julia(::Type{<:SVM}) = :no
MLJBase.package_url(::Type{<:SVM}) = "https://github.com/cstjean/ScikitLearn.jl"
MLJBase.input_kinds(::Type{<:SVM}) = [:continuous, ]
MLJBase.input_quantity(::Type{<:SVM}) = :multivariate
MLJBase.output_kind(::Type{<:SVMC}) = :multiclass
MLJBase.output_kind(::Type{<:SVMR}) = :continuous
MLJBase.output_quantity(::Type{<:SVM}) = :univariate


end # module
