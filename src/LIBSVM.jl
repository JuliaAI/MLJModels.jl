module LIBSVM_

#> export the new models you're going to define (and nothing else):
export LinearSVC, SVC
export NuSVC, NuSVR
export EpsilonSVR
export OneClassSVM

#> for all Supervised models:
import MLJBase

#> for all classifiers:
using CategoricalArrays

#> import package:
import ..LIBSVM

"""
    LinearSVC(; kwargs...)

Linear support vector machine classifier using LIBLINEAR: https://www.csie.ntu.edu.tw/~cjlin/liblinear/

See also SVC, NuSVC
"""
mutable struct LinearSVC <: MLJBase.Deterministic
    solver::LIBSVM.Linearsolver.LINEARSOLVER
    weights::Union{Dict, Nothing}
    tolerance::Float64
    cost::Float64
    p::Float64
    bias::Float64
end

function LinearSVC(
    ;solver::LIBSVM.Linearsolver.LINEARSOLVER = LIBSVM.Linearsolver.L2R_L2LOSS_SVC_DUAL
    ,weights::Union{Dict, Nothing} = nothing
    ,tolerance::Float64 = Inf
    ,cost::Float64 = 1.0
    ,p::Float64 = 0.1
    ,bias::Float64= -1.0)

    model = LinearSVC(
        solver
        ,weights
        ,tolerance
        ,cost
        ,p
        ,bias
    )

    message = MLJBase.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

"""
    SVC(; kwargs...)

Kernel support vector machine classifier using LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

If `gamma==-1.0` then an automatically computed value is used in
fitting. Use the `report` method to inspect value used.

See also LinearSVC, NuSVC
"""
mutable struct SVC <: MLJBase.Deterministic
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    weights::Union{Dict, Nothing}
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    probability::Bool
end

function SVC(
    ;kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = -1.0
    ,weights::Union{Dict, Nothing} = nothing
    ,cost::Float64 = 1.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.0
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true
    ,probability::Bool = false)

    model = SVC(
        kernel
        ,gamma
        ,weights
        ,cost
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
        ,probability
    )

    message = MLJBase.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

"""
    NuSVC(; kwargs...)

Kernel support vector machine classifier using LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

If `gamma==-1.0` then an automatically computed value is used in
fitting. Use the `report` method to inspect value used.

See also LinearSVC, SVC
"""
mutable struct NuSVC <: MLJBase.Deterministic
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    weights::Union{Dict, Nothing}
    nu::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function NuSVC(
    ;kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = -1.0
    ,weights::Union{Dict, Nothing} = nothing
    ,nu::Float64 = 0.5
    ,cost::Float64 = 1.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = NuSVC(
        kernel
        ,gamma
        ,weights
        ,nu
        ,cost
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MLJBase.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

mutable struct OneClassSVM <: MLJBase.Unsupervised
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    nu::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function OneClassSVM(
    ;kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = -1.0
    ,nu::Float64 = 0.1
    ,cost::Float64 = 1.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.0
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = OneClassSVM(
        kernel
        ,gamma
        ,nu
        ,cost
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MLJBase.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

"""
    NuSVR(; kwargs...)

Kernel support vector machine regressor using LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

See also EpsilonSVR
"""
mutable struct NuSVR <: MLJBase.Deterministic
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    nu::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function NuSVR(
    ;kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = -1.0
    ,nu::Float64 = 0.5
    ,cost::Float64 = 1.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = NuSVR(
        kernel
        ,gamma
        ,nu
        ,cost
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MLJBase.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

"""
    EpsilonSVR(; kwargs...)

Kernel support vector machine regressor using LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

See also NuSVR
"""
mutable struct EpsilonSVR <: MLJBase.Deterministic
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    epsilon::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function EpsilonSVR(
    ;kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = -1.0
    ,epsilon::Float64 = 0.1
    ,cost::Float64 = 1.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = EpsilonSVR(
        kernel
        ,gamma
        ,epsilon
        ,cost
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MLJBase.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end


const SVM = Union{LinearSVC, SVC, NuSVC, NuSVR, EpsilonSVR, OneClassSVM} # all SVM models defined here


"""
    map_model_type(model::SVM)

Helper function to map the model to the correct LIBSVM model type needed for function dispatch.
"""
function map_model_type(model::SVM)
    if isa(model, LinearSVC)
        return LIBSVM.LinearSVC
    elseif isa(model, SVC)
        return LIBSVM.SVC
    elseif isa(model, NuSVC)
        return LIBSVM.NuSVC
    elseif isa(model, NuSVR)
        return LIBSVM.NuSVR
    elseif isa(model, EpsilonSVR)
        return LIBSVM.EpsilonSVR
    elseif isa(model, OneClassSVM)
        return LIBSVM.OneClassSVM
    else
        error("Got unsupported model type: $(typeof(model))")
    end
end

"""
    get_svm_parameters(model::Union{SVC, NuSVC, NuSVR, EpsilonSVR, OneClassSVM})

Helper function to get the parameters from the SVM model struct.
"""
function get_svm_parameters(model::Union{SVC, NuSVC, NuSVR, EpsilonSVR, OneClassSVM})
    #Build arguments for calling svmtrain
    params = Tuple{Symbol, Any}[]
    push!(params, (:svmtype, map_model_type(model))) # get LIBSVM model type
    for fn in fieldnames(typeof(model))
        push!(params, (fn, getfield(model, fn)))
    end

    return params
end


function MLJBase.fit(model::LinearSVC, verbosity::Int, X, y)

    Xmatrix = MLJBase.matrix(X)' # notice the transpose
    y_plain = MLJBase.int(y)
    decode  = MLJBase.decoder(y[1]) # for predict method

    cache = nothing

    result = LIBSVM.LIBLINEAR.linear_train(y_plain, Xmatrix,
        weights = model.weights, solver_type = Int32(model.solver),
        C = model.cost, p = model.p, bias = model.bias,
        eps = model.tolerance, verbose = ifelse(verbosity > 1, true, false)
    )

    fitresult = (result, decode)
    report = nothing

    return fitresult, cache, report
end

function MLJBase.fit(model::Union{SVC, NuSVC}, verbosity::Int, X, y)

    Xmatrix = MLJBase.matrix(X)' # notice the transpose
    y_plain = MLJBase.int(y)
    decode  = MLJBase.decoder(y[1]) # for predict method

    cache = nothing

    model = deepcopy(model)
    model.gamma == -1.0 && (model.gamma = 1.0/size(Xmatrix, 1))
    result = LIBSVM.svmtrain(Xmatrix, y_plain;
        get_svm_parameters(model)...,
        verbose = ifelse(verbosity > 1, true, false)
    )

    fitresult = (result, decode)
    report = (gamma=model.gamma,)

    return fitresult, cache, report
end

function MLJBase.fit(model::Union{NuSVR, EpsilonSVR}, verbosity::Int, X, y)

    Xmatrix = MLJBase.matrix(X)' # notice the transpose

    cache = nothing

    model = deepcopy(model)
    model.gamma == -1.0 && (model.gamma = 1.0/size(Xmatrix, 1))
    fitresult = LIBSVM.svmtrain(Xmatrix, y;
        get_svm_parameters(model)...,
        verbose = ifelse(verbosity > 1, true, false)
    )

    report = (gamma=model.gamma,)

    return fitresult, cache, report
end

function MLJBase.fit(model::OneClassSVM, verbosity::Int, X)

    Xmatrix = MLJBase.matrix(X)' # notice the transpose

    cache = nothing

    model = deepcopy(model)
    model.gamma == -1.0 && (model.gamma = 1.0/size(Xmatrix, 1))
    fitresult = LIBSVM.svmtrain(Xmatrix;
        get_svm_parameters(model)...,
        verbose = ifelse(verbosity > 1, true, false)
    )

    report = (gamma=model.gamma,)

    return fitresult, cache, report
end


function MLJBase.predict(model::LinearSVC, fitresult, Xnew)
    result, decode = fitresult
    (p,d) = LIBSVM.LIBLINEAR.linear_predict(result, MLJBase.matrix(Xnew)')
    return decode(p)
end

function MLJBase.predict(model::Union{SVC, NuSVC}, fitresult, Xnew)
    result, decode = fitresult
    (p,d) = LIBSVM.svmpredict(result, MLJBase.matrix(Xnew)')
    return decode(p)
end

function MLJBase.predict(model::Union{NuSVR, EpsilonSVR}, fitresult, Xnew)
    (p,d) = LIBSVM.svmpredict(fitresult, MLJBase.matrix(Xnew)')
    return p
end

function MLJBase.transform(model::OneClassSVM, fitresult, Xnew)
    (p,d) = LIBSVM.svmpredict(fitresult, MLJBase.matrix(Xnew)')
    return categorical(p)
end


# metadata
MLJBase.load_path(::Type{<:LinearSVC}) = "MLJModels.LIBSVM_.LinearSVC"
MLJBase.load_path(::Type{<:SVC}) = "MLJModels.LIBSVM_.SVC"
MLJBase.load_path(::Type{<:NuSVC}) = "MLJModels.LIBSVM_.NuSVC"
MLJBase.load_path(::Type{<:NuSVR}) = "MLJModels.LIBSVM_.NuSVR"
MLJBase.load_path(::Type{<:EpsilonSVR}) = "MLJModels.LIBSVM_.EpsilonSVR"
MLJBase.load_path(::Type{<:OneClassSVM}) = "MLJModels.LIBSVM_.OneClassSVM"

MLJBase.package_name(::Type{<:SVM}) = "LIBSVM"
MLJBase.package_uuid(::Type{<:SVM}) = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
MLJBase.is_pure_julia(::Type{<:SVM}) = false
MLJBase.package_url(::Type{<:SVM}) = "https://github.com/mpastell/LIBSVM.jl"

using Pkg
if Pkg.installed()["MLJBase"] > v"0.3"
    MLJBase.package_license(::Type{<:SVM}) = "BSD-3"
    MLJBase.is_wrapper(::Type{<:SVM}) = true
    MLJBase.input_scitype(::Type{<:SVM}) = MLJBase.Continuous
    MLJBase.target_scitype(::Type{<:Union{LinearSVC, SVC, NuSVC}}) = MLJBase.Finite
    MLJBase.target_scitype(::Type{<:Union{NuSVR, EpsilonSVR}}) = MLJBase.Continuous
    MLJBase.output_scitype(::Type{<:OneClassSVM}) = MLJBase.Finite{2} # Bool (true means inlier)
else
    MLJBase.input_scitype_union(::Type{<:SVM}) = MLJBase.Continuous
    MLJBase.target_scitype_union(::Type{<:Union{LinearSVC, SVC, NuSVC}}) = MLJBase.Finite
    MLJBase.target_scitype_union(::Type{<:Union{NuSVR, EpsilonSVR}}) = MLJBase.Continuous
    MLJBase.output_scitype_union(::Type{<:OneClassSVM}) = MLJBase.Finite{2} # Bool (true means inlier)
end

end # module
