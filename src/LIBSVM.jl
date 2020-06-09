module LIBSVM_

export LinearSVC, SVC
export NuSVC, NuSVR
export EpsilonSVR
export OneClassSVM

import MLJModelInterface
import MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor,
                          Multiclass

const MMI = MLJModelInterface

import ..LIBSVM
using Statistics

"""
    LinearSVC(; kwargs...)

Linear support vector machine classifier using LIBLINEAR: https://www.csie.ntu.edu.tw/~cjlin/liblinear/

See also SVC, NuSVC
"""
mutable struct LinearSVC <: MMI.Deterministic
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

    message = MMI.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

"""
    SVC(; kwargs...)

Kernel support vector machine classifier using LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
If `gamma==-1.0` then  `gamma = 1/nfeatures is used in
fitting.
If `gamma==0.0` then a `gamma = 1/(var(X) * nfeatures)` is
used in fitting

See also LinearSVC, NuSVC
"""
mutable struct SVC <: MMI.Deterministic
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    weights::Union{Dict, Nothing}
    cost::Float64
    cachesize::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    probability::Bool
end

function SVC(
    ;kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = 0.0
    ,weights::Union{Dict, Nothing} = nothing
    ,cost::Float64 = 1.0
    ,cachesize::Float64=200.0
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
        ,cachesize
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
        ,probability
    )

    message = MMI.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

"""
    NuSVC(; kwargs...)

Kernel support vector machine classifier using LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

If `gamma==-1.0` then  `gamma = 1/nfeatures is used in
fitting.
If `gamma==0.0` then a `gamma = 1/(var(X) * nfeatures)` is
used in fitting

See also LinearSVC, SVC
"""
mutable struct NuSVC <: MMI.Deterministic
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    weights::Union{Dict, Nothing}
    nu::Float64
    cost::Float64
    cachesize::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function NuSVC(
    ;kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = 0.0
    ,weights::Union{Dict, Nothing} = nothing
    ,nu::Float64 = 0.5
    ,cost::Float64 = 1.0
    ,cachesize::Float64 = 200.0
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
        ,cachesize
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MMI.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

mutable struct OneClassSVM <: MMI.Unsupervised
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    nu::Float64
    cost::Float64
    cachesize::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function OneClassSVM(
    ;kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = 0.0
    ,nu::Float64 = 0.1
    ,cost::Float64 = 1.0
    ,cachesize::Float64 = 200.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.0
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = OneClassSVM(
        kernel
        ,gamma
        ,nu
        ,cost
        ,cachesize
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MMI.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

"""
    NuSVR(; kwargs...)

Kernel support vector machine regressor using LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

If `gamma==-1.0` then  `gamma = 1/nfeatures is used in
fitting.
If `gamma==0.0` then a `gamma = 1/(var(X) * nfeatures)` is
used in fitting

See also EpsilonSVR
"""
mutable struct NuSVR <: MMI.Deterministic
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    nu::Float64
    cost::Float64
    cachesize::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function NuSVR(
    ;kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = 0.0
    ,nu::Float64 = 0.5
    ,cost::Float64 = 1.0
    ,cachesize::Float64 = 200.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = NuSVR(
        kernel
        ,gamma
        ,nu
        ,cost
        ,cachesize
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MMI.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

"""
    EpsilonSVR(; kwargs...)

Kernel support vector machine regressor using LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

If `gamma==-1.0` then  `gamma = 1/nfeatures is used in
fitting.
If `gamma==0.0` then a `gamma = 1/(var(X) * nfeatures)` is
used in fitting

See also NuSVR
"""
mutable struct EpsilonSVR <: MMI.Deterministic
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    epsilon::Float64
    cost::Float64
    cachesize::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function EpsilonSVR(
    ;kernel::LIBSVM.Kernel.KERNEL = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = 0.0
    ,epsilon::Float64 = 0.1
    ,cost::Float64 = 1.0
    ,cachesize::Float64 = 200.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = EpsilonSVR(
        kernel
        ,gamma
        ,epsilon
        ,cost
        ,cachesize
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MMI.clean!(model)   #> future proof by including these
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


function MMI.fit(model::LinearSVC, verbosity::Int, X, y)

    Xmatrix = MMI.matrix(X)' # notice the transpose
    y_plain = MMI.int(y)
    decode  = MMI.decoder(y[1]) # for predict method

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

function MMI.fit(model::Union{SVC, NuSVC}, verbosity::Int, X, y)

    Xmatrix = MMI.matrix(X)' # notice the transpose
    y_plain = MMI.int(y)
    decode  = MMI.decoder(y[1]) # for predict method

    cache = nothing

    model = deepcopy(model)
    model.gamma == -1.0 && (model.gamma = 1.0/size(Xmatrix, 1))
    model.gamma == 0.0 && (model.gamma = 1.0/(var(Xmatrix) * size(Xmatrix, 1)) )
    result = LIBSVM.svmtrain(Xmatrix, y_plain;
        get_svm_parameters(model)...,
        verbose = ifelse(verbosity > 1, true, false)
    )

    fitresult = (result, decode)
    report = (gamma=model.gamma,)

    return fitresult, cache, report
end

function MMI.fit(model::Union{NuSVR, EpsilonSVR}, verbosity::Int, X, y)

    Xmatrix = MMI.matrix(X)' # notice the transpose

    cache = nothing

    model = deepcopy(model)
    model.gamma == -1.0 && (model.gamma = 1.0/size(Xmatrix, 1))
    model.gamma == 0.0 && (model.gamma = 1.0/(var(Xmatrix) * size(Xmatrix, 1)) )
    fitresult = LIBSVM.svmtrain(Xmatrix, y;
        get_svm_parameters(model)...,
        verbose = ifelse(verbosity > 1, true, false)
    )

    report = (gamma=model.gamma,)

    return fitresult, cache, report
end

function MMI.fit(model::OneClassSVM, verbosity::Int, X)

    Xmatrix = MMI.matrix(X)' # notice the transpose

    cache = nothing

    model = deepcopy(model)
    model.gamma == -1.0 && (model.gamma = 1.0/size(Xmatrix, 1))
    model.gamma == 0.0 && (model.gamma = 1.0/(var(Xmatrix) * size(Xmatrix, 1)) )
    fitresult = LIBSVM.svmtrain(Xmatrix;
        get_svm_parameters(model)...,
        verbose = ifelse(verbosity > 1, true, false)
    )

    report = (gamma=model.gamma,)

    return fitresult, cache, report
end


function MMI.predict(model::LinearSVC, fitresult, Xnew)
    result, decode = fitresult
    (p,d) = LIBSVM.LIBLINEAR.linear_predict(result, MMI.matrix(Xnew)')
    return decode(p)
end

function MMI.predict(model::Union{SVC, NuSVC}, fitresult, Xnew)
    result, decode = fitresult
    (p,d) = LIBSVM.svmpredict(result, MMI.matrix(Xnew)')
    return decode(p)
end

function MMI.predict(model::Union{NuSVR, EpsilonSVR}, fitresult, Xnew)
    (p,d) = LIBSVM.svmpredict(fitresult, MMI.matrix(Xnew)')
    return p
end

function MMI.transform(model::OneClassSVM, fitresult, Xnew)
    (p,d) = LIBSVM.svmpredict(fitresult, MMI.matrix(Xnew)')
    return MMI.categorical(p)
end


# metadata
MMI.load_path(::Type{<:LinearSVC}) = "MLJModels.LIBSVM_.LinearSVC"
MMI.load_path(::Type{<:SVC}) = "MLJModels.LIBSVM_.SVC"
MMI.load_path(::Type{<:NuSVC}) = "MLJModels.LIBSVM_.NuSVC"
MMI.load_path(::Type{<:NuSVR}) = "MLJModels.LIBSVM_.NuSVR"
MMI.load_path(::Type{<:EpsilonSVR}) = "MLJModels.LIBSVM_.EpsilonSVR"
MMI.load_path(::Type{<:OneClassSVM}) = "MLJModels.LIBSVM_.OneClassSVM"

MMI.package_name(::Type{<:SVM}) = "LIBSVM"
MMI.package_uuid(::Type{<:SVM}) = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
MMI.is_pure_julia(::Type{<:SVM}) = false
MMI.package_url(::Type{<:SVM}) = "https://github.com/mpastell/LIBSVM.jl"
MMI.input_scitype(::Type{<:SVM}) = Table(Continuous)
MMI.target_scitype(::Type{<:Union{LinearSVC, SVC, NuSVC}}) = AbstractVector{<:Finite}
MMI.target_scitype(::Type{<:Union{NuSVR, EpsilonSVR}}) = AbstractVector{Continuous}
MMI.output_scitype(::Type{<:OneClassSVM}) = AbstractVector{<:Finite{2}} # Bool (true means inlier)

end # module


