module GaussianProcesses_

export GPClassifier

import MLJBase
import MLJBase: Table, Continuous, Count, Finite, OrderedFactor, Multiclass

using CategoricalArrays

import ..GaussianProcesses # strange lazy-loading syntax

const GP = GaussianProcesses

mutable struct GPClassifier{M<:GP.Mean, K<:GP.Kernel} <: MLJBase.Deterministic
    mean::M
    kernel::K
end

function GPClassifier(
    ; mean=GP.MeanZero()
    , kernel=GP.SE(0.0,1.0)) # binary

    model = GPClassifier(
        mean
        , kernel)

    message = MLJBase.clean!(model)
    isempty(message) || @warn message

    return model
end

# function MLJBase.clean! not provided

function MLJBase.fit(model::GPClassifier{M,K}
            , verbosity::Int
            , X
            , y) where {M,K}

    Xmatrix = MLJBase.matrix(X)

    y_plain = MLJBase.int(y)

    a_target_element = y[1]
    nclasses = length(MLJBase.classes(a_target_element))
    decode = MLJBase.decoder(a_target_element)

    gp = GP.GPE(transpose(Xmatrix)
                , y_plain
                , model.mean
                , model.kernel)
    GP.fit!(gp, transpose(Xmatrix), y_plain)

    fitresult = (gp, nclasses, decode)

    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function MLJBase.predict(model::GPClassifier
                       , fitresult
                       , Xnew)

    Xmatrix = MLJBase.matrix(Xnew)

    gp, nclasses, decode = fitresult

    pred = GP.predict_y(gp, transpose(Xmatrix))[1] # Float
    # rounding with clamping between 1 and nlevels
    pred_rc = clamp.(round.(Int, pred), 1, nclasses)

    return decode(pred_rc)
end

# metadata:
MLJBase.load_path(::Type{<:GPClassifier}) = "MLJModels.GaussianProcesses_.GPClassifier" # lazy-loaded from MLJ
MLJBase.package_name(::Type{<:GPClassifier}) = "GaussianProcesses"
MLJBase.package_uuid(::Type{<:GPClassifier}) = "891a1506-143c-57d2-908e-e1f8e92e6de9"
MLJBase.package_url(::Type{<:GPClassifier}) = "https://github.com/STOR-i/GaussianProcesses.jl"
MLJBase.is_pure_julia(::Type{<:GPClassifier}) = true
MLJBase.input_scitype(::Type{<:GPClassifier}) = Table(Continuous)
MLJBase.target_scitype(::Type{<:GPClassifier}) = AbstractVector{<:Finite}

end # module
