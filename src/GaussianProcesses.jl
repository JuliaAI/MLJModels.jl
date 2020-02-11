module GaussianProcesses_

export GPClassifier

import MLJModelInterface
import MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor,
                          Multiclass

const MMI = MLJModelInterface

using CategoricalArrays

import ..GaussianProcesses # strange lazy-loading syntax

const GP = GaussianProcesses

mutable struct GPClassifier{M<:GP.Mean, K<:GP.Kernel} <: MMI.Deterministic
    mean::M
    kernel::K
end

function GPClassifier(
    ; mean=GP.MeanZero()
    , kernel=GP.SE(0.0,1.0)) # binary

    model = GPClassifier(
        mean
        , kernel)

    message = MMI.clean!(model)
    isempty(message) || @warn message

    return model
end

# function MMI.clean! not provided

function MMI.fit(model::GPClassifier{M,K}
            , verbosity::Int
            , X
            , y) where {M,K}

    Xmatrix = MMI.matrix(X)

    y_plain = MMI.int(y)

    a_target_element = y[1]
    nclasses = length(MMI.classes(a_target_element))
    decode = MMI.decoder(a_target_element)

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

function MMI.predict(model::GPClassifier
                       , fitresult
                       , Xnew)

    Xmatrix = MMI.matrix(Xnew)

    gp, nclasses, decode = fitresult

    pred = GP.predict_y(gp, transpose(Xmatrix))[1] # Float
    # rounding with clamping between 1 and nlevels
    pred_rc = clamp.(round.(Int, pred), 1, nclasses)

    return decode(pred_rc)
end

# metadata:
MMI.load_path(::Type{<:GPClassifier}) = "MLJModels.GaussianProcesses_.GPClassifier" # lazy-loaded from MLJ
MMI.package_name(::Type{<:GPClassifier}) = "GaussianProcesses"
MMI.package_uuid(::Type{<:GPClassifier}) = "891a1506-143c-57d2-908e-e1f8e92e6de9"
MMI.package_url(::Type{<:GPClassifier}) = "https://github.com/STOR-i/GaussianProcesses.jl"
MMI.is_pure_julia(::Type{<:GPClassifier}) = true
MMI.input_scitype(::Type{<:GPClassifier}) = Table(Continuous)
MMI.target_scitype(::Type{<:GPClassifier}) = AbstractVector{<:Finite}

end # module
