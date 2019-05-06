module GaussianProcesses_

export GPClassifier

import MLJBase

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
    
    decoder = MLJBase.CategoricalDecoder(y, Int)
    y_plain = MLJBase.transform(decoder, y)

    if VERSION < v"1.0"
        XT = collect(transpose(Xmatrix))
        yP = convert(Vector{Float64}, y_plain)
        gp = GP.GPE(XT
                  , yP
                  , model.mean
                  , model.kernel)

        GP.fit!(gp, XT, yP)
    else
        gp = GP.GPE(transpose(Xmatrix)
                  , y_plain
                  , model.mean
                  , model.kernel)
        GP.fit!(gp, transpose(Xmatrix), y_plain)
    end

    fitresult = (gp, decoder)

    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function MLJBase.predict(model::GPClassifier
                       , fitresult
                       , Xnew) 

    Xmatrix = MLJBase.matrix(Xnew)
    
    gp, decoder = fitresult

    nlevels = length(decoder.pool.levels)
    pred = GP.predict_y(gp, transpose(Xmatrix))[1] # Float
    # rounding with clamping between 1 and nlevels
    pred_rc = clamp.(round.(Int, pred), 1, nlevels)

    return MLJBase.inverse_transform(decoder, pred_rc)
end

# metadata:
MLJBase.load_path(::Type{<:GPClassifier}) = "MLJModels.GaussianProcesses_.GPClassifier" # lazy-loaded from MLJ
MLJBase.package_name(::Type{<:GPClassifier}) = "GaussianProcesses"
MLJBase.package_uuid(::Type{<:GPClassifier}) = "891a1506-143c-57d2-908e-e1f8e92e6de9"
MLJBase.package_url(::Type{<:GPClassifier}) = "https://github.com/STOR-i/GaussianProcesses.jl"
MLJBase.is_pure_julia(::Type{<:GPClassifier}) = true
MLJBase.input_scitype_union(::Type{<:GPClassifier}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:GPClassifier}) = Union{MLJBase.Multiclass,MLJBase.OrderedFactor}
MLJBase.input_is_multivariate(::Type{<:GPClassifier}) = true

end # module

