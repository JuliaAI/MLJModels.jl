BayesianLDA_ = SKDA.LinearDiscriminantAnalysis
@sk_clf mutable struct BayesianLDA <: MLJBase.Probabilistic
    solver::String                   = "svd"::(_ in ("svd", "lsqr", "eigen"))
    shrinkage::Union{Nothing,String,Float64} = nothing::(_ === nothing || _ == "auto" || 0 < _ < 1)
    priors::Option{AbstractVector}   = nothing
    n_components::Option{Int}        = nothing
    store_covariance::Bool           = false
    tol::Float64                     = 1e-4::(_ > 0)
end
MLJBase.fitted_params(m::BayesianLDA, (f, _, _)) = (
    coef       = f.coef_,
    intercept  = f.intercept_,
    covariance = m.store_covariance ? f.covariance_ : nothing,
    means      = f.means_,
    priors     = f.priors_,
    scalings   = f.scalings_,
    xbar       = f.xbar_,
    classes    = f.classes_,
    explained_variance_ratio = f.explained_variance_ratio_
    )
metadata_model(BayesianLDA,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Bayesian Linear Discriminant Analysis."
    )

BayesianQDA_ = SKDA.QuadraticDiscriminantAnalysis
@sk_clf mutable struct BayesianQDA <: MLJBase.Probabilistic
    priors::Option{AbstractVector} = nothing
    reg_param::Float64             = 0.0::(_ ≥ 0)
    store_covariance::Bool         = false
    tol::Float64                   = 1e-4::(_ > 0)
end
MLJBase.fitted_params(m::BayesianQDA, (f, _, _)) = (
    covariance = m.store_covariance ? f.covariance_ : nothing,
    means      = f.means_,
    priors     = f.priors_,
    rotations  = f.rotations_,
    scalings   = f.scalings_
    )
metadata_model(BayesianQDA,
    input=MLJBase.Table(MLJBase.Continuous),
    target=AbstractVector{<:MLJBase.Finite},
    weights=false,
    descr="Bayesian Quadratic Discriminant Analysis."
    )
