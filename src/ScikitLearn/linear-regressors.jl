ARDRegressor_ = SKLM.ARDRegression
@sk_reg mutable struct ARDRegressor <: MLJBase.Deterministic
    n_iter::Int               = 300::(_ > 0)
    tol::Float64              = 1e-3::(_ > 0)
    alpha_1::Float64          = 1e-6::(_ > 0)
    alpha_2::Float64          = 1e-6::(_ > 0)
    lambda_1::Float64         = 1e-6::(_ > 0)
    lambda_2::Float64         = 1e-6::(_ > 0)
    compute_score::Bool       = false
    threshold_lambda::Float64 = 1e4::(_ > 0)
    fit_intercept::Bool       = true
    normalize::Bool           = false
    copy_X::Bool              = true
    verbose::Bool             = false
end
MLJBase.fitted_params(model::ARDRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    lambda    = fitresult.lambda_,
    sigma     = fitresult.sigma_,
    scores    = fitresult.scores_
    )

# ==============================================================================
BayesianRidgeRegressor_ = SKLM.BayesianRidge
@sk_reg mutable struct BayesianRidgeRegressor <: MLJBase.Deterministic
    n_iter::Int         = 300::(_ ≥ 1)
    tol::Float64        = 1e-3::(_ > 0)
    alpha_1::Float64    = 1e-6::(_ > 0)
    alpha_2::Float64    = 1e-6::(_ > 0)
    lambda_1::Float64   = 1e-6::(_ > 0)
    lambda_2::Float64   = 1e-6::(_ > 0)
    compute_score::Bool = false
    fit_intercept::Bool = true
    normalize::Bool     = false
    copy_X::Bool        = true
    verbose::Bool       = false
end
MLJBase.fitted_params(model::BayesianRidgeRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    lambda    = fitresult.lambda_,
    sigma     = fitresult.sigma_,
    scores    = fitresult.scores_
    )

# ==============================================================================
ElasticNetRegressor_ = SKLM.ElasticNet
@sk_reg mutable struct ElasticNetRegressor <: MLJBase.Deterministic
    alpha::Float64      = 1.0::(_ ≥ 0)   # 0 is OLS
    l1_ratio::Float64   = 0.5::(0 ≤ _ ≤ 1)
    fit_intercept::Bool = true
    normalize::Bool     = false
    precompute::Union{Bool,AbstractMatrix} = false
    max_iter::Int       = 1_000::(_ ≥ 1)
    copy_X::Bool        = true
    tol::Float64        = 1e-4::(_ > 0)
    warm_start::Bool    = false
    positive::Bool      = false
    random_state::Any   = nothing  # Int, random state, or nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MLJBase.fitted_params(model::ElasticNetRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    )

# ==============================================================================
ElasticNetCVRegressor_ = SKLM.ElasticNetCV
@sk_reg mutable struct ElasticNetCVRegressor <: MLJBase.Deterministic
    l1_ratio::Union{Float64,Vector{Float64}} = 0.5::(all(0 .≤ _ .≤ 1))
    eps::Float64        = 1e-3::(_ > 0)
    n_alphas::Int       = 100::(_ > 0)
    alphas::Any         = nothing::(_ === nothing || all(0 .≤ _ .≤ 1))
    fit_intercept::Bool = true
    normalize::Bool     = false
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    cv::Any             = 5 # can be Int, Nothing or an iterable / cv splitter
    copy_X::Bool        = true
    verbose::Union{Bool, Int}  = 0
    n_jobs::Option{Int} = nothing
    positive::Bool      = false
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MLJBase.fitted_params(model::ElasticNetCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    l1_ratio  = fitresult.l1_ratio_,
    mse_path  = fitresult.mse_path_,
    alphas    = fitresult.alphas_
    )

# ==============================================================================
HuberRegressor_ = SKLM.HuberRegressor
@sk_reg mutable struct HuberRegressor <: MLJBase.Deterministic
    epsilon::Float64    = 1.35::(_ > 1.0)
    max_iter::Int       = 100::(_ > 0)
    alpha::Float64      = 1e-4::(_ > 0)
    warm_start::Bool    = false
    fit_intercept::Bool = true
    tol::Float64        = 1e-5::(_ > 0)
end
MLJBase.fitted_params(model::HuberRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    scale     = fitresult.scale_,
    outliers  = fitresult.outliers_
    )

# ==============================================================================
LarsRegressor_ = SKLM.Lars
@sk_reg mutable struct LarsRegressor <: MLJBase.Deterministic
    fit_intercept::Bool      = true
    verbose::Union{Bool,Int} = false
    normalize::Bool = true
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    n_nonzero_coefs::Int     = 500::(_ > 0)
    eps::Float64    = eps(Float64)::(_ > 0)
    copy_X::Bool    = true
    fit_path::Bool  = true
end
MLJBase.fitted_params(model::LarsRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alphas    = fitresult.alphas_,
    active    = fitresult.active_,
    coef_path = fitresult.coef_path_
    )

# ==============================================================================
LarsCVRegressor_ = SKLM.LarsCV
@sk_reg mutable struct LarsCVRegressor <: MLJBase.Deterministic
    fit_intercept::Bool      = true
    verbose::Union{Bool,Int} = false
    max_iter::Int     = 500::(_ > 0)
    normalize::Bool   = true
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    cv::Any           = 5
    max_n_alphas::Int = 1_000::(_ > 0)
    n_jobs::Option{Int} = nothing
    eps::Float64      = eps(Float64)::(_ > 0)
    copy_X::Bool      = true
end
MLJBase.fitted_params(model::LarsCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    alphas    = fitresult.alphas_,
    cv_alphas = fitresult.cv_alphas_,
    mse_path  = fitresult.mse_path_,
    coef_path = fitresult.coef_path_
    )

# ==============================================================================
LassoRegressor_ = SKLM.Lasso
@sk_reg mutable struct LassoRegressor <: MLJBase.Deterministic
    alpha::Float64      = 1.0::(_ ≥ 0) # not recommended to use alpha=0 (use OLS)
    fit_intercept::Bool = true
    normalize::Bool     = false
    precompute::Union{Bool,AbstractMatrix} = false
    copy_X::Bool        = true
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    warm_start::Bool    = false
    positive::Bool      = false
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MLJBase.fitted_params(model::LassoRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    )

# ==============================================================================
LassoCVRegressor_ = SKLM.LassoCV
@sk_reg mutable struct LassoCVRegressor <: MLJBase.Deterministic
    eps::Float64        = 1e-3::(_ > 0)
    n_alphas::Int       = 100::(_ > 0)
    alphas::Any         = nothing::(_ === nothing || all(0 .≤ _ .≤ 1))
    fit_intercept::Bool = true
    normalize::Bool     = false
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    copy_X::Bool        = true
    cv::Any             = 5
    verbose::Union{Bool, Int} = false
    n_jobs::Option{Int} = nothing
    positive::Bool      = false
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MLJBase.fitted_params(model::LassoCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha    = fitresult.alpha_,
    alphas   = fitresult.alphas_,
    mse_path = fitresult.mse_path_,
    dual_gap = fitresult.dual_gap_
    )

# ==============================================================================
LassoLarsRegressor_ = SKLM.LassoLars
@sk_reg mutable struct LassoLarsRegressor <: MLJBase.Deterministic
    alpha::Float64      = 1.0::(_ ≥ 0) # 0 should be OLS
    fit_intercept::Bool = true
    verbose::Union{Bool, Int} = false
    normalize::Bool     = true
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    max_iter::Int       = 500::(_ > 0)
    eps::Float64        = eps(Float64)::(_ > 0)
    copy_X::Bool        = true
    fit_path::Bool      = true
    positive::Any       = false
end
MLJBase.fitted_params(model::LassoLarsRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alphas    = fitresult.alphas_,
    active    = fitresult.active_,
    coef_path = fitresult.coef_path_
    )

# ==============================================================================
LassoLarsCVRegressor_ = SKLM.LassoLarsCV
@sk_reg mutable struct LassoLarsCVRegressor <: MLJBase.Deterministic
    fit_intercept::Bool = true
    verbose::Union{Bool, Int} = false
    max_iter::Int       = 500::(_ > 0)
    normalize::Bool     = true
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    cv::Any             = 5
    max_n_alphas::Int   = 1_000::(_ > 0)
    n_jobs::Option{Int} = nothing
    eps::Float64        = eps(Float64)::(_ > 0.0)
    copy_X::Bool        = true
    positive::Any       = false
end
MLJBase.fitted_params(model::LassoLarsCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    coef_path = fitresult.coef_path_,
    alpha     = fitresult.alpha_,
    alphas    = fitresult.alphas_,
    cv_alphas = fitresult.cv_alphas_,
    mse_path  = fitresult.mse_path_
    )

LassoLarsICRegressor_ = SKLM.LassoLarsIC
@sk_reg mutable struct LassoLarsICRegressor <: MLJBase.Deterministic
    criterion::String   = "aic"::(_ in ("aic","bic"))
    fit_intercept::Bool = true
    verbose::Union{Bool, Int} = false
    normalize::Bool     = true
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
    max_iter::Int       = 500::(_ > 0)
    eps::Float64        = eps(Float64)::(_ > 0.0)
    copy_X::Bool        = true
    positive::Any       = false
end
MLJBase.fitted_params(model::LassoLarsICRegressor, (fitresult, _, _)) = (
    coef = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha = fitresult.alpha_
    )

# ==============================================================================
LinearRegressor_ = SKLM.LinearRegression
@sk_reg mutable struct LinearRegressor <: MLJBase.Deterministic
    fit_intercept::Bool = true
    normalize::Bool     = false
    copy_X::Bool        = true
    n_jobs::Option{Int} = nothing
end
MLJBase.fitted_params(model::LinearRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )

# ==============================================================================
OrthogonalMatchingPursuitRegressor_ = SKLM.OrthogonalMatchingPursuit
@sk_reg mutable struct OrthogonalMatchingPursuitRegressor <: MLJBase.Deterministic
    n_nonzero_coefs::Option{Int} = nothing
    tol::Option{Float64} = nothing
    fit_intercept::Bool  = true
    normalize::Bool      = true
    precompute::Union{Bool,String,AbstractMatrix} = "auto"
end
MLJBase.fitted_params(model::OrthogonalMatchingPursuitRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )

# ==============================================================================
OrthogonalMatchingPursuitCVRegressor_ = SKLM.OrthogonalMatchingPursuitCV
@sk_reg mutable struct OrthogonalMatchingPursuitCVRegressor <: MLJBase.Deterministic
    copy::Bool            = true
    fit_intercept::Bool   = true
    normalize::Bool       = false
    max_iter::Option{Int} = nothing::(_ === nothing||_ > 0)
    cv::Any               = 5
    n_jobs::Option{Int}   = 1
    verbose::Union{Bool,Int} = false
end
MLJBase.fitted_params(model::OrthogonalMatchingPursuitCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    n_nonzero_coefs = fitresult.n_nonzero_coefs_
    )

# ==============================================================================
PassiveAggressiveRegressor_ = SKLM.PassiveAggressiveRegressor
@sk_reg mutable struct PassiveAggressiveRegressor <: MLJBase.Deterministic
    C::Float64                   = 1.0::(_ > 0)
    fit_intercept::Bool          = true
    max_iter::Int                = 1_000::(_ > 0)
    tol::Float64                 = 1e-4::(_ > 0)
    early_stopping::Bool         = false
    validation_fraction::Float64 = 0.1::(0 < _ < 1)
    n_iter_no_change::Int        = 5::(_ > 0)
    shuffle::Bool                = true
    verbose::Union{Bool,Int}     = 0
    loss::String                 = "epsilon_insensitive"::(_ in ("epsilon_insensitive","squared_epsilon_insensitive"))
    epsilon::Float64             = 0.1::(_ > 0)
    random_state::Any            = nothing
    warm_start::Bool             = false
    average::Union{Bool,Int}     = false
end
MLJBase.fitted_params(model::PassiveAggressiveRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )

# ==============================================================================
RANSACRegressor_ = SKLM.RANSACRegressor
@sk_reg mutable struct RANSACRegressor <: MLJBase.Deterministic
    base_estimator::Any         = nothing
    min_samples::Union{Int,Float64}     = 5::(_ isa Int ? _ ≥ 1 : (0 ≤ _ ≤ 1))
    residual_threshold::Option{Float64} = nothing
    is_data_valid::Any          = nothing
    is_model_valid::Any         = nothing
    max_trials::Int             = 100::(_ > 0)
    max_skips::Int              = typemax(Int)::(_ > 0)
    stop_n_inliers::Int         = typemax(Int)::(_ > 0)
    stop_score::Float64         = Inf::(_ > 0)
    stop_probability::Float64   = 0.99::(0 ≤ _ ≤ 1.0)
    loss::Union{Function,String}= "absolute_loss"::((_ isa Function) || _ in ("absolute_loss","squared_loss"))
    random_state::Any           = nothing
end
MLJBase.fitted_params(m::RANSACRegressor, (f, _, _)) = (
    estimator             = f.estimator_,
    n_trials              = f.n_trials_,
    inlier_mask           = f.inlier_mask_,
    n_skips_no_inliers    = f.n_skips_no_inliers_,
    n_skips_invalid_data  = f.n_skips_invalid_data_,
    n_skips_invalid_model = f.n_skips_invalid_model_
    )

# ==============================================================================
RidgeRegressor_ = SKLM.Ridge
@sk_reg mutable struct RidgeRegressor <: MLJBase.Deterministic
    alpha::Union{Float64,Vector{Float64}} = 1.0::(all(_ .> 0))
    fit_intercept::Bool = true
    normalize::Bool     = false
    copy_X::Bool        = true
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    solver::String      = "auto"::(_ in ("auto","svd","cholesky","lsqr","sparse_cg","sag","saga"))
    random_state::Any   = nothing
end
MLJBase.fitted_params(model::RidgeRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )

# ==============================================================================
RidgeCVRegressor_ = SKLM.RidgeCV
@sk_reg mutable struct RidgeCVRegressor <: MLJBase.Deterministic
    alphas::Any              = (0.1, 1.0, 10.0)::(all(_ .> 0))
    fit_intercept::Bool      = true
    normalize::Bool          = false
    scoring::Any             = nothing
    cv::Any                  = 5
    gcv_mode::Option{String} = nothing::(_ === nothing || _ in ("auto","svd","eigen"))
    store_cv_values::Bool    = false
end
MLJBase.fitted_params(model::RidgeCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    cv_values = model.store_cv_values ? fitresult.cv_values_ : nothing
    )

# ==============================================================================
SGDRegressor_ = SKLM.SGDRegressor
@sk_reg mutable struct SGDRegressor <: MLJBase.Deterministic
    loss::String             = "squared_loss"::(_ in ("squared_loss","huber","epsilon_insensitive","squared_epsilon_insensitive"))
    penalty::String          = "l2"::(_ in ("none","l2","l1","elasticnet"))
    alpha::Float64           = 1e-4::(_ > 0)
    l1_ratio::Float64        = 0.15::(_ > 0)
    fit_intercept::Bool      = true
    max_iter::Int            = 1_000::(_ > 0)
    tol::Float64             = 1e-3::(_ > 0)
    shuffle::Bool            = true
    verbose::Union{Int,Bool} = 0
    epsilon::Float64         = 0.1
    random_state::Any        = nothing
    learning_rate::String    = "invscaling"::(_ in ("constant","optimal","invscaling","adaptive"))
    eta0::Float64            = 0.01::(_ > 0)
    power_t::Float64         = 0.25::(_ > 0)
    early_stopping::Bool     = false
    validation_fraction::Float64 = 0.1::(0 < _ < 1)
    n_iter_no_change::Int    = 5::(_ > 0)
    warm_start::Bool         = false
    average::Union{Int,Bool} = false
end
MLJBase.fitted_params(model::SGDRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    average_coef      = model.average ? fitresult.average_coef_ : nothing,
    average_intercept = model.average ? ifelse(model.fit_intercept, fitresult.average_intercept_, nothing) : nothing
    )

# ==============================================================================
TheilSenRegressor_ = SKLM.TheilSenRegressor
@sk_reg mutable struct TheilSenRegressor <: MLJBase.Deterministic
    fit_intercept::Bool = true
    copy_X::Bool        = true
    max_subpopulation::Int    = 10_000::(_ > 0)
    n_subsamples::Option{Int} = nothing::(_ === nothing||_ > 0)
    max_iter::Int       = 300::(_ > 0)
    tol::Float64        = 1e-3::(_ > 0)
    random_state::Any   = nothing
    n_jobs::Option{Int} = nothing
    verbose::Bool       = false
end
MLJBase.fitted_params(model::TheilSenRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    breakdown       = fitresult.breakdown_,
    n_subpopulation = fitresult.n_subpopulation_
    )

# Metadata for Continuous -> Vector{Continuous}
const SKL_REGS_SINGLE = Union{Type{<:ARDRegressor},Type{<:BayesianRidgeRegressor},
                              Type{<:ElasticNetRegressor},Type{<:ElasticNetCVRegressor},
                              Type{<:HuberRegressor},Type{<:LarsRegressor},Type{<:LarsCVRegressor},
                              Type{<:LassoRegressor},Type{<:LassoCVRegressor},
                              Type{<:LassoLarsRegressor},Type{<:LassoLarsCVRegressor},
                              Type{<:LassoLarsICRegressor},Type{<:LinearRegressor},
                              Type{<:OrthogonalMatchingPursuitRegressor},
                              Type{<:OrthogonalMatchingPursuitCVRegressor},
                              Type{<:PassiveAggressiveRegressor}, Type{<:RANSACRegressor},
                              Type{<:RidgeRegressor},Type{<:RidgeCVRegressor},Type{<:SGDRegressor},
                              Type{<:TheilSenRegressor}}

MLJBase.input_scitype(::SKL_REGS_SINGLE)  = MLJBase.Table(Continuous)
MLJBase.target_scitype(::SKL_REGS_SINGLE) = AbstractVector{Continuous}

##############
# MULTI TASK #
##############

# ==============================================================================
MultiTaskLassoRegressor_ = SKLM.MultiTaskLasso
@sk_reg mutable struct MultiTaskLassoRegressor <: MLJBase.Deterministic
    alpha::Float64      = 1.0::(_ ≥ 0)
    fit_intercept::Bool = true
    normalize::Bool     = false
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    copy_X::Bool        = true
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MLJBase.fitted_params(model::MultiTaskLassoRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )

# ==============================================================================
MultiTaskLassoCVRegressor_ = SKLM.MultiTaskLassoCV
@sk_reg mutable struct MultiTaskLassoCVRegressor <: MLJBase.Deterministic
    eps::Float64        = 1e-3::(_ > 0)
    n_alphas::Int       = 100::(_ > 0)
    alphas::Any         = nothing::(_ === nothing || all(0 .≤ _ .≤ 1))
    fit_intercept::Bool = true
    normalize::Bool     = false
    max_iter::Int       = 300::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    copy_X::Bool        = true
    cv::Any             = 5
    verbose::Union{Bool, Int} = false
    n_jobs::Option{Int} = 1
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MLJBase.fitted_params(model::MultiTaskLassoCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    mse_path  = fitresult.mse_path_,
    alphas    = fitresult.alphas_
    )

# ==============================================================================
MultiTaskElasticNetRegressor_ = SKLM.MultiTaskElasticNet
@sk_reg mutable struct MultiTaskElasticNetRegressor <: MLJBase.Deterministic
    alpha::Float64      = 1.0::(_ ≥ 0)
    l1_ratio::Union{Float64, Vector{Float64}} = 0.5::(0 ≤ _ ≤ 1)
    fit_intercept::Bool = true
    normalize::Bool     = true
    copy_X::Bool        = true
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    warm_start::Bool    = false
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MLJBase.fitted_params(model::MultiTaskElasticNetRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing)
    )

# ==============================================================================
MultiTaskElasticNetCVRegressor_ = SKLM.MultiTaskElasticNetCV
@sk_reg mutable struct MultiTaskElasticNetCVRegressor <: MLJBase.Deterministic
    l1_ratio::Union{Float64, Vector{Float64}} = 0.5::(0 ≤ _ ≤ 1)
    eps::Float64        = 1e-3::(_ > 0)
    n_alphas::Int       = 100::(_ > 0)
    alphas::Any         = nothing::(_ === nothing || all(0 .≤ _ .≤ 1))
    fit_intercept::Bool = true
    normalize::Bool     = false
    max_iter::Int       = 1_000::(_ > 0)
    tol::Float64        = 1e-4::(_ > 0)
    cv::Any             = 5
    copy_X::Bool        = true
    verbose::Union{Bool,Int} = 0
    n_jobs::Option{Int} = nothing
    random_state::Any   = nothing
    selection::String   = "cyclic"::(_ in ("cyclic","random"))
end
MLJBase.fitted_params(model::MultiTaskElasticNetCVRegressor, (fitresult, _, _)) = (
    coef      = fitresult.coef_,
    intercept = ifelse(model.fit_intercept, fitresult.intercept_, nothing),
    alpha     = fitresult.alpha_,
    mse_path  = fitresult.mse_path_,
    l1_ratio  = fitresult.l1_ratio_
    )

const SKL_REGS_MULTI = Union{Type{<:MultiTaskLassoRegressor}, Type{<:MultiTaskLassoCVRegressor},
       Type{<:MultiTaskElasticNetRegressor}, Type{<:MultiTaskElasticNetCVRegressor}}

MLJBase.input_scitype(::SKL_REGS_MULTI)  = MLJBase.Table(Continuous)
MLJBase.target_scitype(::SKL_REGS_MULTI) = MLJBase.Table(Continuous)
