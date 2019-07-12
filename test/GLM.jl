using Test, MLJBase, Distributions
using Random: seed!
import MLJModels, GLM
using MLJModels.GLM_

# ----------------------------------------------------------------------------
# OLS Regression
# ----------------------------------------------------------------------------

task = load_boston()
X, y = task();
train, test = partition(eachindex(y), 0.7)

ols = OLSRegressor()

Xtrain = selectrows(X, train)
ytrain = selectrows(y, train)
Xtest  = selectrows(X, test)

fitresult, _, report = fit(ols, 1, Xtrain, ytrain)

fitparams = MLJBase.fitted_params(ols, fitresult)

@test isa(fitparams.coef, Vector{Float64})
@test isa(fitparams.intercept, Float64)

p = predict_mean(ols, fitresult, Xtest)

# hand made regression to compare

Xa    = MLJBase.matrix(X) # convert(Matrix{Float64}, X)
Xa1   = hcat(Xa, ones(size(Xa, 1)))
coefs = Xa1[train, :] \ y[train]
p2    = Xa1[test, :] * coefs

@test p ≈ p2

info(ols)

p_distr = predict(ols, fitresult, selectrows(X, test))

@test isa(p_distr, Vector{Normal{Float64}})

# ----------------------------------------------------------------------------
# GLM COUNT (poisson link)
# ----------------------------------------------------------------------------

seed!(0)

X = randn(100, 3) .* randn(3)'
Xtable = table(X)

α = 0.1
β = [-0.3, 0.2, -0.1]
λ = exp.(α .+ X * β)

y = [rand(Poisson(λᵢ)) for λᵢ ∈ λ]

glmcount = GLMCount()

fitresult, _, _ = fit(glmcount, 1, Xtable, y)

p = predict_mean(glmcount, fitresult, Xtable)

@test isa(p, Vector{Float64})

p_distr = predict(glmcount, fitresult, Xtable)

@test isa(p_distr, Vector{Poisson{Float64}})
