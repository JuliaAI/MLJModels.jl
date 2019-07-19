using Test, MLJBase, Distributions
using Random: seed!
import MLJModels, GLM
using MLJModels.GLM_

using RDatasets

task = load_boston()
X, y = task()
train, test = partition(eachindex(y), 0.7)

# -----------
#  💡 OLS 💡
# -----------

ols = OLSRegressor()

Xtrain = selectrows(X, train)
ytrain = selectrows(y, train)
Xtest  = selectrows(X, test)

fitresult, _, report = fit(ols, 1, Xtrain, ytrain)

n = size(Xtrain, 1)
p = size(Xtrain, 2)
@test report.dof_residual == n - (p + 1)

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

infos = info(ols)

@test infos[:is_pure_julia]
@test infos[:input_is_multivariate]
@test !infos[:is_wrapper]
@test infos[:target_scitype_union] == Union{MLJBase.Continuous, MLJBase.Count}
@test infos[:package_name] == "GLM"

p_distr = predict(ols, fitresult, selectrows(X, test))

@test isa(p_distr, Vector{Normal{Float64}})

Distributions.mean(p_distr[1]) ≈ p2[1]

# --------------------------
#  💡 Logistic Regression 💡
# --------------------------

# data drawn from https://stats.idre.ucla.edu/r/dae/poisson-regression/

data = dataset("MASS", "Melanoma")

X = data[[:Status, :Sex, :Age, :Year, :Thickness]]
y = data[:Ulcer]

n = length(y)

# MSE is not the right metric here but gives an idea that something is recovered
baseline_mse = sum((0.5ones(n) - y).^2)/n

for model ∈ (LogitRegressor, ProbitRegressor, CauchitRegressor, CloglogRegressor)
    m = model()
    fitresult, _, _ = fit(m, 1, X, y)
    p_mean = predict_mean(m, fitresult, X)
    mse = sum((p_mean.-y).^2)/n
    @test mse < baseline_mse
end
