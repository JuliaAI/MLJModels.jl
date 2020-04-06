module TestGLM

using Test

using MLJBase
using LinearAlgebra
using Statistics

import MLJModels
import Distributions
import GLM
using MLJModels.GLM_
using Random: seed!
using Tables

###
### OLSREGRESSOR
###

X, y = @load_boston

train, test = partition(eachindex(y), 0.7)

atom_ols = LinearRegressor()

Xtrain = selectrows(X, train)
ytrain = selectrows(y, train)
Xtest  = selectrows(X, test)

fitresult, _, report = fit(atom_ols, 1, Xtrain, ytrain)
θ = MLJBase.fitted_params(atom_ols, fitresult)

p = predict_mean(atom_ols, fitresult, Xtest)

# hand made regression to compare

Xa = MLJBase.matrix(X) # convert(Matrix{Float64}, X)
Xa1 = hcat(Xa, ones(size(Xa, 1)))
coefs = Xa1[train, :] \ y[train]
p2 = Xa1[test, :] * coefs

@test p ≈ p2

infos = info_dict(atom_ols)

@test infos[:name] == "LinearRegressor"
@test infos[:package_name] == "GLM"
@test infos[:is_pure_julia]
@test infos[:is_supervised]
@test infos[:package_license] == "MIT"
@test infos[:prediction_type] == :probabilistic
@test infos[:hyperparameters] == (:fit_intercept, :allowrankdeficient)
@test infos[:hyperparameter_types] == ("Bool", "Bool")

p_distr = predict(atom_ols, fitresult, selectrows(X, test))

@test p_distr[1] == Distributions.Normal(p[1], GLM.dispersion(fitresult))

###
### Logistic regression
###

seed!(0)

N = 100
X = MLJBase.table(rand(N, 4));
ycont = 2*X.x1 - X.x3 + 0.6*rand(N)
y = (ycont .> mean(ycont)) |> categorical;

lr = LinearBinaryClassifier()
fitresult, _, report = fit(lr, 1, X, y)

yhat = predict(lr, fitresult, X)
@test mean(cross_entropy(yhat, y)) < 0.25

pr = LinearBinaryClassifier(link=GLM.ProbitLink())
fitresult, _, report = fit(pr, 1, X, y)
yhat = predict(lr, fitresult, X)
@test mean(cross_entropy(yhat, y)) < 0.25


# info_dict(atom_glmcount)

###
### Count regression
###

seed!(1512)

X = randn(500, 5)
θ = randn(5)
y = rand.(Distributions.Poisson.(exp.(X*θ)))

XTable = MLJBase.table(X)

lcr = LinearCountRegressor(fit_intercept=false)
fitresult, _, _ = fit(lcr, 1, XTable, y)

θ̂ = fitted_params(lcr, fitresult).coef

@test norm(θ̂ .- θ)/norm(θ) ≤ 5e-3

end # module
true
