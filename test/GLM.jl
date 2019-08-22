module TestGLM

using Test

using MLJBase
using RDatasets

import MLJModels
import Distributions
import GLM
using MLJModels.GLM_
using Random: seed!
using RDatasets


###
### OLSREGRESSOR
###

boston = dataset("MASS", "Boston")
X = MLJBase.selectcols(boston, [:Crim, :Zn, :Indus, :NOx, :Rm, :Age,
                                :Dis, :Rad, :Tax, :PTRatio, :Black,
                                :LStat])
y = MLJBase.selectcols(boston, :MedV)

train, test = partition(eachindex(y), 0.7)

atom_ols = OLSRegressor()

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

infos = info(atom_ols)

@test infos[:name] == "NormalRegressor"
@test infos[:package_name] == "GLM"
@test infos[:is_pure_julia]
@test infos[:is_supervised]
#@test infos[:package_license] == "MIT"
@test infos[:is_probabilistic]

p_distr = predict(atom_ols, fitresult, selectrows(X, test))

@test p_distr[1] == Distributions.Normal(p[1], GLM.dispersion(fitresult))

###
### Logistic regression
###

seed!(0)

# data drawn from https://stats.idre.ucla.edu/r/dae/poisson-regression/
data = dataset("MASS", "Melanoma")

X = data[[:Status, :Sex, :Age, :Year, :Thickness]]
y_plain = data[:Ulcer]
y = categorical(y_plain)

n = length(y)

baseline_y = convert.(Int, rand(n) .> 0.5)
baseline_mse = sum((baseline_y - y_plain).^2)/n

@test baseline_mse ≤ 0.55

lr = BinaryClassifier()
fitresult, _, report = fit(lr, 1, X, y)

p_mean  = predict_mean(lr, fitresult, X)
p_mode1 = convert.(Int, p_mean .> 0.5)
@test sum((p_mode1 - y_plain).^2)/n < 0.26

p_mode = predict_mode(lr, fitresult, X)

@test p_mode1 == convert.(Int, p_mode)

pr = BinaryClassifier(link=GLM.ProbitLink())
fitresult, _, report = fit(pr, 1, X, y)
p_mode = convert.(Int, predict_mode(pr, fitresult, X))
@test sum((p_mode - y_plain).^2)/n < 0.26

# info(atom_glmcount)


end # module
true
