module TestGLM

using Test

using MLJBase
import MLJModels
import Distributions
import GLM
using MLJModels.GLM_
using Random: seed!
using RDatasets

###
### OLSREGRESSOR
###

task = load_boston()
X, y = task();

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

@test infos[:package_name] == "GLM"
@test infos[:is_pure_julia]
@test infos[:is_probabilistic]
@test infos[:target_scitype_union] == MLJBase.Continuous

p_distr = predict(atom_ols, fitresult, selectrows(X, test))

@test p_distr[1] == Distributions.Normal(p[1], GLM.dispersion(fitresult))

###
### Logistic regression
###

seed!(0)

# data drawn from https://stats.idre.ucla.edu/r/dae/poisson-regression/
data = dataset("MASS", "Melanoma")

X = data[[:Status, :Sex, :Age, :Year, :Thickness]]
y = data[:Ulcer]

n = length(y)

baseline_y = convert.(Int, rand(n) .> 0.5)
baseline_mse = sum((baseline_y - y).^2)/n

lr = BinaryClassifier()
fitresult, _, report = fit(lr, 1, X, y)
p_mode = predict_mode(lr, fitresult, X)
# rough test
mse = sum((p_mode .- y).^2)/n
@test mse < baseline_mse

pr = BinaryClassifier(link=GLM.ProbitLink())
fitresult, _, report = fit(pr, 1, X, y)
p_mode = predict_mode(pr, fitresult, X)
mse = sum((p_mode .- y).^2)/n
@test mse < baseline_mse

end # module
true
