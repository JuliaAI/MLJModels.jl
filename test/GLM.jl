module TestGLM

# using Revise
using Test

using MLJBase
using RDatasets

import MLJModels
import GLM
using MLJModels.GLM_


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
MLJBase.fitted_params(atom_ols, fitresult)

p = predict_mean(atom_ols, fitresult, Xtest)

# hand made regression to compare

Xa = MLJBase.matrix(X) # convert(Matrix{Float64}, X)
Xa1 = hcat(Xa, ones(size(Xa, 1)))
coefs = Xa1[train, :] \ y[train]
p2 = Xa1[test, :] * coefs

@test p ≈ p2

info(atom_ols)

p_distr = predict(atom_ols, fitresult, selectrows(X, test))

###
### GLMCOUNT
###

using Random: seed!
using Distributions

seed!(0)

X = randn(100, 3) .* randn(3)'
Xtable = table(X)

α = 0.1
β = [-0.3, 0.2, -0.1]
λ = exp.(α .+ X * β)

y = [rand(Poisson(λᵢ)) for λᵢ ∈ λ]

atom_glmcount = GLMCount()

fitresult, _, _ = fit(atom_glmcount, 1, Xtable, y)

p = predict_mean(atom_glmcount, fitresult, Xtable)

p_distr = predict(atom_glmcount, fitresult, Xtable)

info(atom_glmcount)


end # module
true
