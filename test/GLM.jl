module TestGLM

# using Revise
using Test
using MLJBase

import MLJModels
import GLM # MLJModels.GLM_ now available for loading
using MLJModels.GLM_

task = load_boston()
X, y = X_and_y(task)

train, test = partition(eachindex(y), 0.7)

atom_ols = OLSRegressor()

fitresult, cache, report = MLJBase.fit(atom_ols, 1, MLJBase.selectrows(X, train), y[train])
p = predict(atom_ols, fitresult, MLJBase.selectrows(X, test))

p = predict_mean(ols, MLJBase.selectrows(X, test))

# hand made regression to compare

Xa = MLJBase.matrix(X) # convert(Matrix{Float64}, X)
Xa1 = hcat(Xa, ones(size(Xa, 1)))
coefs = Xa1[train, :] \ y[train]

p2 = Xa1[test, :] * coefs

@test p ≈ p2

info(atom_ols)

p_distr = predict(ols, MLJBase.selectrows(X, test))

###

using Random: seed!
using Distributions

seed!(0)

X = randn(100, 3) .* randn(3)'
Xtable = MLJ.table(X)

α = 0.1
β = [-0.3, 0.2, -0.1]
λ = exp.(α .+ X * β)

y = [rand(Poisson(λᵢ)) for λᵢ ∈ λ]

atom_glmcount = GLMCount()

glmc = machine(atom_glmcount, Xtable, y)
fit!(glmc)

p = predict_mean(glmc, Xtable)

p_distr = predict(glmc, Xtable)

end # module
true
