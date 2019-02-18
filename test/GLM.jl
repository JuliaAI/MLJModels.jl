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

typeof(fitresult) isa  MLJBase.fitresult_type(atom_ols)

# ols = machine(atom_ols, MLJBase.selectrows(X, train), y[train])
# fit!(ols)

# p = predict(ols, MLJBase.selectrows(X, test))

# hand made regression to compare

Xa = MLJBase.matrix(X) # convert(Matrix{Float64}, X)
Xa1 = hcat(Xa, ones(size(Xa, 1)))
coefs = Xa1[train, :] \ y[train]

p2 = Xa1[test, :] * coefs

@test p ≈ p2

info(atom_ols)

end # module
true
