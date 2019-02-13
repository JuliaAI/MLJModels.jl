module TestGLM

# using Revise
using MLJ
using Test

import MLJModels
import GLM # MLJModels.GLM_ now available for loading
using MLJModels.GLM_

###
### Linear Model
###

task = load_boston()
X, y = X_and_y(task)

train, test = partition(eachindex(y), 0.7)

atom_ols = OLSRegressor()

ols = machine(atom_ols, MLJ.selectrows(X, train), y[train])
fit!(ols)

p = predict(ols, MLJ.selectrows(X, test))

# hand made regression to compare

Xa = MLJ.matrix(X) # convert(Matrix{Float64}, X)
Xa1 = hcat(Xa, ones(size(Xa, 1)))
coefs = Xa1[train, :] \ y[train]

p2 = Xa1[test, :] * coefs

@test p ≈ p2

info(atom_ols)

###
### Generalized Linear Model
###

task = load_crabs()
X, y = X_and_y(task)

# XXX  use a proper encoder decoder
y01 = [ifelse(yi=="B", 0, 1) for yi ∈ y]

atom_glm = GLMRegressor(distribution=GLM.Poisson())

glm = machine(atom_glm, X, y01)
fit!(glm)

p = predict(glm, X)

end # module
true
