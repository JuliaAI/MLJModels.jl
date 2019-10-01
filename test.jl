using MLJBase
using RDatasets
using LinearAlgebra

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

atom_ols = LinearRegressor()

Xtrain = selectrows(X, train)
ytrain = selectrows(y, train)
Xtest  = selectrows(X, test)

fitresult, _, report = fit(atom_ols, 1, Xtrain, ytrain)
