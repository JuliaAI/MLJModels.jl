using MLJBase
using RDatasets
using LinearAlgebra
using MLJ
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

function var_rule(n,t,s;h=0.1)
    return s <: Continuous ? (std(X[n])>h ? true : false) : true
end


std(X[:Crim])

sch = MLJBase.schema(X)
fsr=MLJModels.FeatureSelectorRule(rule=var_rule, kwargs=(h=>9.0,)
params(fsr)
fitresult, _, report = fit(atom_ols, 1, Xtrain, ytrain)
pipe=@pipeline MyPipe(fsr=fsr, ols=atom_ols)

TunedModel(model=pipe, tuning=Grid(), resampling=Holdout(),
measure=rms, operation=predict, ranges=range(fsr,:kwargs,[(:h=>8.0,),(:h=>9.0,)]),
minimize=true, full_report=true)
