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
@load SVMRegressor pkg="ScikitLearn"
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

mutable struct StdRule <: MLJModels.SelectorRule
     threshold::Float64
end
(sr::StdRule)(X,name,type,scitypes) = std(X[name]) < sr.threshold
const StdSelector = MLJModels.FeatureSelectorRule{StdRule}
StdSelector(;threshold=0.1)= StdSelector(StdRule(threshold))
fsr2=StdSelector()

sr=StdRule(0.1)




fsr=MLJModels.FeatureSelectorRule{StdRule}(sr)

const StdSelector = MLJModels.FeatureSelectorRule{StdRule}
StdSelector(;threshold=0.1)= StdSelector(StdRule(threshold))
fsr2=StdSelector()
sr(X,:Crim,Float64,Continuous)

fsr_fit,=fit(fsr,1,X)
fsr_fit2,=fit(fsr2,1,X)
transform(fsr,fsr_fit,X)
params(fsr)
fitresult, _, report = fit(atom_ols, 1, Xtrain, ytrain)
pipe=@pipeline MyPipe(fsr=fsr, ols=atom_ols) is_probabilistic=false
fit(pipe,1,X,y)

tm= TunedModel(model=pipe, tuning=Grid(), resampling=Holdout(),
measure=rms, operation=predict, ranges=range(pipe,:(fsr.rule.threshold),lower=8.0,upper=9.0),
minimize=true, full_report=true)
fit(tm,1, X,y)
params(pipe)
params(pipe)
