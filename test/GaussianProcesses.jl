module TestGaussianProcesses

using MLJBase
using RDatasets
using Test
using Random:seed!
import CategoricalArrays

seed!(113355)

data = dataset("MASS", "crabs")
X = MLJBase.selectcols(data, [:FL, :RW, :CL, :CW, :BD])   
y = MLJBase.selectcols(data, :Sp)

# load code to be tested:
import MLJModels
import GaussianProcesses
using MLJModels.GaussianProcesses_

baregp = GPClassifier()

# split the rows:
allrows = eachindex(y)
train, test = partition(allrows, 0.7, shuffle=true)

fitresult, cache, report =
    MLJBase.fit(baregp, 1, MLJBase.selectrows(X, train), y[train])
yhat = predict(baregp, fitresult, MLJBase.selectrows(X, test));

@test sum(yhat .== y[test]) / length(y[test]) >= 0.7 # around 0.7

fitresult, cache, report = MLJBase.fit(baregp, 1, X, y)
yhat2 = predict(baregp, fitresult, MLJBase.selectrows(X, test));


# gp = machine(baregp, X, y)
# fit!(gp)
# yhat2 = predict(gp, MLJBase.selectrows(X, test))

@test sum(yhat2 .== y[test]) / length(y[test]) >= 0.7

info_dict(baregp)

end # module
true
