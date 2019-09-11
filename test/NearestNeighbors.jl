module TestNearestNeighbors

using Test
import MLJModels
import NearestNeighbors
using MLJModels.NearestNeighbors_
using CategoricalArrays
using MLJBase
using Random

Random.seed!(5151)

# create something rather silly: 3 massive clusters very separated

n, p = 50, 3

x1 = randn(n, p)
x2 = randn(n, p) .+ 2
x3 = randn(n, p) .- 2

x = table(vcat(x1, x2, x3))

y1 = fill("A", n)
y2 = fill("B", n)
y3 = fill("C", n)

y = categorical(vcat(y1, y2, y3))

ntest =  5
xtest1 = randn(ntest, p)
xtest2 = randn(ntest, p) .+ 2
xtest3 = randn(ntest, p) .- 2

xtest = table(vcat(xtest1, xtest2, xtest3))

ytest1 = fill("A", ntest)
ytest2 = fill("B", ntest)
ytest3 = fill("C", ntest)

ytest = vcat(ytest1, ytest2, ytest3)

knn = KNNClassifier()

f,_,_ = fit(knn, 1, x, y)

p = predict(knn, f, xtest)

@test p[1] isa UnivariateFinite

p = predict_mode(knn, f, xtest)

@test sum(p .== ytest)/length(ytest) ≥ 0.9

end
true
