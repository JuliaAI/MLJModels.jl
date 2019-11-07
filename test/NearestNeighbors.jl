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
x2 = randn(n, p) .+ 5
x3 = randn(n, p) .- 5

x = table(vcat(x1, x2, x3))

y1 = fill("A", n)
y2 = fill("B", n)
y3 = fill("C", n)

y = categorical(vcat(y1, y2, y3))

w = abs.(50 * randn(nrows(x)))

ntest =  5
xtest1 = randn(ntest, p)
xtest2 = randn(ntest, p) .+ 5
xtest3 = randn(ntest, p) .- 5

xtest = table(vcat(xtest1, xtest2, xtest3))

ytest1 = fill("A", ntest)
ytest2 = fill("B", ntest)
ytest3 = fill("C", ntest)

ytest = vcat(ytest1, ytest2, ytest3)

knn = KNNClassifier(weights=:distance)

f,_,_ = fit(knn, 1, x, y)

f2,_,_ = fit(knn, 1, x, y, w)
@test f2[3] == w

p = predict(knn, f, xtest)
p2 = predict(knn, f2, xtest)

@test p[1] isa UnivariateFinite
@test p2[1] isa UnivariateFinite

p = predict_mode(knn, f, xtest)
p2 = predict_mode(knn, f2, xtest)

@test sum(p .== ytest)/length(ytest) == 1.0
@test sum(p2 .== ytest)/length(ytest) == 1.0

# === regression case

y1 = fill( 0.0, n)
y2 = fill( 2.0, n)
y3 = fill(-2.0, n)

y = vcat(y1, y2, y3)

knnr = KNNRegressor(weights=:distance)

f,_,_ = fit(knnr, 1, x, y)
f2,_,_ = fit(knnr, 1, x, y, w)

p = predict(knnr, f, xtest)
p2 = predict(knnr, f2, xtest)

@test all(p[1:ntest] .≈ 0.0)
@test all(p[ntest+1:2*ntest] .≈ 2.0)
@test all(p[2*ntest+1:end] .≈ -2.0)



# test metadata

infos = info_dict(knn)

# PKG
@test infos[:package_name] == "NearestNeighbors"

@test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
@test infos[:target_scitype] == AbstractVector{<:MLJBase.Finite}

infos[:docstring]

infos = info_dict(knnr)

@test infos[:input_scitype] == MLJBase.Table(MLJBase.Continuous)
@test infos[:target_scitype] == AbstractVector{MLJBase.Continuous}

infos[:docstring]

end
true
