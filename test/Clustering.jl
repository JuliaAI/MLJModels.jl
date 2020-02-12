module TestClustering

using MLJBase
using Test
using Random:seed!
import LinearAlgebra: norm
import Distances: evaluate
using RDatasets
using CategoricalArrays

# load code to be tested:
import MLJModels
import Clustering # MLJModels.Clustering_ now available for loading
using MLJModels.Clustering_

seed!(132442)

data = dataset("MASS", "crabs")
X = selectcols(data, [:FL, :RW, :CL, :CW, :BD])
y = selectcols(data, :Sp)

####
#### KMEANS
####

barekm = KMeans()

fitresult, cache, report = fit(barekm, 1, X)

R = matrix(transform(barekm, fitresult, X))

X_array = matrix(X)

# distance from first point to second center
@test R[1, 2] ≈ norm(view(X_array, 1, :) .- view(fitresult[1], :, 2))^2
@test R[10, 3] ≈ norm(view(X_array, 10, :) .- view(fitresult[1], :, 3))^2

p = predict(barekm, fitresult, X)

@test argmin(R[1, :]) == p[1]
@test argmin(R[10, :]) == p[10]

infos = info_dict(barekm)

@test infos[:package_name] == "Clustering"
@test infos[:is_pure_julia]
@test infos[:package_license] == "MIT"

@test infos[:input_scitype] == Table(Continuous)
@test infos[:output_scitype] == Table(Continuous)

infos[:docstring]

####
#### KMEDOIDS
####

barekm = KMedoids()

fitresult, cache, report = fit(barekm, 1, X)

R = matrix(transform(barekm, fitresult, X))

@test R[1, 2] ≈ evaluate(barekm.metric, view(X_array, 1, :), view(fitresult[1], :, 2))
@test R[10, 3] ≈ evaluate(barekm.metric, view(X_array, 10, :), view(fitresult[1], :, 3))

p = predict(barekm, fitresult, X)

@test all(report.assignments .== p)

# km = machine(barekm, X)
# fit!(km)

infos = info_dict(barekm)

@test infos[:input_scitype] == Table(Continuous)
@test infos[:output_scitype] == Table(Continuous)

end # module
true
