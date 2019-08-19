module TestClustering

using MLJBase
using CSV
using Test
using Random:seed!
import LinearAlgebra: norm
import Distances: evaluate

# load code to be tested:
import MLJModels
import Clustering # MLJModels.Clustering_ now available for loading
using MLJModels.Clustering_

seed!(132442)

task = load_crabs()

X, y = X_and_y(task)

####
#### KMEANS
####

barekm = KMeans()

fitresult, cache, report = MLJBase.fit(barekm, 1, X)

R = MLJBase.matrix(MLJBase.transform(barekm, fitresult, X))

X_array = MLJBase.matrix(X)

# distance from first point to second center
@test R[1, 2] ≈ norm(view(X_array, 1, :) .- view(fitresult, :, 2))^2
@test R[10, 3] ≈ norm(view(X_array, 10, :) .- view(fitresult, :, 3))^2

p = MLJBase.predict(barekm, fitresult, X)

@test argmin(R[1, :]) == p[1]
@test argmin(R[10, :]) == p[10]

# km = machine(barekm, X)
# fit!(km)

info(barekm)

####
#### KMEDOIDS
####

barekm = KMedoids()

fitresult, cache, report = MLJBase.fit(barekm, 1, X)

R = MLJBase.matrix(MLJBase.transform(barekm, fitresult, X))

@test R[1, 2] ≈ evaluate(barekm.metric, view(X_array, 1, :), view(fitresult, :, 2))
@test R[10, 3] ≈ evaluate(barekm.metric, view(X_array, 10, :), view(fitresult, :, 3))

p = MLJBase.predict(barekm, fitresult, X)

@test all(report.assignments .== p)

# km = machine(barekm, X)
# fit!(km)

info(barekm)

end # module
true
