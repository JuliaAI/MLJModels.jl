module TestKNN

# using Revise
using Test
using MLJModels
import MLJBase

Xtr = [4 2 5 3;
       2 1 6 0.0];

@test MLJModels.KNN.distances_and_indices_of_closest(3,
        MLJModels.KNN.euclidean, Xtr, [1, 1])[2] == [2, 4, 1]

X = Xtr' |> collect
y = Float64[2, 1, 3, 8]
knn = KNNRegressor(K=3)
allrows = 1:4
Xtable = MLJBase.table(X)
fitresult, cache, report = MLJBase.fit(knn, 0, X, y); 

r = 1 + 1/sqrt(5) + 1/sqrt(10)
Xtest = MLJBase.table([1.0 1.0])
ypred = (1 + 8/sqrt(5) + 2/sqrt(10))/r
@test MLJBase.predict(knn, fitresult, Xtest)[1] ≈ ypred

knn.K = 2
fitresult, cache, report = MLJBase.update(knn, 0, fitresult, cache, X, y); 
@test MLJBase.predict(knn, fitresult, Xtest)[1] !=  ypred

MLJBase.info(knn)

N =100
X = (x1=rand(N), x2=rand(N), x3=rand(N))
y = 2X.x1  - X.x2 + 0.05*rand(N)


end
true
