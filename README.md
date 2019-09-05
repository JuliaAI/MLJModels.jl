## MLJModels

[![Build Status](https://travis-ci.com/alan-turing-institute/MLJModels.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJModels.jl)

Selected packages which do not yet provide native
[MLJ](https://github.com/alan-turing-institute/MLJ.jl) implementations
of their models are provided "strap-on" implementations contained in
this repository. The implementation code is automatically loaded by
MLJ when the relevant package is imported (using "lazy-loading" - see
[Requires.jl](https://github.com/MikeInnes/Requires.jl)).


### Packages and models

Note that for some of these packages, the interface is incomplete; contributions are welcome!

* [Clustering.jl](https://github.com/JuliaStats/Clustering.jl)
    * `KMeans`, `KMedoids`
* [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl)
    * `DecisionTreeClassifier`, `DecisionTreeRegressor`
* [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl)
    * `GPClassifier`
* [GLM.jl](https://github.com/STOR-i/GaussianProcesses.jl)
    * `LinearRegressor`, `LinearBinaryClassifier`, `LinearCountRegressor`
* [LIBSVM.jl](https://github.com/mpastell/LIBSVM.jl) (**note**: _these models are effectively duplicated by the sklearn models below_.)
    * `LinearSVC`, `SVC`, `NuSVC`, `NuSVR`, `EpsilonSVR`, `OneClassSVM`
* [MultivariateStats.jl](https://github.com/mpastell/LIBSVM.jl)
    * `RidgeRegressor`, `PCA`, `KernelPCA`, `ICA`
* [NaiveBayes.jl](https://github.com/dfdx/NaiveBayes.jl)
    * `GaussianNBClassifier`, `MultinomialNBClassifier`, `HybridNBClassifier`
* [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl)
    * **SVM**: `SVMClassifier`, `SVMRegressor`, `SVMNuClassifier`, `SVMNuRegressor`, `SVMLClassifier`, `SVMLRegressor`,
    * **Linear Models** (regressors): `ARDRegressor`, `BayesianRidgeRegressor`, `ElasticNetRegressor`, `ElasticNetCVRegressor`, `HuberRegressor`, `LarsRegressor`, `LarsCVRegressor`, `LassoRegressor`, `LassoCVRegressor`, `LassoLarsRegressor`, `LassoLarsCVRegressor`, `LassoLarsICRegressor`, `LinearRegressor`, `OrthogonalMatchingPursuitRegressor`, `OrthogonalMatchingPursuitCVRegressor`, `PassiveAggressiveRegressor`, `RidgeRegressor`, `RidgeCVRegressor`, `SGDRegressor`, `TheilSenRegressor`
    * **Gaussian Processes**: `GaussianProcessRegressor`
    * **Ensemble**: `AdaBoostRegressor`, `BaggingRegressor`, `GradientBoostingRegressor`, `RandomForestRegressor`
* [XGBoost.jl](https://github.com/dmlc/XGBoost.jl)
    * `XGBoostRegressor`, `XGBoostClassifier`, `XGBoostCount`
