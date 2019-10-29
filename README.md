## MLJModels

Repository of [MLJ](https://github.com/alan-turing-institute/MLJ.jl)
model interfaces, and for the MLJ model
registry.

[MLJ](https://github.com/alan-turing-institute/MLJ.jl) is a
machine learning toolbox written entirely in julia, but which
interfaces models written in julia and other languages.

[![Build Status](https://travis-ci.com/alan-turing-institute/MLJModels.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJModels.jl)

Selected packages which do not yet provide native
[MLJ](https://github.com/alan-turing-institute/MLJ.jl) implementations
of their models are provided "strap-on" implementations contained in
this repository. The implementation code is automatically loaded by
MLJ when the relevant package is imported (using "lazy-loading" provided by
[Requires.jl](https://github.com/MikeInnes/Requires.jl)).

MLJModels also provides a few "built-in" models, such as basic
transformers, immediately available to MLJ users. Do `using MLJ` or
`using MLJModels` and then:

- Run `localmodels()` to list built-in models (updated when external models are loaded with `@load`)

- Run `models()` to list all registered models, or see [this list](/src/registry/Models.toml).

MLJModels also houses the MLJ [Model Registry](/src/registry) which
administrators can use to register new models implementing the MLJ
interface, following [these
instructions](https://github.com/alan-turing-institute/MLJ.jl/blob/master/REGISTRY.md).

The following lists may also be reasonably complete:


### Built-in models

* Transformers (unsupervised models): `StaticTransformer`,
  `FeatureSelector`, `FillImputer`, `UnivariateStandardizer`, `Standardizer`,
  `UnivariateBoxCoxTransformer`, `OneHotEncoder`

* Constant predictors (for baselines and testing): `ConstantRegressor`,
  `ConstantClassifier`

* `KNNRegressor`


### External packages and models

Note that for some of these packages, the interface is incomplete; contributions are welcome!

* [Clustering.jl](https://github.com/JuliaStats/Clustering.jl)
    * `KMeans`, `KMedoids`
* [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl)
    * `DecisionTreeClassifier`, `DecisionTreeRegressor`
* [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl)
    * `GPClassifier`
* [GLM.jl](https://github.com/JuliaStats/GLM.jl)
    * `LinearRegressor`, `LinearBinaryClassifier`, `LinearCountRegressor`
* [LIBSVM.jl](https://github.com/mpastell/LIBSVM.jl) (**note**: _these models are effectively duplicated by the sklearn models below_.)
    * `LinearSVC`, `SVC`, `NuSVC`, `NuSVR`, `EpsilonSVR`, `OneClassSVM`
* [MLJLinearModels.jl](https://github.com/alan-turing-institute/MLJLinearModels.jl)
    * `LinearRegressor`, `RidgeRegressor`, `LassoRegressor`, `ElasticNetRegressor`, `QuantileRegressor`, `HuberRegressor`, `RobustRegressor`, `LADRegressor` (all with optional elastic net regression)
    * `LogisticClassifier`, `MultinomialClassifier` (with elastic net regularisation)
* [MultivariateStats.jl](https://github.com/mpastell/LIBSVM.jl)
    * `RidgeRegressor`, `PCA`, `KernelPCA`, `ICA`, `LDA` (multiclass)
* [NaiveBayes.jl](https://github.com/dfdx/NaiveBayes.jl)
    * `GaussianNBClassifier`, `MultinomialNBClassifier`, `HybridNBClassifier`
* [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl)
    * **SVM**: `SVMClassifier`, `SVMRegressor`, `SVMNuClassifier`, `SVMNuRegressor`, `SVMLClassifier`, `SVMLRegressor`,
    * **Linear Models** (regressors): `ARDRegressor`, `BayesianRidgeRegressor`, `ElasticNetRegressor`, `ElasticNetCVRegressor`, `HuberRegressor`, `LarsRegressor`, `LarsCVRegressor`, `LassoRegressor`, `LassoCVRegressor`, `LassoLarsRegressor`, `LassoLarsCVRegressor`, `LassoLarsICRegressor`, `LinearRegressor`, `OrthogonalMatchingPursuitRegressor`, `OrthogonalMatchingPursuitCVRegressor`, `PassiveAggressiveRegressor`, `RidgeRegressor`, `RidgeCVRegressor`, `SGDRegressor`, `TheilSenRegressor`
    * **Linear Models** (classifiers): `LogisticClassifier`, `LogisticCVClassifier`, `PerceptronClassifier`, `RidgeClassifier`, `RidgeCVClassifier`, `PassiveAggressiveClassifier`, `SGDClassifier`
    * **Gaussian Processes**: `GaussianProcessRegressor`, `GaussianProcessClassifier`
    * **Ensemble**: `AdaBoostRegressor`, `AdaBoostClassifier`, `BaggingRegressor`, `BaggingClassifier`, `GradientBoostingRegressor`, `GradientBoostingClassifier`, `RandomForestRegressor`, `RandomForestClassifier`
    * **Naive Bayes**: `GaussianNB`, `MultinomialNB`, `ComplementNB`
* [XGBoost.jl](https://github.com/dmlc/XGBoost.jl)
    * `XGBoostRegressor`, `XGBoostClassifier`, `XGBoostCount`
* [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl)
    * `KNNClassifier`, `KNNRegressor`
