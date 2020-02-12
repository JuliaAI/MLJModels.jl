# MLJModels

[![Build Status](https://travis-ci.com/alan-turing-institute/MLJModels.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJModels.jl)

Repository of some [MLJ](https://github.com/alan-turing-institute/MLJ.jl)
model _interfaces_, as well as the **MLJ model registry**.

## About

This repository contains interfaces for a few "essential" Julia ML packages which do not yet provide their native interface to MLJ.
It also provides a few "built-in" models, such as basic transformers (One Hot Encoder, Standardizer, ...).

You can do `using MLJ` or `using MLJModels` and:

- `localmodels()` to list built-in models (updated when external models are loaded with `@load`)
- `models()` to list all registered models, or see [this list](/src/registry/Models.toml).

MLJModels houses the MLJ **model registry**: the list of models that can be called from MLJ using `@load`.
Package developers can register new models by implementing the MLJ interface in their package and following the instructions below.

## Writing an interface to MLJ

In the instructions below, we assume you have a **registered** package `YourPackage.jl` which implements some models; that you would like to register these models with MLJ, and that you have a rough understanding of how things work with MLJ.
In particular you are expected to be familiar with

* what [Scientific Types](https://github.com/alan-turing-institute/ScientificTypes.jl) are
* what `Probabilistic`, `Deterministic` and `Unsupervised` models are
* the fact that MLJ generally works with tables rather than bare bone
  matrices. Here a *table* is a container satisfying the
  [Tables.jl](https://github.com/JuliaData/Tables.jl) API (e.g., DataFrame, JuliaDB table, CSV file, named tuple of equi-length vectors)
* [CategoricalArrays.jl](https://github.com/JuliaData/CategoricalArrays.jl) (if working with finite discrete data)

If you're not familiar with any one of these points, please refer to the general [MLJ docs](https://alan-turing-institute.github.io/MLJ.jl/dev/) and specifically, read the [detailed documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/#Adding-Models-for-General-Use-1) for adding models.

*But tables don't make sense for my model!* If a case can be made that
tabular input does not make sense for your particular model, then MLJ can
still handle this; you just need to define a non-tabular
`input_scitype` trait. However, you should probably open an issue to
clarify the appropriate declaration. The discussion below assumes
input data is tabular.

### Overview

Writing an interface is fairly straightforward: just create a file or a module in your package including

* a `using MLJBase` or `import MLJBase` statement
* MLJ-compatible model types and constructors,
* implementation of the `fit`, `predict`/`transform` and optionally `fitted_params` for your models,
* metadata for your package and for each of your models

We give some details for each step below with, each time, a few examples that you can mimic.
The instructions are intentionally brief; refer to the [full documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/#Adding-Models-for-General-Use-1) for additional details.

### Model type and constructor

MLJ-compatible constructors for your models need to meet the following requirements:

* be `mutable struct`,
* be subtypes of `MLJBase.Probabilistic` or `MLJBase.Deterministic` or `MLJBase.Unsupervised`,
* have fields corresponding exclusively to hyperparameters,
* have a keyword constructor assigning default values to all
  hyperparameters.

It is recommended that you use the `@mlj_model` macro from `MLJBase` to
declare a (non parametric) model type:

```julia
@mlj_model mutable struct YourModel <: MLJBase.Deterministic
    a::Float64 = 0.5::(_ > 0)
    b::String  = "svd"::(_ in ("svd","qr"))
end
```

That macro specifies:

* A keyword constructor (here `YourModel(; a=..., b=...)`),
* Default values for the hyperparameters,
* Constraints on the hyperparameters where `_` refers to a value passed.

Further to the last point, `a::Float64 = 0.5::(_ > 0)` indicates that the field `a` is a `Float64`, takes `0.5` as default value, and expects its value to be positive.

If you decide **not** to use the `@mlj_model` macro (e.g. in the case of a parametric type), you will need to write a keyword constructor and a `clean!` method:

```julia
mutable struct YourModel <: MLJBase.Deterministic
    a::Float64
end
function YourModel(; a=0.5)
    model   = YourModel(a)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end
function clean!(m::YourModel)
    warning = ""
    if m.a <= 0
        warning *= "Parameter `a` expected to be positive, resetting to 0.5"
        m.a = 0.5
    end
    return warning
end
```

**Additional notes**:

- Please type all your fields if possible,
- Please prefer `Symbol` over `String` if you can (e.g. to pass the name of a solver),
- Please add constraints to your fields even if they seem obvious to you,,
- Your model may have 0 fields, that's fine.

**Examples**:

- [KNNClassifier](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/NearestNeighbors.jl#L62-L69) which uses `@mlj_model`,
- [XGBoostRegressor](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/XGBoost.jl#L17-L161) which does not.

### Fit

The implementation of `fit` will look like

```julia
function MLJBase.fit(m::YourModel, verbosity::Int, X, y, w=nothing)
    # body ...
    return (fitresult, cache, report)
end
```

where `y` should only be there for a supervised model and `w` for a supervised model that supports sample weights.
You **must** type `verbosity` to `Int` and you **must not** type `X`, `y` and `w` (MLJ handles that).

#### Regressor

In the body of the `fit` function, you should assume that `X` is a table and that `y` is an `AbstractVector` (for multitask regression it may be a table).

Typical steps in the body of the `fit` function will be:

* forming a matrix-view of the data, possibly transposed if your model expects a `p x n` formalism (MLJ assumes columns are features by default i.e. `n x p`), use `MLJBase.matrix` for this,
* passing the data to your model,
* returning the results as a tuple `(fitresult, cache, report)`.

The `fitresult` part should contain everything that is needed at the
`predict` or `transform` step, it should not be expected to be
accessed by users.  The `cache` should be left to `nothing` for now.
The `report` should be a `NamedTuple` with any auxiliary useful
information that a user would want to know about the fit (e.g.,
feature rankings). See more on this below.

**Example**: GLM's [LinearRegressor](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/GLM.jl#L95-L105)


#### Classifier

For a classifier, the steps are fairly similar to a regressor with two particularities:

1. `y` will be a categorical vector and you will typically want to use the integer encoding of `y` instead of the raw labels; use `MLJBase.int` for this,
1.  You will need to pass the full pool of target labels (not just
   those observed in the training data) and additionally, in the
   `Deterministic` case, the encoding, to make these available to
   `predict`. A simple way to do this is to pass `y[1]` in the
   `fitresult`, for then `MLJBase.classes(y[1])` is a complete list of
   possible categorical elements, and `d = MLJBase.decoder(y[1])` is a
   method for recovering categorical elements from their integer
   representations (e.g., `d(2)` is the categorical element with `2`
   as encoding).

**Examples**:

-  GLM's [BinaryClassifier](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/GLM.jl#L119-L131) (`Probabilistic`)

- LIBSVM's [SVC](https://github.com/alan-turing-institute/MLJModels.jl/blob/master/src/LIBSVM.jl) (`Deterministic`)


#### Transformer

Nothing special for a transformer.

**Example**: our [FillImputer](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/builtins/Transformers.jl#L54-L64)

### Fitted parameters

There is a function you can optionally implement which will return the
learned parameters of your model for purposes of user-inspection. For
instance, in the case of a linear regression, the user may want to get
direct access to the coefficients and intercept. This should be as human and
machine readable as practical (not a graphical representation) and the
information should be combined in the form of a named tuple.

The function will always look like:

```julia
function MLJBase.fitted_params(model::YourModel, fitresult)
    # extract what's relevant from `fitresult`
    # ...
    # then return as a NamedTuple
    return (learned_param1 = ..., learned_param2 = ...)
end
```

**Example**: for [GLM models](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/GLM.jl#L133-L137)


### Summary of user interface points (or, What to put where?)

Recall that the `fitresult` returned as part of `fit` represents
everything needed by `predict` (or `transform`) to make new
predictions. It is not intended to be directly inspected by the
user. Here is a summary of the interface points for users that your
implementation creates:

- Use `fitted_params` to expose *learned parameters*, such as linear
  coefficients, to the user in a machine and human readable form (for
  re-use in another model, for example).
- Use the fields of your model struct for *hyperparameters*, i.e.,
  those parameters declared by the user ahead of time that generally
  affect the outcome of training. It is okay to add "control"
  parameters (such a specifying an `acceleration` parameter specifying
  computational resources, as
  [here](https://github.com/alan-turing-institute/MLJ.jl/blob/master/src/ensembles.jl#L193)).
- Use `report` to return *everything else*, including model-specific
  *methods* (or other callable objects). This includes: feature rankings,
  decision boundaries, SVM support vectors, clustering centres,
  methods for visualizing training outcomes, methods for saving
  learned parameters in a custom format, degrees of freedom, deviance,
  etc. If there is a performance cost to extra functionality you want
  to expose, the functionality can be toggled on/off through a
  hyperparameter, but this should otherwise be avoided. For, example,
  in a decision tree model `report.print_tree(depth)` might generate
  a pretty tree representation of the learned tree, up to the
  specified `depth`.

### Predict/Transform

The implementation of `predict` (for a supervised model) or
`transform` (for an unsupervised one) will look like:

```julia
function MLJBase.predict(m::YourModel, fitresult, Xnew)
    # ...
end
```

Where `Xnew` should be expected to be a table and part of the logic in `predict` or `transform` may be similar to that in `fit`.

The values returned should be:

* (**Deterministic**): a vector of values (or Table if multi-target)
* (**Probabilistic**): a vector of `Distribution` objects, for classifiers in particular, a vector of `UnivariateFinite`
* (**Transformer**): a table

In the case of a `Probabilistic` model, you may further want to
implement a `predict_mean` or a `predict_mode`. However,
MLJBase provides fallbacks, defined in terms of `predict`, whose performance may suffice.


**Examples**

- Deterministic regression: [KNNRegressor](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/NearestNeighbors.jl#L124-L145)
- Probabilistic regression: [LinearRegressor](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/GLM.jl#L154-L158) and the [`predict_mean`](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/GLM.jl#L144-L147)
- Probabilistic classification: [LogisticClassifier](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/GLM.jl#L165-L168)

### Metadata

Adding metadata for your model(s) is crucial for the discoverability of your package and its models and to make sure your model is used with data it can handle.
You should use the `metadata_model` and `metadata_pkg` functionalities from `MLJBase` to do this:

```julia
const ALL_MODELS = Union{YourModel1, YourModel2, ...}

metadata_pkg.(ALL_MODELS
    name = "YourPackage",
    uuid = "6ee0df7b-...", # see your Project.toml
    url  = "https://...",  # URL to your package repo
    julia = true,          # is it written entirely in Julia?
    license = "MIT",       # your package license
    is_wrapper = false,    # does it wrap around some other package?
)

# Then for each model,
metadata_model(YourModel1,
    input   = Table(Continuous),  # what input data is supported?
    target  = AbstractVector{Continuous}, # for a supervised model, what target?
    output  = Table(Continuous),  # for an unsupervised, what output?
    weights = false,                              # does the model support sample weights?
    descr   = "A short description of your model"
	path    = "YourPackage.ModuleContainingModelStructDefinition.YourModel1"
    )
```

**Examples**:

- package metadata
  - [GLM](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/GLM.jl#L179-L186)
  - [MLJLinearModels](https://github.com/alan-turing-institute/MLJLinearModels.jl/blob/289a373a8357c4afc191711d0218aa1523e97f70/src/mlj/interface.jl#L91-L97)
- model metadata
  - [LinearRegressor](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/GLM.jl#L188-L193)
  - [DecisionTree](https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/DecisionTree.jl#L225-L229)
  - [A series of regressors](https://github.com/alan-turing-institute/MLJLinearModels.jl/blob/289a373a8357c4afc191711d0218aa1523e97f70/src/mlj/interface.jl#L105-L111)

---

## Register a model to the model registry

This is the final step, once you have a working interface with MLJ in your package.

1. fork this repository and clone it locally, then, in your REPL,
1. cd to `MLJModels/src/registry` and activate that environment,
1. add `YourPackage` to the environment,
1. cd back to `MLJModels` and activate that environment,
1. run `using MLJModels; @update`.

The last step updates

- `MLJModels/src/registry/Metadata.toml` and
- `MLJModels/src/registry/Models.toml`

#### Does it work?

1. in the activated environment from before, add `MLJ` and `YourPackage`,
1. run `using MLJModels; @load YourModel pkg=YourPackage`

Assuming that worked, commit and push your changes then open a PR on MLJModels' **`dev`** branch.

MLJ maintainers will merge your PR once they've had a chance at making sure your interface works and your model(s) are appropriately tested.
[
