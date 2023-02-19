# MLJModels

[![Build Status](https://github.com/alan-turing-institute/MLJModels.jl/workflows/CI/badge.svg)](https://github.com/alan-turing-institute/MLJModels.jl/actions)

Repository of the "built-in" models available for use in the
[MLJ](https://github.com/alan-turing-institute/MLJ.jl) MLJ machine
learning framework; and the home of the MLJ model registry.

For instructions on integrating a new model with MLJ visit
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/)


### Contents

 - [Who is this repo for?](#who-is-this-repo-for)
 - [What is provided here?](#what-is-provided-here)
 - [Instructions for updating the MLJ model registry](#instructions-for-updating-the-mlj-model-registry)

## Who is this repo for?

General users of the MLJ machine learning platform should refer to
[MLJ home page](https://alan-turing-institute.github.io/MLJ.jl/dev/)
for usage and installation instructions. MLJModels is a dependency of
MLJ that the general user can ignore.

This repository is for developers wishing to
[register](#instructions-for-updating-the-mlj-model-registry) new MLJ
model interfaces, whether they be:

- implemented **natively** in a
  package providing the core machine learning algorithm, as in
  [`EvoTrees.jl`](https://github.com/Evovest/EvoTrees.jl/blob/master/src/MLJ.jl); or
  
- implemented in a separate **interface package**, such as
  [MLJDecisionTreeInterface.jl](https://github.com/alan-turing-institute/MLJDecisionTreeInterface.jl).

It also a place for developers to add models (mostly transformers)
such as `OneHotEncoder`, that are exported for "built-in" use in
MLJ. (In the future these models may live in a separate package.)

To list *all* model interfaces currently registered, do `using MLJ` or
`using MLJModels` and run:

- `localmodels()` to list built-in models (updated when external models are loaded with `@load`)

- `models()` to list all registered models, or see [this list](/src/registry/Models.toml).

Recall that an interface is loaded from within MLJ, together with the
package providing the underlying algorithm, using the syntax `@load
RidgeRegressor pkg=GLM`, where the `pkg` keyword is only necessary in
ambiguous cases.


## What is provided here?

MLJModels contains:

- transformers to be pre-loaded into MLJ, located at
  [/src/builtins](/src/builtins), such as `OneHotEncoder`
  and `ConstantClassifier`. 

- the MLJ [model registry](src/registry/Metadata.toml), listing all
  models that can be called from MLJ using `@load`. Package developers
  can register new models by implementing the MLJ interface in their
  package and following [these
  instructions](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/).
  

## Instructions for updating the MLJ model registry

Generally model registration is performed by administrators. If you
have an interface you would like registered, open an issue
[here](https://github.com/alan-turing-institute/MLJ.jl/issues). 

**Administrator instructions.** These are given in the
`MLJModels.@update` document string. After registering the model, make a PR to MLJ 
updating [this dictionary of model descriptors](https://github.com/alan-turing-institute/MLJ.jl/blob/dev/docs/ModelDescriptors.toml) 
to ensure the new models appear in the right places in MLJ's [Model Browser](https://alan-turing-institute.github.io/MLJ.jl/dev/model_browser/#Model-Browser)
