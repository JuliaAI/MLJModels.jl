# MLJModels.jl

[![Build Status](https://github.com/JuliaAI/MLJModels.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJModels.jl/actions)
[![codecov](https://codecov.io/gh/JuliaAI/MLJModels.jl/graph/badge.svg?token=KgarnnCc0K)](https://codecov.io/gh/JuliaAI/MLJModels.jl)

Home of the [MLJ](https://juliaml.ai) Model Registry and tools for model search and model code loading. 

For instructions on integrating a new model into MLJ visit
[here](https://juliaai.github.io/MLJModelInterface.jl/stable/).


### Contents

 - [Who is this repo for?](#who-is-this-repo-for)
 - [How to register new models](#how-to-register-new-models)
 - [What is provided here?](#what-is-provided-here)

## Who is this repo for?

Newcomers to MLJ should refer to [this page](https://juliaml.ai) for usage and
installation instructions. MLJModels.jl is a dependency of MLJ that the general user can
ignore.

This repository is for developers maintaining:

- The [MLJ Model Registry](/src/registry), a database of packages implementing the MLJ
  interface for machine learning models, together with metadata about those models.

- MLJ tools for searching the database (`models(...)` and `matching(...)`) and for loading
  model code (`@load`, `@iload`).

## How to register new models

The model registry lives at "/src/registry" but
is maintained using
[MLJModelRegistryTools.jl](https://juliaai.github.io/MLJModelRegistryTools.jl/dev/).

New MLJ model interfaces can be implemented either:

- **natively** in a package providing the core machine learning algorithm, as in
  [`EvoTrees.jl`](https://github.com/Evovest/EvoTrees.jl/blob/master/src/MLJ.jl); or
  
-  in a separate **interface package**, such as
  [MLJDecisionTreeInterface.jl](https://github.com/JuliaAI/MLJDecisionTreeInterface.jl).

In either case, the package providing the implementation needs to be added to the MLJ
Model Registry to make it discoverable by MLJ users, and to make the model metadata
searchable. To register a package, prepare a pull request to MLJModels.jl by following [these instructions](https://juliaai.github.io/MLJModelRegistryTools.jl/dev/registry_management_tools/#Registry-management-tools).

Currently, after registering the model, one must also make a PR to MLJ updating [this
dictionary of model
descriptors](https://github.com/JuliaAI/MLJ.jl/blob/dev/docs/ModelDescriptors.toml) to
ensure the new models appear in the right places in MLJ's [Model
Browser](https://JuliaAI.github.io/MLJ.jl/dev/model_browser/#Model-Browser)

To list *all* model interfaces currently registered, do `using MLJ` or `using MLJModels`
and run `models()` to list all registered models.

Recall that an interface is loaded from within MLJ, together with the
package providing the underlying algorithm, using the syntax `@load
RidgeRegressor pkg=GLM`, where the `pkg` keyword is only necessary in
ambiguous cases.

## What is provided here?

The actual MLJ Model Registry consists of the TOML files in [this
directory](/src/registry). A few models available for immediate use in MLJ (without
loading model code using `@load`) are also provided by this package, under "/src/builtins"
but these may be moved out in the future.

### Historical note

Older versions of MLJModels.jl contained some of the models now residing at
[MLJTransforms.jl](https://github.com/JuliaAI/MLJTransforms.jl/tree/dev). Even older
versions provided implementations of all the non-native implementations of the
MLJ interface.
