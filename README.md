# MLJModels

[![Build Status](https://travis-ci.com/alan-turing-institute/MLJModels.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJModels.jl)

Repository of selected
[MLJ](https://github.com/alan-turing-institute/MLJ.jl) model
interfaces, and home of the MLJ model registry.

For instructions on integrating a new model with MLJ visit
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/)


### Contents

 - [Who is this repo for?](#who-is-this-repo-for)
 - [What is provided here?](#what-is-provided-here)
 - [Instructions for updating the MLJ model registry](#instructions-for-updating-the-mlj-model-registry)

## Who is this repo for?

General users of the MLJ machine learning platform should refer to
[MLJ home page](https://github.com/alan-turing-institute/MLJ.jl) for
usage and installation instructions. While MLJ users are required to
have MLJModels installed in their project environment, they can
otherwise ignore it.

This repository is for developers wishing to: 

- add a model interface for a third party package that does not provide, or is
  not willing to provide, an MLJ interface natively
  
- [register](#instructions-for-updating-the-mlj-model-registry) new
  MLJ interfaces, whether they be defined here or in a third party
  package

To list *all* model interfaces currently registered, do `using MLJ` or `using MLJModels` and run:

- `localmodels()` to list built-in models (updated when external models are loaded with `@load`)

- `models()` to list all registered models, or see [this list](/src/registry/Models.toml).

Recall that an interface is loaded from within MLJ, together with the
package providing the underlying algorithm, using the syntax `@load
RidgeRegressor pkg=GLM`, where the `pkg` keyword is only necessary in
ambiguous cases.


## What is provided here?

MLJModels contains:

- interfaces, under [/src/](/src/), for "essential" Julia machine
  learning packages which do not yet provide, or are unlikely to
  provide, native MLJ model interfaces. The bulk of these are
  [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) models.
  
- a few models that are pre-loaded into MLJ, located at
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

To register all the models in GreatNewPackage with MLJ:

- In the dev branch of a clone of the dev branch of MLJModels, change
  to the `/src/registry/` directory and, in Julia, activate the
  environment specified by the Project.toml there, after checking the
  [compat] conditions there are up to date. **Do not use** `Revise`.
  
- Add `GreatNewPackage` to the environment.

- In some environment to which your MLJModels clone has been added
  (using `Pkg.dev`) execute `using MLJModels; @update`. This updates
  `src/registry/Metadata.toml` and `src/registry/Models.toml` (the
  latter is generated for convenience and not used by MLJ).
  
- Test that interfaces load with `MLJModels.check_registry()`

- Quit your REPL session, whose namespace is now polluted.

- *Note.* that your local MLJModels will not immediately adopt the
  updated registry because that requires pre-compilation; for
  technical reasons the registry is not loaded in `__init__`()`.

- Push your changes to an appropriate branch of MLJModels to make
  the updated metadata available to users of the next MLJModels tagged
  release.
