# MLJModels

[![Build Status](https://github.com/alan-turing-institute/MLJModels.jl/workflows/CI/badge.svg)](https://github.com/alan-turing-institute/MLJModels.jl/actions)

Repository of selected models for use in the
[MLJ](https://github.com/alan-turing-institute/MLJ.jl) MLJ machine
learning framework, without the need to import third party packages; and the
home of the MLJ model registry.

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

**Administrator instructions.** To register all the models in
GreatNewPackage with MLJ:

- In the dev branch of a clone of the dev branch of MLJModels, change
  to the `/src/registry/` directory and, in Julia, activate the
  environment specified by the Project.toml there, after checking the
  [compat] conditions there are up to date. **Do not use**
  `Revise`. If supporting Julia versions earlier than 1.3, you will
  need to run a version of Julia earlier or equal to 1.3 to avoid
  `@var_str` problems in the metadata (Julia 1.0 can't read things
  like `var"_s24"`).
  
- Add `GreatNewPackage` to the environment.

- In some environment to which your MLJModels clone has been added
  (using `Pkg.dev`) execute `using MLJModels; @update`. This updates
  `src/registry/Metadata.toml` and `src/registry/Models.toml` (the
  latter is generated for convenience and not used by MLJ). If the new
  package does not appear in the list of packages generated, you may
  have to force precompilation of MLJModels.
  
- Test that interfaces load with `MLJModels.check_registry()`

- Quit your REPL session, whose namespace is now polluted.

- *Note.* that your local MLJModels will not immediately adopt the
  updated registry because that requires pre-compilation; for
  technical reasons the registry is not loaded in `__init__`()`.

- Push your changes to an appropriate branch of MLJModels to make
  the updated metadata available to users of the next MLJModels tagged
  release.
