## MLJModels

Repository of [MLJ](https://github.com/alan-turing-institute/MLJ.jl)
model interfaces, as well as the MLJ model registry.

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
