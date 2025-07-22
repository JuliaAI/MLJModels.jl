# The MLJ Model Registry

The *MLJ Model Registry*, also known as the *model metadata database*,
consists of the files in this directory:

- [Project.toml](Project.toml): Project file for a Julia package environment whose
  dependencies are all packages providing models with metadata searchable by the MLJ user
  after running `using MLJ` (or just `using MLJModels`).

- [Metadata.toml](Metadata.toml): The detailed model metadata, keyed on package name.

MLJ developers should use
[MLJModelRegistryTools.jl](https://github.com/JuliaAI/MLJModelRegistryTools.jl) to make
updates and corrections, following [the
instructions](https://juliaai.github.io/MLJModelRegistryTools.jl/stable/registry_management_tools/#Registry-management-tools) there.
