"""
    MLJModels.registry_project()

Experimental, private method.

Return, as a `Vector{String}`, the lines of the Project.toml associated with the MLJ Model
Registry. This Project.toml file lists as dependencies all packages that provide
registered models.

Using this method, one can create a clone of the MLJ Model Registry environment and
activate it, as in the example below. This may be useful in MLJ integrations tests.

```julia
mkdir("MyEnv")
open("MyEnv/Project.toml", "w") do file
   for line in MLJModels.Registry.registry_project()
        write(file, line*"\n")
    end
end
"""
registry_project() = open(REGISTRY_PROJECT) do io
    readlines(io)
end
