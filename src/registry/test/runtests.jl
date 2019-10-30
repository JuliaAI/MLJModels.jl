import Pkg
import Pkg:TOML
using DelimitedFiles
using MLJ

curdir = @__DIR__

Pkg.activate(joinpath(curdir, ".."))

# Read Metadata.toml
dict = TOML.parsefile(joinpath(curdir, "..", "Metadata.toml"))

# There will be warnings for ambiguous things, ignore them
for (package, model_dict) in dict
    for (model, meta) in model_dict
        # check if new entry or changed entry, otherwise don't test
        key = "$package.$model"
        try
            load(model; pkg=package, allow_ambiguous=true)
            # add/refresh entry
            print("Entry for $key was loaded properly ✓    \r")
        catch
            println("⚠ there was an issue trying to load $key")
        end
    end
end
