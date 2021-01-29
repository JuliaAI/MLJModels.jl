function check_registry()

    basedir = Registry.environment_path
    Pkg.activate(basedir)

    # Read Metadata.toml
    dict = TOML.parsefile(joinpath(basedir, "Metadata.toml"))

    # There will be warnings for ambiguous things, ignore them
    for (package, model_dict) in dict
        for (model, meta) in model_dict
            # check if new entry or changed entry, otherwise don't test
            key = "$package.$model"
            program = quote
                @load $model pkg=$package verbosity=-1
            end
            try
                eval(program)
                # add/refresh entry
                print(rpad("Entry for $key was loaded properly ✓", 79)*"\r")
            catch
                @error "⚠ there was an issue trying to load $key"
            end
        end
    end
end
