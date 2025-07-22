using MLJModels
using Pkg
using Distributed
using Suppressor

@testset "loading all models in the MLJ Model Registry" begin
    # assemble the @load commands - one for each model in the registry:
    model_proxies =filter( models()) do proxy
        !proxy.is_wrapper
    end
    load_commands = map(model_proxies) do proxy
        :(MLJModels.@load $(proxy.name) pkg=$(proxy.package_name))
    end

    # make a clone of the MLJModel registry, to test `registry_project` method:
    filename, stream = mktemp()
    for line in MLJModels.registry_project()
        write(stream, line*"\n")
    end
    close(stream)
    registry = dirname(filename) # we need to rename project file to ..../Project.toml
    mv(filename, joinpath(registry, "Project.toml"); force=true)

    # open a new Julia process in which to activate the registry project and attempt to
    # load all models:
    id = only(addprocs(1))

    # define the programs to run in that process:
    # 1. To instantiate the registry environment:
    program1 = quote
        using Pkg
        Pkg.activate($registry)
        Pkg.instantiate()
        using MLJModels
        !isempty(keys(Pkg.dependencies()))
    end
    # 2. To load all the models:
    program2 = quote
        $(load_commands...)
        true
    end
    # remove `@suppress` to debug:
    @test @suppress remotecall_fetch(Main.eval, id, program1)
    @info "Attempting to load all MLJ Model Registry models into a Julia process. "
    @info "Be patient, this may take five minutes or so..."
    @test @suppress remotecall_fetch(Main.eval, id, program2)
    rmprocs(id)
end
