using MLJModels
using Pkg

@testset "registry project can be instantiated" begin
    filename, stream = mktemp()
    for line in MLJModels.Registry.registry_project()
        write(stream, line*"\n")
        @show line
    end
    close(stream)
    envname = dirname(filename)
    mv(filename, joinpath(envname, "Project.toml"); force=true)
    Pkg.activate(envname)
    # remove `@suppress` if debugging:
    @suppress Pkg.instantiate()

    # smoke test:
    @test !isempty(keys(Pkg.dependencies()))
end
