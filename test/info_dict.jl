module TestInfo

using MLJModels
using MLJModelInterface
using Test
using OrderedCollections
using ScientificTypes

const MMI = MLJModelInterface

mutable struct DummyProb <: Probabilistic
    an_int::Int
    a_float::Float64
    a_vector::Vector{Float64}
    untyped
end

MMI.load_path(::Type{DummyProb})        = "GreatPackage.MLJ.DummyProb"
MMI.input_scitype(::Type{DummyProb})    = Table(Finite)
MMI.target_scitype(::Type{DummyProb})   = AbstractVector{<:Continuous}
MMI.is_pure_julia(::Type{DummyProb})    = true
MMI.supports_weights(::Type{DummyProb}) = true
MMI.package_name(::Type{DummyProb})     = "GreatPackage"
MMI.package_uuid(::Type{DummyProb}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MMI.package_url(::Type{DummyProb})     = "https://mickey.mouse.org"
MMI.package_license(::Type{DummyProb}) = "MIT"
MMI.hyperparameter_ranges(::Type{DummyProb}) =
          (range(Int, :an_int, values=[1,2]),
           range(Float64, :a_float, lower=1, upper=2),
           range(Vector{Float64}, :a_vector, values=[[1.0], [2.0]]),
           nothing)
MMI.predict(::DummyProb, fr, X) = nothing

mutable struct DummyDeterm <: Deterministic end

MMI.load_path(::Type{DummyDeterm})        = "GreatPackage.MLJ.DummyDeterm"
MMI.input_scitype(::Type{DummyDeterm})    = Table(Finite)
MMI.target_scitype(::Type{DummyDeterm})   = AbstractVector{<:Continuous}
MMI.is_pure_julia(::Type{DummyDeterm})    = true
MMI.supports_weights(::Type{DummyDeterm}) = true
MMI.package_name(::Type{DummyDeterm})     = "GreatPackage"
MMI.package_uuid(::Type{DummyDeterm}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MMI.package_url(::Type{DummyDeterm}) = "https://mickey.mouse.org"
MMI.package_license(::Type{DummyDeterm}) = "MIT"
MMI.predict(::DummyDeterm, fr, X) = nothing

mutable struct DummyInt <: Interval end
MMI.load_path(::Type{DummyInt}) = "GreatPackage.MLJ.DummyInt"
MMI.input_scitype(::Type{DummyInt}) = Table(Finite)
MMI.target_scitype(::Type{DummyInt}) = AbstractVector{<:Continuous}
MMI.is_pure_julia(::Type{DummyInt}) = true
MMI.supports_weights(::Type{DummyInt}) = true
MMI.package_name(::Type{DummyInt}) = "GreatPackage"
MMI.package_uuid(::Type{DummyInt}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MMI.package_url(::Type{DummyInt}) = "https://mickey.mouse.org"
MMI.package_license(::Type{DummyInt}) = "MIT"
MMI.predict(::DummyInt, fr, X) = nothing

mutable struct DummyUnsup <: Unsupervised end
MMI.load_path(::Type{DummyUnsup}) = "GreatPackage.MLJ.DummyUnsup"
MMI.input_scitype(::Type{DummyUnsup}) = Table(Finite)
MMI.output_scitype(::Type{DummyUnsup}) = AbstractVector{<:Continuous}
MMI.is_pure_julia(::Type{DummyUnsup}) = true
MMI.supports_weights(::Type{DummyUnsup}) = true
MMI.package_name(::Type{DummyUnsup}) = "GreatPackage"
MMI.package_uuid(::Type{DummyUnsup}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MMI.package_url(::Type{DummyUnsup}) = "https://mickey.mouse.org"
MMI.package_license(::Type{DummyUnsup}) = "MIT"
MMI.transform(::DummyUnsup, fr, X) = nothing

# helper to check that dictionary `d1` agrees on the key of `d2`,
# which must be subset of those of `d1`:
function _issubset(d1, d2)
    k1 = keys(d1)
    issubset(k1, keys(d2)) || return false
    return all(k1) do k
        d1[k] == d2[k]
    end
end
d1 = Dict('a'=> 1, 'b' => 2)
d2 = Dict('a' => 1, 'b' => 2, 'c' => 3)
@test _issubset(d1, d2)
d2['b'] = 4
@test !_issubset(d1, d2)

@testset "info on probabilistic models" begin
    d = LittleDict{Symbol,Any}(
            :name             => "DummyProb",
            :load_path        => "GreatPackage.MLJ.DummyProb",
            :is_pure_julia    => true,
            :package_uuid     => "6f286f6a-111f-5878-ab1e-185364afe411",
            :package_name     => "GreatPackage",
            :package_license  => "MIT",
            :input_scitype    => Table(Finite),
            :output_scitype   => Unknown,
            :supports_weights => true,
            :supports_class_weights => false,
            :supports_online  => false,
            :target_scitype   => AbstractVector{<:Continuous},
            :prediction_type  => :probabilistic,
            :package_url      => "https://mickey.mouse.org",
            :is_supervised    => true,
            :is_wrapper       => false,
            :docstring        => "DummyProb from GreatPackage.jl.\n[Documentation](https://mickey.mouse.org).",
            :implemented_methods  => [:predict, ],
            :hyperparameters  => (:an_int, :a_float, :a_vector, :untyped),
            :hyperparameter_ranges =>
                (range(Int, :an_int, values=[1,2]),
                 range(Float64, :a_float, lower=1, upper=2),
                 range(Vector{Float64}, :a_vector, values=[[1.0], [2.0]]),
                 nothing))
    @test _issubset(d, MLJModels.info_dict(DummyProb))
    @test _issubset(d, MLJModels.info_dict(DummyProb(42, 3.14, [1.0, 2.0], :cow)))
end

@testset "info on deterministic models" begin
    d = LittleDict{Symbol,Any}(
            :name             => "DummyDeterm",
            :load_path        => "GreatPackage.MLJ.DummyDeterm",
            :is_pure_julia    => true,
            :package_uuid     => "6f286f6a-111f-5878-ab1e-185364afe411",
            :package_name     => "GreatPackage",
            :package_license  => "MIT",
            :input_scitype    => Table(Finite),
            :output_scitype   => Unknown,
            :supports_weights => true,
            :supports_class_weights => false,
            :supports_online  => false,
            :target_scitype   => AbstractVector{<:Continuous},
            :prediction_type  => :deterministic,
            :package_url      => "https://mickey.mouse.org",
            :is_supervised    => true,
            :is_wrapper       => false,
            :docstring        => "DummyDeterm from GreatPackage.jl.\n[Documentation](https://mickey.mouse.org).",
            :implemented_methods   => [:predict, ],
            :hyperparameter_types  => (),
            :hyperparameters       => (),
            :hyperparameter_ranges => ())

    @test _issubset(d, MLJModels.info_dict(DummyDeterm))
    @test _issubset(d, MLJModels.info_dict(DummyDeterm()))
end

@testset "info on interval models" begin
    d = LittleDict{Symbol,Any}(
            :name => "DummyInt",
            :load_path => "GreatPackage.MLJ.DummyInt",
            :is_pure_julia => true,
            :package_uuid  => "6f286f6a-111f-5878-ab1e-185364afe411",
            :package_name  => "GreatPackage",
            :package_license => "MIT",
            :input_scitype => Table(Finite),
            :output_scitype   => Unknown,
            :supports_weights => true,
            :supports_class_weights => false,
            :supports_online => false,
            :target_scitype => AbstractVector{<:Continuous},
            :prediction_type => :interval,
            :package_url   => "https://mickey.mouse.org",
            :is_supervised => true,
            :is_wrapper => false,
            :docstring => "DummyInt from GreatPackage.jl.\n[Documentation](https://mickey.mouse.org).",
            :implemented_methods => [:predict, ],
            :hyperparameter_types  => (),
            :hyperparameters => (),
            :hyperparameter_ranges => ())

    @test _issubset(d, MLJModels.info_dict(DummyInt))
    @test _issubset(d, MLJModels.info_dict(DummyInt()))
end

@testset "info on unsupervised models" begin
    d = LittleDict{Symbol,Any}(
            :name            => "DummyUnsup",
            :load_path       => "GreatPackage.MLJ.DummyUnsup",
            :is_pure_julia   => true,
            :package_uuid    => "6f286f6a-111f-5878-ab1e-185364afe411",
            :package_name    => "GreatPackage",
            :package_license => "MIT",
            :input_scitype   => Table(Finite),
            :target_scitype   => Unknown,
            :supports_weights => true,
            :supports_class_weights => false,
            :prediction_type  => :unknown,
            :output_scitype  => AbstractVector{<:Continuous},
            :package_url     => "https://mickey.mouse.org",
            :is_supervised   => false,
            :supports_online => false,
            :is_wrapper      => false,
            :docstring       => "DummyUnsup from GreatPackage.jl.\n[Documentation](https://mickey.mouse.org).",
            :implemented_methods   => [:transform, ],
            :hyperparameter_types  => (),
            :hyperparameters       => (),
            :hyperparameter_ranges => ())

    @test _issubset(d, MLJModels.info_dict(DummyUnsup))
    @test _issubset(d, MLJModels.info_dict(DummyUnsup()))
end

end
true
