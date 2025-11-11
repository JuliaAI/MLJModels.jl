module TestConstant

using Test, MLJModels, CategoricalArrays
import Distributions, MLJBase, Tables


# Any X will do for constant models:
X = NamedTuple{(:x1,:x2,:x3)}((rand(10), rand(10), rand(10)))

@testset "ConstantRegressor" begin
    y = [1.0, 1.0, 2.0, 2.0]

    model = ConstantRegressor(distribution_type=
                              Distributions.Normal{Float64})
    fitresult, cache, report = MLJBase.fit(model, 1, X, y)

    d = Distributions.Normal(1.5, 0.5)
    @test fitresult.μ ≈ d.μ
    @test fitresult.σ ≈ d.σ
    @test MLJBase.predict(model, fitresult, X)[7].μ ≈ d.μ
    @test MLJBase.predict_mean(model, fitresult, X) ≈ fill(1.5, 10)

    @test MLJBase.input_scitype(model) == MLJBase.Table
    @test MLJBase.target_scitype(model) == AbstractVector{MLJBase.Continuous}
    @test MLJBase.name(model) == "ConstantRegressor"
    @test MLJBase.load_path(model) == "MLJModels.ConstantRegressor"
end

@testset "DeterministicConstantRegressor" begin

    X = (; x=ones(3))
    S = MLJBase.target_scitype(DeterministicConstantRegressor())

    # vector target:
    y = Float64[2, 3, 4]
    @test MLJBase.scitype(y) <: S
    mach = MLJBase.machine(MLJModels.DeterministicConstantRegressor(), X, y)
    MLJBase.fit!(mach, verbosity=0)
    @test MLJBase.predict(mach, X) ≈ [3, 3, 3]
    @test only(MLJBase.fitted_params(mach).mean) ≈ 3

    # matrix target:
    y = Float64[2 5; 3 6; 4 7]
    @test MLJBase.scitype(y) <: S
    mach = MLJBase.machine(MLJModels.DeterministicConstantRegressor(), X, y)
    MLJBase.fit!(mach, verbosity=0)
    @test MLJBase.predict(mach, X) ≈ [3 6; 3 6; 3 6]
    @test MLJBase.fitted_params(mach).mean ≈ [3 6]

    # tabular target:
    y = Float64[2 5; 3 6; 4 7] |> Tables.table |> Tables.rowtable
    @test MLJBase.scitype(y) <: S
    mach = MLJBase.machine(MLJModels.DeterministicConstantRegressor(), X, y)
    MLJBase.fit!(mach, verbosity=0)
    yhat = MLJBase.predict(mach, X)
    @test yhat isa Vector{<:NamedTuple}
    @test Tables.matrix(yhat)  ≈ [3 6; 3 6; 3 6]
    @test MLJBase.fitted_params(mach).mean ≈ [3 6]
end

@testset "ConstantClassifier" begin
    yraw = ["Perry", "Antonia", "Perry", "Skater"]
    y = categorical(yraw)

    model = ConstantClassifier()
    fitresult, cache, report =  MLJBase.fit(model, 1, X, y)

    d = MLJBase.UnivariateFinite([y[1], y[2], y[4]], [0.5, 0.25, 0.25])

    yhat = MLJBase.predict_mode(model, fitresult, X)
    @test levels(yhat[1]) == levels(y[1])
    @test yhat[5] == y[1]
    @test length(yhat) == 10

    yhat = MLJBase.predict(model, fitresult, X)
    yhat1 = yhat[1]

    for c in levels(d)
        Distributions.pdf(yhat1, c) ≈ Distributions.pdf(d, c)
    end

    # with weights:
    w = [2, 3, 2, 5]
    model = ConstantClassifier()
    fitresult, cache, report =  MLJBase.fit(model, 1, X, y, w)
    d = MLJBase.UnivariateFinite([y[1], y[2], y[4]], [1/3, 1/4, 5/12])

    @test MLJBase.input_scitype(model) == MLJBase.Table
    @test MLJBase.target_scitype(model) == AbstractVector{<:MLJBase.Finite}
    @test MLJBase.name(model) == "ConstantClassifier"
    @test MLJBase.load_path(model) == "MLJModels.ConstantClassifier"

end

end # module
true
