module TestConstant

using Test, MLJModels, CategoricalArrays
import Distributions, MLJBase

# Any X will do for constant models:
X = NamedTuple{(:x1,:x2,:x3)}((rand(10), rand(10), rand(10)))

@testset "Regressor" begin
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

@testset "Classifier" begin
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
