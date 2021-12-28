module TestThresholdPredictors
using Test, MLJModels, CategoricalArrays
import MLJBase
using CategoricalDistributions

const MMI = MLJModels.MLJModelInterface

X = NamedTuple{(:x1,:x2,:x3)}((rand(4), rand(4), rand(4)))
yraw = ["in", "out", "out", "out"]
y = categorical(yraw, ordered = true) # `AbstractArray{OrderedFactor{2}, 1}`
y1 = categorical(yraw)                # `AbstractArray{Multiclass{2}, 1}
y2 = categorical(yraw[2:end], ordered=true)

@testset "BinaryThresholdPredictor - Probablistic" begin

    atom = ConstantClassifier()

    @test_throws MLJModels.ERR_MODEL_UNSPECIFIED BinaryThresholdPredictor()
    model = BinaryThresholdPredictor(atom)

    # Check warning when `y` is not ordered:
    @test_logs((:warn, MLJModels.warn_classes(levels(y)...)),
                MMI.fit(model, 1, X, y1))
    # Check predictions containing two classes
    @test_throws ArgumentError BinaryThresholdPredictor(ConstantRegressor())
    @test_logs((:warn, r"`threshold` should be"),
               BinaryThresholdPredictor(atom, threshold=-1))
    @test_logs((:warn, r"`threshold` should be"),
               BinaryThresholdPredictor(atom, threshold=1))

    # Compare fitresult and fitted_params with that of model, and
    # check no warning issued:
    model_fr, model_cache, model_report = @test_logs MMI.fit(model, 1, X, y)
    atom_fr, atom_cache, atom_report =
        MMI.fit(model.model, 1, X, y)
    @test model_fr[1] == atom_fr

    # Check model update
    model_up, model_cache_up, model_report_up =
        MMI.update(model, 1, model_fr, model_cache, X, y)
    atom_up, atom_cache_up, atom_report_up =
        MMI.update(model.model, 1, atom_fr, atom_cache, X, y)
    @test model_up[1] == atom_up
    @test model_cache_up[1] == atom_cache_up
    @test model_report_up[1] == atom_report_up

    # Check fitted_params
    @test MMI.fitted_params(model, model_fr).model_fitted_params ==
         MMI.fitted_params(model.model, atom_fr)

    # Check deterministic predictions
    @test MMI.predict(model, model_fr, X) ==
        MMI.predict_mode(model.model, atom_fr, X)

    model.threshold = 0.8
    model_fr, cache, report = MMI.fit(model, 1, X, y)
    @test MMI.predict(model, model_fr, X) ==
        [y[1] for i in 1:MMI.nrows(X)]

    d = MLJModels.info_dict(model)
    @test d[:supports_weights] == MMI.supports_weights(model.model)
    @test d[:input_scitype] == MMI.input_scitype(model.model)
    @test d[:target_scitype] == AbstractVector{<:MMI.Finite{2}}
    @test d[:is_pure_julia] == MMI.is_pure_julia(model.model)
    @test d[:name] == "BinaryThresholdPredictor"
    @test d[:load_path] == "MLJModels.BinaryThresholdPredictor"
end

@testset "_predict_threshold" begin
    v1 = categorical(['a', 'b', 'a'])
    v2 = categorical(['a', 'b', 'a', 'c'])
    # Test with UnivariateFinite object
    d1 = UnivariateFinite(MMI.classes(v1), [0.4, 0.6])
    @test_throws ArgumentError MLJModels._predict_threshold(d1, 0.7)
    @test MLJModels._predict_threshold(d1, (0.7, 0.3)) == v1[2]
    @test MLJModels._predict_threshold(d1, [0.5, 0.5]) == v1[2]
    @test MLJModels._predict_threshold(d1, (0.4, 0.6)) == v1[1]
    @test MLJModels._predict_threshold(d1, [0.2, 0.8]) == v1[1]
    d2 = UnivariateFinite(MMI.classes(v2), [0.4, 0.3, 0.3])
    @test_throws ArgumentError MLJModels._predict_threshold(d2, (0.7, 0.3))
    @test MLJModels._predict_threshold(d2, (0.2, 0.5, 0.3)) == v2[1]
    @test MLJModels._predict_threshold(d2, [0.3, 0.2, 0.5]) == v2[2]
    @test MLJModels._predict_threshold(d2, (0.4, 0.4, 0.2)) == v2[4]
    @test MLJModels._predict_threshold(d2, [0.2, 0.5, 0.3]) == v2[1]

    # Test with Array{UnivariateFinite, 1} object
    d1_arr = [d1 for i in 1:3]
    @test_throws ArgumentError MLJModels._predict_threshold(d1_arr, 0.7)
    @test MLJModels._predict_threshold(d1_arr, (0.7, 0.3)) == [v1[2] for i in 1:3]
    @test MLJModels._predict_threshold(d1_arr, [0.5, 0.5]) == [v1[2] for i in 1:3]
    @test MLJModels._predict_threshold(d1_arr, (0.4, 0.6)) == [v1[1] for i in 1:3]
    @test MLJModels._predict_threshold(d1_arr, [0.2, 0.8]) == [v1[1] for i in 1:3]
    d2_arr = [d2 for i in 1:3]
    @test_throws ArgumentError MLJModels._predict_threshold(d2_arr, (0.7, 0.3))
    @test MLJModels._predict_threshold(d2_arr, (0.2, 0.5, 0.3)) == [v2[1] for i in 1:3]
    @test MLJModels._predict_threshold(d2_arr, [0.3, 0.2, 0.5]) == [v2[2] for i in 1:3]
    @test MLJModels._predict_threshold(d2_arr, (0.4, 0.4, 0.2)) == [v2[4] for i in 1:3]
    @test MLJModels._predict_threshold(d2_arr, [0.2, 0.5, 0.3]) == [v2[1] for i in 1:3]

    # Test with UnivariateFiniteArray oject
    probs1 = [0.2 0.8; 0.7 0.3; 0.1 0.9]
    unf_arr1 = UnivariateFinite(MMI.classes(v1), probs1)
    @test_throws ArgumentError MLJModels._predict_threshold(unf_arr1, 0.7)
    @test MLJModels._predict_threshold(unf_arr1, (0.7, 0.3)) == [v1[2], v1[1], v1[2]]
    @test MLJModels._predict_threshold(unf_arr1, [0.5, 0.5]) == [v1[2], v1[1], v1[2]]
    @test MLJModels._predict_threshold(unf_arr1, (0.4, 0.6)) == [v1[2], v1[1], v1[2]]
    @test MLJModels._predict_threshold(unf_arr1, [0.2, 0.8]) == [v1[1], v1[1], v1[2]]
    probs2 = [0.2 0.3 0.5;0.1 0.6 0.3; 0.4 0.0 0.6]
    unf_arr2 = UnivariateFinite(MMI.classes(v2), probs2)
    @test_throws ArgumentError MLJModels._predict_threshold(unf_arr2, (0.7, 0.3))
    @test MLJModels._predict_threshold(unf_arr2, (0.2, 0.5, 0.3)) == [v2[4], v2[2], v2[1]]
    @test MLJModels._predict_threshold(unf_arr2, [0.3, 0.2, 0.5]) == [v2[2], v2[2], v2[1]]
    @test MLJModels._predict_threshold(unf_arr2, (0.4, 0.4, 0.2)) == [v2[4], v2[2], v2[4]]
    @test MLJModels._predict_threshold(unf_arr2, [0.2, 0.5, 0.3]) == [v2[4], v2[2], v2[1]]
end

# dummy detector always predicts outliers and inliers with equal
# probability:
struct DummyDetector <: MMI.ProbabilisticUnsupervisedDetector end
MMI.fit(::DummyDetector, verbosity, X) = nothing, nothing, nothing
MMI.predict(::DummyDetector, verbosity, X) =
    MLJBase.UnivariateFinite(["in", "out"],
                             fill(0.5, MLJBase.nrows(X)),
                             augment=true, pool=missing)
MMI.input_scitype(::Type{<:DummyDetector}) = MMI.Table

@testset "BinaryThresholdPredictor - ProbabilisticUnsupervisedDetector" begin
    detector = BinaryThresholdPredictor(DummyDetector(), threshold=0.2)
    @test_throws MLJModels.ERR_CLASSES_DETECTOR MMI.fit(detector, 1, X, y1)

    fr, _, _ = MMI.fit(detector, 0, X, y)
    @test MMI.predict(detector, fr, X) == fill("out", length(y))
    fr, _, _ = MMI.fit(detector, 0, X)
    @test MMI.predict(detector, fr, X) == fill("out", length(y))


    detector.threshold = 0.8
    fr, _, _ = MMI.fit(detector, 0, X, y)
    @test MMI.predict(detector, fr, X) == fill("in", length(y))
    fr, _, _ = MMI.fit(detector, 0, X)
    @test MMI.predict(detector, fr, X) == fill("in", length(y))

    # integration (y == ["in", "out", "out", "out"]):
    e = MLJBase.evaluate(detector, X, y,
                         resampling=MLJBase.Holdout(fraction_train=0.5),
                         measure=MLJBase.accuracy)
    @test e.measurement[1] ≈ 0
end

end
true
