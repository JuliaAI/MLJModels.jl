module TestThresholdPredictors
using Test, MLJModels, CategoricalArrays
using ScientificTypes
using CategoricalDistributions

import Distributions
import MLJBase

const MMI = MLJModels.MLJModelInterface

X_ = NamedTuple{(:x1,:x2,:x3)}((rand(4), rand(4), rand(4)))
yraw = ["in", "out", "out", "out"]
y_ = categorical(yraw, ordered = true) # `AbstractArray{OrderedFactor{2}, 1}`
y1_ = categorical(yraw)                # `AbstractArray{Multiclass{2}, 1}
y2_ = categorical(yraw[2:end], ordered=true)

@testset "BinaryThresholdPredictor - Probablistic" begin

    atom = ConstantClassifier()

    @test_throws MLJModels.ERR_MODEL_UNSPECIFIED BinaryThresholdPredictor()
    model = BinaryThresholdPredictor(atom)

    @test_throws(
        MLJModels.err_unsupported_model_type(MLJBase.Unsupervised),
        BinaryThresholdPredictor(Standardizer()),
    )

    # Check warning when `y` is not ordered:
    @test_logs((:warn, MLJModels.warn_classes(levels(y_)...)),
                MMI.fit(model, 1, MMI.reformat(model, X_, y1_)...))
    # Check predictions containing two classes
    @test_throws ArgumentError BinaryThresholdPredictor(ConstantRegressor())
    @test_logs((:warn, r"`threshold` should be"),
               BinaryThresholdPredictor(atom, threshold=-1))
    @test_logs((:warn, r"`threshold` should be"),
               BinaryThresholdPredictor(atom, threshold=1))

    # data reformat
    X, y = MMI.reformat(model, X_, y_)

    # Compare fitresult and fitted_params with that of model, and
    # check no warning issued:
    model_fr, model_cache, model_report = @test_logs MMI.fit(
        model, 1, X, y
    )
    atom_fr, atom_cache, atom_report = MMI.fit(
        model.model, 1, X_, y_
    )
    @test model_fr[1] == atom_fr

    # Check model update
    model_up, model_cache_up, model_report_up = MMI.update(
        model, 1, model_fr, model_cache, X, y
    )
    atom_up, atom_cache_up, atom_report_up = MMI.update(
        model.model, 1, atom_fr, atom_cache, X_, y_
    )
    @test model_up[1] == atom_up
    @test model_cache_up[1] == atom_cache_up
    @test model_report_up[1] == atom_report_up

    # Check fitted_params
    @test MMI.fitted_params(model, model_fr).model_fitted_params ==
         MMI.fitted_params(model.model, atom_fr)

    # Check deterministic predictions
    @test MMI.predict(model, model_fr, X) == MMI.predict_mode(
        model.model, atom_fr, X_
    )

    model.threshold = 0.8
    model_fr, cache, report = MMI.fit(
        model, 1, X, y
    )
    @test MMI.predict(model, model_fr, X) ==
        [y_[1] for i in 1:MMI.nrows(X_)]

    @test MMI.supports_weights(model) == MMI.supports_weights(model.model)
    @test MMI.input_scitype(model) == MMI.input_scitype(model.model)
    @test MMI.target_scitype(model) == AbstractVector{<:MMI.Finite{2}}
    @test MMI.is_pure_julia(model) == MMI.is_pure_julia(model.model)
    @test MMI.name(model) == "BinaryThresholdPredictor"
    @test MMI.load_path(model) == "MLJModels.BinaryThresholdPredictor"
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
    @test_throws MLJModels.ERR_CLASSES_DETECTOR MMI.fit(
        detector, 1, MMI.reformat(detector, X_, y1_)...
    )

    X, y = MMI.reformat(detector, X_, y_)
    fr, _, _ = MMI.fit(detector, 0, X, y)
    @test MMI.predict(detector, fr, X) == fill("out", length(y_))
    fr, _, _ = MMI.fit(detector, 0, X)
    @test MMI.predict(detector, fr, X) == fill("out", length(y_))


    detector.threshold = 0.8
    fr, _, _ = MMI.fit(detector, 0, X, y)
    @test MMI.predict(detector, fr, X) == fill("in", length(y_))
    fr, _, _ = MMI.fit(detector, 0, X)
    @test MMI.predict(detector, fr, X) == fill("in", length(y_))

    # integration (y == ["in", "out", "out", "out"]):
    accuracy(yhat, y) = sum(yhat .== y)/length(y)
    e = MLJBase.evaluate(detector, X_, y_,
                         resampling=MLJBase.Holdout(fraction_train=0.5),
                         measure=accuracy)
    @test e.measurement[1] ≈ 0
end

@testset "_make_binary" begin
    @test MLJModels._make_binary(AbstractVector{<:Multiclass}) ==
        AbstractVector{<:Multiclass{2}}
    @test MLJModels._make_binary(AbstractVector{<:Union{Missing,Multiclass}}) ==
        AbstractVector{<:Union{Missing,Multiclass{2}}}
    @test MLJModels._make_binary(AbstractVector{<:OrderedFactor}) ==
        AbstractVector{<:OrderedFactor{2}}
    @test MLJModels._make_binary(AbstractVector{<:Union{Missing,OrderedFactor}}) ==
        AbstractVector{<:Union{Missing,OrderedFactor{2}}}
    @test MLJModels._make_binary(AbstractVector{<:Finite}) ==
        AbstractVector{<:Finite{2}}
    @test MLJModels._make_binary(AbstractVector{<:Union{Missing,Finite}}) ==
        AbstractVector{<:Union{Missing,Finite{2}}}
end

struct DummyIterativeClassifier <: MMI.Probabilistic end

MMI.fit(::DummyIterativeClassifier, verbosity, data...) =
    42, nothing, (; losses = [1.0, 2.0])
MMI.training_losses(::DummyIterativeClassifier, report) = report.losses

MMI.iteration_parameter(::Type{<:DummyIterativeClassifier}) = :n
MMI.supports_training_losses(::Type{<:DummyIterativeClassifier}) = true
MMI.target_scitype(::Type{<:DummyIterativeClassifier}) =
    AbstractVector{Multiclass{2}}

@testset "training losses support" begin
    X = ones(3, 2)
    y = ScientificTypes.coerce(["Y", "Y", "N"], OrderedFactor)

    thresholder = BinaryThresholdPredictor(DummyIterativeClassifier())

    __, __, re = MMI.fit(thresholder, 0, MMI.reformat(thresholder, X, y)...)

    @test MMI.supports_training_losses(thresholder)
    @test MMI.training_losses(thresholder, re) == [1.0, 2.0]
end

## Data Front-end Tests
struct NaiveClassifier <: MMI.Probabilistic end

function MMI.fit(::NaiveClassifier, verbosity, reformatted_X, reformatted_target)
    fitresult = Distributions.fit(MLJBase.UnivariateFinite, reformatted_target[1])
    return fitresult, nothing, NamedTuple()
end

function MMI.predict(::NaiveClassifier, fitresult, reformatted_Xnew)
    return fill(fitresult, size(reformatted_Xnew, 1))
end

MMI.reformat(::NaiveClassifier, X, target) = (MMI.matrix(X), (target,))
MMI.reformat(::NaiveClassifier, X) = (MMI.matrix(X),)

function MMI.selectrows(::NaiveClassifier, I, reformatted_X, reformatted_target)
        return (reformatted_X[I, :], (reformatted_target[1][I],))
end

MMI.selectrows(::NaiveClassifier, I, reformatted_X) = (reformatted_X[I, :],)
MMI.target_scitype(::Type{<:NaiveClassifier}) = AbstractVector{OrderedFactor{2}}
MMI.input_scitype(::Type{<:NaiveClassifier}) = Table(Continuous)

@testset "ThresholdUnion Data Front-End" begin
    X = MMI.table(rand(10, 10))
    y = categorical([2, 1, 1, 2, 1, 1, 2, 1, 1, 2], ordered=true)

    naive_classifier = NaiveClassifier()
    threshold_classifier = BinaryThresholdPredictor(naive_classifier)
    reformatted_args = MMI.reformat(threshold_classifier, X)
    @test reformatted_args == (MMI.matrix(X),)
    reformatted_args_ = MMI.reformat(threshold_classifier, X, y)
    @test reformatted_args_ == (
        MMI.matrix(X),
        MLJModels.ReformattedTarget(
            (y,), levels(y), scitype(y)
        )
    )
    I = 2:5
    @test MMI.selectrows(
        threshold_classifier,
        I,
        reformatted_args...
    ) == (MMI.matrix(X)[I, :],)

    r = MMI.selectrows(threshold_classifier, I, reformatted_args_...)
    @test r[1] == MMI.matrix(X)[I, :]
    s = MLJModels.ReformattedTarget(
        (y[I],), levels(y[I]), scitype(y[I])
    )
    @test r[2].y == s.y
    @test r[2].levels == s.levels
    @test r[2].scitype == s.scitype

    # machine end-end test
    mach = MLJBase.machine(threshold_classifier, X, y)
    MLJBase.fit!(mach, rows=I)
    @test MLJBase.predict(mach, X) == fill(
        mode(Distributions.fit(MLJBase.UnivariateFinite, y[I])), MLJBase.nrows(X)
    )
end


# define a probabilistic classifier with non-persistent `fitresult`, but which addresses
# this by overloading `save`/`restore`:
thing = []
struct EphemeralClassifier <: MLJBase.Probabilistic end
function MLJBase.fit(::EphemeralClassifier, verbosity, X, y)
    # if I serialize/deserialized `thing` then `id` below changes:
    id = objectid(thing)
    p = Distributions.fit(UnivariateFinite, y)
    fitresult = (thing, id, p)
    return fitresult, nothing, NamedTuple()
end
function MLJBase.predict(::EphemeralClassifier, fitresult, X)
    thing, id, p = fitresult
    return id == objectid(thing) ? fill(p, MLJBase.nrows(X)) :
        throw(ErrorException("dead fitresult"))
end
MLJBase.target_scitype(::Type{<:EphemeralClassifier}) = AbstractVector{OrderedFactor{2}}
function MLJBase.save(::EphemeralClassifier, fitresult)
    thing, _, p = fitresult
    return (thing, p)
end
function MLJBase.restore(::EphemeralClassifier, serialized_fitresult)
    thing, p = serialized_fitresult
    id = objectid(thing)
    return (thing, id, p)
end

# X, y = (; x = rand(8)), categorical(collect("OXXXXOOX"), ordered=true)
# mach = machine(EphemeralClassifier(), X, y) |> fit!
# io = IOBuffer()
# MLJBase.save(io, mach)
# seekstart(io)
# mach2 = machine(io)
# predict(mach2, X)

@testset "serialization for atomic models with non-persistent fitresults" begin
    # https://github.com/alan-turing-institute/MLJ.jl/issues/1099
    X, y = (; x = rand(8)), categorical(collect("OXXXXOOX"), ordered=true)
    deterministic_classifier = BinaryThresholdPredictor(
        EphemeralClassifier(),
        threshold=0.5,
    )
    mach = MLJBase.machine(deterministic_classifier, X, y)
    MLJBase.fit!(mach, verbosity=0)
    yhat = MLJBase.predict(mach, MLJBase.selectrows(X, 1:2))
    io = IOBuffer()
    MLJBase.save(io, mach)
    seekstart(io)
    mach2 = MLJBase.machine(io)
    close(io)
    @test MLJBase.predict(mach2, (; x = rand(2))) ==  yhat
end

end # module

true
