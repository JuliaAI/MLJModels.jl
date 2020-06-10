## THE CONSTANT REGRESSOR

"""
    ConstantRegressor(; distribution_type=Distributions.Normal)

A regressor that, for any new input pattern, predicts the univariate
probability distribution best fitting the training target data. Use
`predict_mean` to predict the mean value instead.
"""
struct ConstantRegressor{D<:Distributions.Sampleable} <: Probabilistic
    distribution_type::Type{D}
end

ConstantRegressor(; distribution_type=Distributions.Normal) =
    ConstantRegressor(distribution_type)

function MLJModelInterface.fit(::ConstantRegressor{D},
                               verbosity::Int,
                               X,
                               y) where D
    fitresult = Distributions.fit(D, y)
    cache     = nothing
    report    = NamedTuple()
    return fitresult, cache, report
end

MLJModelInterface.fitted_params(::ConstantRegressor, fitresult) =
    (target_distribution=fitresult,)

MLJModelInterface.predict(::ConstantRegressor, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))

##
## THE CONSTANT DETERMINISTIC REGRESSOR (FOR TESTING)
##

struct DeterministicConstantRegressor <: Deterministic end

function MLJModelInterface.fit(::DeterministicConstantRegressor,
                               verbosity::Int,
                               X,
                               y)
    fitresult = mean(y)
    cache     = nothing
    report    = NamedTuple()
    return fitresult, cache, report
end

MLJModelInterface.predict(::DeterministicConstantRegressor, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))

##
## THE CONSTANT CLASSIFIER
##

"""
    ConstantClassifier()

A classifier that, for any new input pattern, `predict`s the
`UnivariateFinite` probability distribution `d` best fitting the
training target data. So, `pdf(d, level)` is the proportion of levels
in the training data coinciding with `level`. Use `predict_mode` to
obtain the training target mode instead.
"""
struct ConstantClassifier <: Probabilistic end

# here `args` is `y` or `y, w`:
function MLJModelInterface.fit(::ConstantClassifier,
                               verbosity::Int,
                               X,
                               y,
                               w=nothing)
    # We need MLJBase and not MMI here, because we need the *type* not
    # a method:
    d = Distributions.fit(MLJBase.UnivariateFinite, y, w)
    C = classes(d)
    fitresult = (C, pdf([d, ], C))
    cache     = nothing
    report    = NamedTuple
    return fitresult, cache, report
end

MLJModelInterface.fitted_params(::ConstantClassifier, fitresult) =
    (target_distribution=fitresult,)

function MLJModelInterface.predict(::ConstantClassifier, fitresult, Xnew)
    _classes, probs1 = fitresult
    N = nrows(Xnew)
    probs = reshape(vcat(fill(probs1, N)...), N, length(_classes))
    return UnivariateFinite(_classes, probs)
end


##
## DETERMINISTIC CONSTANT CLASSIFIER (FOR TESTING)
##

struct DeterministicConstantClassifier <: Deterministic end

function MLJModelInterface.fit(::DeterministicConstantClassifier, verbosity::Int, X, y)
    # dump missing target values and make into a regular array:
    fitresult = mode(skipmissing(y) |> collect) # a CategoricalValue
    cache     = nothing
    report    = NamedTuple()
    return fitresult, cache, report
end

MLJModelInterface.predict(::DeterministicConstantClassifier, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))

##
## METADATA
##

metadata_pkg.(
    (ConstantRegressor, ConstantClassifier,
     DeterministicConstantRegressor, DeterministicConstantClassifier),
    name       = "MLJModels",
    uuid       = "d491faf4-2d78-11e9-2867-c94bc002c0b7",
    url        = "https://github.com/alan-turing-institute/MLJModels.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false)

metadata_model(ConstantRegressor,
    input   = Table,
    target  = AbstractVector{Continuous},
    weights = false,
    descr   = "Constant regressor (Probabilistic).",
    path    = "MLJModels.ConstantRegressor")

metadata_model(DeterministicConstantRegressor,
    input   = Table,
    target  = AbstractVector{Continuous},
    weights = false,
    descr   = "Constant regressor (Deterministic).",
    path    = "MLJModels.DeterministicConstantRegressor")

metadata_model(ConstantClassifier,
    input   = Table,
    target  = AbstractVector{<:Finite},
    weights = true,
    descr   = "Constant classifier (Probabilistic).",
    path    = "MLJModels.ConstantClassifier")

metadata_model(DeterministicConstantClassifier,
    input   = Table,
    target  = AbstractVector{<:Finite},
    weights = false,
    descr   = "Constant classifier (Deterministic).",
    path    = "MLJModels.DeterministicConstantClassifier")
