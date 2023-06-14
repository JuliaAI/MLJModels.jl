## THE CONSTANT REGRESSOR

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

struct ConstantClassifier <: Probabilistic end

# here `args` is `y` or `y, w`:
function MLJModelInterface.fit(::ConstantClassifier,
                               verbosity::Int,
                               X,
                               y,
                               w=nothing)
    d = Distributions.fit(UnivariateFinite, y, w)
    C = classes(d)
    fitresult = (C, Distributions.pdf([d, ], C))
    cache     = nothing
    report    = NamedTuple()
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

metadata_model(
    ConstantRegressor,
    input_scitype = Table,
    target_scitype = AbstractVector{Continuous},
    supports_weights = false,
    load_path = "MLJModels.ConstantRegressor"
)

metadata_model(
    DeterministicConstantRegressor,
    input_scitype = Table,
    target_scitype = AbstractVector{Continuous},
    supports_weights = false,
    load_path    = "MLJModels.DeterministicConstantRegressor"
)

metadata_model(
    ConstantClassifier,
    input_scitype = Table,
    target_scitype = AbstractVector{<:Finite},
    supports_weights = true,
    load_path = "MLJModels.ConstantClassifier"
)

metadata_model(
    DeterministicConstantClassifier,
    input_scitype = Table,
    target_scitpye = AbstractVector{<:Finite},
    supports_weights = false,
    load_path = "MLJModels.DeterministicConstantClassifier"
)


# # DOCUMENT STRINGS

"""
    ConstantRegressor

This "dummy" probabilistic predictor always returns the same distribution, irrespective of
the provided input pattern. The distribution returned is the one of the type specified that
best fits the training target data. Use `predict_mean` or `predict_median` to predict the
mean or median values instead. If not specified, a normal distribution is fit.

Almost any reasonable model is expected to outperform `ConstantRegressor` which is used
almost exclusively for testing and establishing performance baselines.

In MLJ (or MLJModels) do `model = ConstantRegressor()` or `model =
ConstantRegressor(distribution=...)` to construct a model instance.

# Training data

In MLJ (or MLJBase) bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`)

- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `schema(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `distribution_type=Distributions.Normal`: The distribution to be fit to the target
  data. Must be a subtype of `Distributions.ContinuousUnivariateDistribution `.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given
  features `Xnew` (which for this model are ignored). Predictions are probabilistic.

- `predict_mean(mach, Xnew)`: Return instead the means of the probabilistic predictions
  returned above.

- `predict_median(mach, Xnew)`: Return instead the medians of the probabilistic
  predictions returned above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `target_distribution`: The distribution fit to the supplied target data.

# Examples

```julia
using MLJ

X, y = make_regression(10, 2) # synthetic data: a table and vector
regressor = ConstantRegressor()
mach = machine(regressor, X, y) |> fit!

fitted_params(mach)

Xnew, _ = make_regression(3, 2)
predict(mach, Xnew)
predict_mean(mach, Xnew)

```
See also
[`ConstantClassifier`](@ref)
"""
ConstantRegressor

"""
    ConstantClassifier

This "dummy" probabilistic predictor always returns the same distribution, irrespective of
the provided input pattern. The distribution `d` returned is the `UnivariateFinite`
distribution based on frequency of classes observed in the training target data. So,
`pdf(d, level)` is the number of times the training target takes on the value `level`.
Use `predict_mode` instead of `predict` to obtain the training target mode instead. For
more on the `UnivariateFinite` type, see the CategoricalDistributions.jl package.

Almost any reasonable model is expected to outperform `ConstantClassifier`, which is used
almost exclusively for testing and establishing performance baselines.

In MLJ (or MLJModels) do `model = ConstantClassifier()` to construct an instance.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`)

- `y` is the target, which can be any `AbstractVector` whose element scitype is `Finite`;
  check the scitype with `schema(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

None.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given features `Xnew` (which for
  this model are ignored). Predictions are probabilistic.

- `predict_mode(mach, Xnew)`: Return the mode of the probabilistic predictions
  returned above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `target_distribution`: The distribution fit to the supplied target data.

# Examples

```julia
using MLJ

clf = ConstantClassifier()

X, y = @load_crabs # a table and a categorical vector
mach = machine(clf, X, y) |> fit!

fitted_params(mach)

Xnew = (;FL = [8.1, 24.8, 7.2],
        RW = [5.1, 25.7, 6.4],
        CL = [15.9, 46.7, 14.3],
        CW = [18.7, 59.7, 12.2],
        BD = [6.2, 23.6, 8.4],)

# probabilistic predictions:
yhat = predict(mach, Xnew)
yhat[1]

# raw probabilities:
pdf.(yhat, "B")

# probability matrix:
L = levels(y)
pdf(yhat, L)

# point predictions:
predict_mode(mach, Xnew)
```

See also [`ConstantRegressor`](@ref)

"""
ConstantClassifier
