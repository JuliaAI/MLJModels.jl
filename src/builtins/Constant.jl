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


# # DOCUMENT STRINGS

"""
$(MMI.doc_header(ConstantRegressor))
`ConstantRegressor`: A regressor that, for any new input pattern, predicts the
univariate probability distribution best fitting the training target data. Use
`predict_mean` to predict the mean value instead.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

Where
- `X`: is any table of input features (eg, a `DataFrame`); check the scitype
  with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `schema(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `distribution_type=Distributions.Normal`: The distribution from which to sample. Must be a subtype of `Distributions.Sampleable`.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic but uncalibrated.
- `predict_mean(mach, Xnew)`: Return the means of the probabilistic predictions
  returned above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `target_distribution`: The distribution of the supplied target data.

# Examples

```julia
using MLJ

ConstantRegressor = @load ConstantRegressor pkg=MLJModels

X, y = make_regression(100, 60) # synthetic data
regressor = ConstantRegressor()
mach = machine(regressor, X, y) |> fit!

fitted_params(mach)

Xnew, _ = make_regression(3, 60)
predict(mach, Xnew)
predict_mean(mach, Xnew)

```
See also
[`ConstantClassifier`](@ref)
"""
ConstantRegressor

"""
$(MMI.doc_header(ConstantClassifier))
`ConstantClassifier`: A classifier that, for any new input pattern, `predict`s
the `UnivariateFinite` probability distribution `d` best fitting the training
target data. So, `pdf(d, level)` is the proportion of levels in the training
data coinciding with `level`. Use `predict_mode` to obtain the training target
mode instead.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

Where

- `X`: is any table of input features (eg, a `DataFrame`); check the scitype
  with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Finite`; check the scitype with `schema(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `distribution_type=Distributions.Normal`: The distribution from which to sample. Must be a subtype of `Distributions.Sampleable`.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic but uncalibrated.
- `predict_mode(mach, Xnew)`: Return the means of the probabilistic predictions
  returned above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `target_distribution`: The distribution of the supplied target data.

# Examples

```julia
using MLJ

ConstantClassifier = @load ConstantClassifier pkg=MLJModels
clf = ConstantClassifier()

X, y = @load_crabs
mach = machine(clf, X, y) |> fit!

fitted_params(mach)

Xnew = (;FL = [8.1, 24.8, 7.2],
        RW = [5.1, 25.7, 6.4],
        CL = [15.9, 46.7, 14.3],
        CW = [18.7, 59.7, 12.2],
        BD = [6.2, 23.6, 8.4],)

predict(mach, Xnew)
predict_mode(mach, Xnew)
```
See also [`ConstantRegressor`](@ref)
"""
ConstantClassifier
