module TestGLM

using Test

using MLJBase
using LinearAlgebra
using Statistics

import MLJModels
import Distributions
import GLM
using MLJModels.GLM_
using Random: seed!
using Tables

import DataFrames
import RDatasets

###
### OLSREGRESSOR
###

X, y = @load_boston

train, test = partition(eachindex(y), 0.7)

atom_ols = LinearRegressor()

Xtrain = selectrows(X, train)
ytrain = selectrows(y, train)
Xtest  = selectrows(X, test)

fitresult, _, report = fit(atom_ols, 1, Xtrain, ytrain)
θ = MLJBase.fitted_params(atom_ols, fitresult)

p = predict_mean(atom_ols, fitresult, Xtest)

# hand made regression to compare

Xa = MLJBase.matrix(X) # convert(Matrix{Float64}, X)
Xa1 = hcat(Xa, ones(size(Xa, 1)))
coefs = Xa1[train, :] \ y[train]
p2 = Xa1[test, :] * coefs

@test p ≈ p2

infos = info_dict(atom_ols)

@test infos[:name] == "LinearRegressor"
@test infos[:package_name] == "GLM"
@test infos[:is_pure_julia]
@test infos[:is_supervised]
@test infos[:package_license] == "MIT"
@test infos[:prediction_type] == :probabilistic
@test infos[:hyperparameters] == (:fit_intercept, :allowrankdeficient)
@test infos[:hyperparameter_types] == ("Bool", "Bool")

p_distr = predict(atom_ols, fitresult, selectrows(X, test))

@test p_distr[1] == Distributions.Normal(p[1], GLM.dispersion(fitresult))

###
### Logistic regression
###

seed!(0)

N = 100
X = MLJBase.table(rand(N, 4));
ycont = 2*X.x1 - X.x3 + 0.6*rand(N)
y = (ycont .> mean(ycont)) |> categorical;

lr = LinearBinaryClassifier()
fitresult, _, report = fit(lr, 1, X, y)

yhat = predict(lr, fitresult, X)
@test mean(cross_entropy(yhat, y)) < 0.25

pr = LinearBinaryClassifier(link=GLM.ProbitLink())
fitresult, _, report = fit(pr, 1, X, y)
yhat = predict(lr, fitresult, X)
@test mean(cross_entropy(yhat, y)) < 0.25

fitted_params(pr, fitresult)


###
### Count regression
###

seed!(1512)

X = randn(500, 5)
θ = randn(5)
y = rand.(Distributions.Poisson.(exp.(X*θ)))

XTable = MLJBase.table(X)

lcr = LinearCountRegressor(fit_intercept=false)
fitresult, _, _ = fit(lcr, 1, XTable, y)

θ̂ = fitted_params(lcr, fitresult).coef

@test norm(θ̂ .- θ)/norm(θ) ≤ 5e-3

###
### GenericGLMModel - write the formula out by hand
###

seed!(123)

formula = GLM.@formula(MedV ~ Crim + Zn + Indus + Crim*Zn + Crim*Indus)
distribution = GLM.Normal()
link = GLM.IdentityLink()

feature_names = [:Crim, :Zn, :Indus]
label_name = :MedV

model = GenericGLMModel(formula, label_name, distribution, link)

boston_data = RDatasets.dataset("MASS", "Boston") |> DataFrames.DataFrame
X = boston_data[!, feature_names]
y = boston_data[!, label_name]

fitresult, cache, report = MLJBase.fit(model, 1, X, y)

@test GLM.coeftable(fitresult).rownms == ["(Intercept)", "Crim", "Zn", "Indus", "Crim & Zn", "Crim & Indus"]

@test Statistics.mean(abs2, MLJBase.predict(model, fitresult, X) .- boston_data[!, label_name]) < 60
@test Statistics.mean(abs2, MLJBase.predict(model, fitresult, X) .- boston_data[!, label_name]) > 50

###
### GenericGLMModel - auto-generate the formula
###

# generate the RHS of a formula with all pairwise interactions
function generate_formula_rhs(feature_names::AbstractVector{<:Symbol})
    terms = []
    for i = 1:length(feature_names)
        term = GLM.Term(feature_names[i])
        push!(terms, term)
    end
    for i = 1:length(feature_names)
        for j = (i+1):length(feature_names)
            term = GLM.Term(feature_names[i]) & GLM.Term(feature_names[j])
            push!(terms, term)
        end
    end
    return sum(terms)
end

seed!(123)

boston_data = RDatasets.dataset("MASS", "Boston")
all_names = Symbol.(DataFrames.names(boston_data))
label_name = :MedV
feature_names = setdiff(all_names, [label_name])
formula = GLM.Term(Symbol(label_name)) ~ generate_formula_rhs(feature_names)

distribution = GLM.Normal()
link = GLM.IdentityLink()

model = GenericGLMModel(formula, label_name, distribution, link)

X = boston_data[!, feature_names]
y = boston_data[!, label_name]

fitresult, cache, report = MLJBase.fit(model, 1, X, y)
@test GLM.coeftable(fitresult).rownms == ["(Intercept)", "Crim", "Zn", "Indus", "Chas", "NOx", "Rm", "Age", "Dis", "Rad", "Tax", "PTRatio", "Black", "LStat", "Crim & Zn", "Crim & Indus", "Crim & Chas", "Crim & NOx", "Crim & Rm", "Crim & Age", "Crim & Dis", "Crim & Rad", "Crim & Tax", "Crim & PTRatio", "Crim & Black", "Crim & LStat", "Zn & Indus", "Zn & Chas", "Zn & NOx", "Zn & Rm", "Zn & Age", "Zn & Dis", "Zn & Rad", "Zn & Tax", "Zn & PTRatio", "Zn & Black", "Zn & LStat", "Indus & Chas", "Indus & NOx", "Indus & Rm", "Indus & Age", "Indus & Dis", "Indus & Rad", "Indus & Tax", "Indus & PTRatio", "Indus & Black", "Indus & LStat", "Chas & NOx", "Chas & Rm", "Chas & Age", "Chas & Dis", "Chas & Rad", "Chas & Tax", "Chas & PTRatio", "Chas & Black", "Chas & LStat", "NOx & Rm", "NOx & Age", "NOx & Dis", "NOx & Rad", "NOx & Tax", "NOx & PTRatio", "NOx & Black", "NOx & LStat", "Rm & Age", "Rm & Dis", "Rm & Rad", "Rm & Tax", "Rm & PTRatio", "Rm & Black", "Rm & LStat", "Age & Dis", "Age & Rad", "Age & Tax", "Age & PTRatio", "Age & Black", "Age & LStat", "Dis & Rad", "Dis & Tax", "Dis & PTRatio", "Dis & Black", "Dis & LStat", "Rad & Tax", "Rad & PTRatio", "Rad & Black", "Rad & LStat", "Tax & PTRatio", "Tax & Black", "Tax & LStat", "PTRatio & Black", "PTRatio & LStat", "Black & LStat"]

@test Statistics.mean(abs2, MLJBase.predict(model, fitresult, X) .- boston_data[!, label_name]) < 7
@test Statistics.mean(abs2, MLJBase.predict(model, fitresult, X) .- boston_data[!, label_name]) > 6

end # module
true
