module TestNaiveBayes

using Pkg
using MLJBase
using Test
import Random.seed!

import MLJModels
import NaiveBayes

using MLJModels.NaiveBayes_
using CategoricalArrays

## GAUSSIAN

gaussian_classifier = GaussianNBClassifier()
info_dict(gaussian_classifier)

# gaussian classifier takes continuous features
using RDatasets
iris = dataset("datasets", "iris")
X = iris[:, 1:4]
y = iris[:, 5]

train, test = partition(eachindex(y), 0.6)

fitresultG, cacheG, reportG = fit(gaussian_classifier, 1,
                    selectrows(X, train), y[train]);

gaussian_pred = predict(gaussian_classifier, fitresultG, selectrows(X, test));

yhat1 = gaussian_pred[1]
@test Set(classes(yhat1)) == Set(classes(y[1]))

# test with linear data:
seed!(1234)
x1 = randn(3000);
x2 = randn(3000);
x3 = randn(3000);
X = (x1=x1, x2=x2, x3=x3);
ycont = x1 - x2 -2x3;
y = map(ycont) do η
    η > 0 ? "go" : "stop"
end |> categorical;
train, test = partition(eachindex(y), 0.8);

gaussian_classifier = GaussianNBClassifier()

fitresultG, cacheG, reportG = MLJBase.fit(gaussian_classifier, 1,
             selectrows(X, train), y[train])

gaussian_pred = MLJBase.predict_mode(gaussian_classifier,
                                     fitresultG, selectrows(X, test))

@test sum(gaussian_pred .!= y[test])/length(y) < 0.05


## MULTINOMIAL

# first contrive some test data

# some word counts in children's books about colours:
red = [2, 0, 1, 0, 1]
blue = [4, 1, 2, 3, 2]
green = [0, 2, 0, 6, 1]
X = (red=red, blue=blue, green=green)

# gender of author:
y = categorical([:m, :f, :m, :f, :m])
male = y[1]
female = y[2]

# Note: The smoothing algorithm is to add to the training data, for
# each class observed, a row with every feature getting count of
# alpha. So smoothing also effects the class marginals (is this
# standard)? Only integer values of alpha allowed.

# computing conditional probabilities by hand with Lagrangian
# smoothing (alpha=1):
red_given_m = 5/16
blue_given_m = 9/16
green_given_m = 2/16
red_given_f = 1/15
blue_given_f = 5/15
green_given_f = 9/15

m_(red, blue, green) =
    4/7*(red_given_m^red)*(blue_given_m^blue)*(green_given_m^green)
f_(red, blue, green) =
    3/7*(red_given_f^red)*(blue_given_f^blue)*(green_given_f^green)
normalizer(red, blue, green) = m_(red, blue, green) + f_(red, blue, green)
m(a...) = m_(a...)/normalizer(a...)
f(a...) = f_(a...)/normalizer(a...)

Xnew = (red=[1, 1], blue=[1, 2], green=[1, 3])

# prediction by hand:

yhand =[MLJBase.UnivariateFinite([male, female], [m(1, 1, 1), f(1, 1, 1)]),
        MLJBase.UnivariateFinite([male, female], [m(1, 2, 3), f(1, 2, 3)])]

multinomial_classifier = MultinomialNBClassifier()
info_dict(multinomial_classifier)

fitresultMLT, cacheMLT, reportMLT =
    MLJBase.fit(multinomial_classifier, 1, X, y)

yhat = MLJBase.predict(multinomial_classifier, fitresultMLT, Xnew)

# see issue https://github.com/dfdx/NaiveBayes.jl/issues/42
@test_broken pdf(yhand[1], :m) ≈ pdf(yhat[1], :m)
@test_broken pdf(yhand[1], :f) ≈ pdf(yhat[1], :f)

end # module
true
