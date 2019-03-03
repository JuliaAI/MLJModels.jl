module TestNaiveBayes

# using Revise
using MLJBase
using Test

import MLJModels
import NaiveBayes

using MLJModels.NaiveBayes_
using CategoricalArrays

function get_output(val)
    max = 0
    res = 0
    for ele in val.prob_given_level
        if ele[2] > max
            max = ele[2]
            res = ele[1]
        end
    end
    return res
end

gaussian_classifier = GaussianNBClassifier()

# gaussian classifier takes continuous features
task = load_iris()
X, y = X_and_y(task)
train, test = partition(eachindex(y), 0.6)

fitresultG, cacheG, reportG = fit(gaussian_classifier, 1,
                    selectrows(X, train), y[train]);

gaussian_pred = predict(gaussian_classifier, fitresultG, selectrows(X, test));

@test levels(keys(gaussian_pred[1].prob_given_level)) == levels(y[train])


# test with linear data:
x1 = randn(3000);
x2 = randn(3000);
x3 = randn(3000);
X = (x1=x1, x2=x2, x3=x3);
y = x1 - x2 -2x3;
ycat = map(y) do η
    η > 0 ? "go" : "stop"
end |> categorical;
train, test = partition(eachindex(ycat), 0.8);

gaussian_classifier = GaussianNBClassifier()

fitresultG, cacheG, reportG = MLJBase.fit(gaussian_classifier, 1,
             selectrows(X, train), ycat[train])

gaussian_pred = MLJBase.predict(gaussian_classifier, fitresultG, selectrows(X, test))

@test sum(get_output.(gaussian_pred) .!= ycat[test])/length(ycat) < 0.05


multinomial_classifier = MultinomialNBClassifier()

generate(n, m) = map(rand(Int, n)) do x mod(x,m) end |> categorical
x1 = generate(1000, 2);
x2 = generate(1000, 3);
x3 = generate(1000, 5);
function f(x)
    if x == 0
        return :zero
    elseif x == 1
        return :one
    else
        return :other
    end
end

X = (x1=x1, x2=x2, x3=x3) # X is tabular
y = map(f, x3)
train, test = partition(eachindex(y), 0.8);

fitresultMLT, cacheMLT, reportMLT = MLJBase.fit(multinomial_classifier, 1, selectrows(X, train), y[train])

MLTpred = MLJBase.predict(multinomial_classifier, fitresultMLT, selectrows(X, test))

