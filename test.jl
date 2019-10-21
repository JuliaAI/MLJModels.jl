using MLJ
using MLJBase
using MLBase
using RDatasets
data = dataset("MASS", "Melanoma")
@load LinearBinaryClassifier
X = data[:, [:Status, :Sex, :Age, :Year, :Thickness]]
y_plain = data.Ulcer
y = categorical(y_plain)

n = length(y)

baseline_y = convert.(Int, rand(n) .> 0.5)
baseline_mse = sum((baseline_y - y_plain).^2)/n


lr = LinearBinaryClassifier()
fitresult, _, report = fit(lr, 1, X, y)

predict_mode(lr,fitresult,X)
y


confusmat(2,predict_mode(lr,fitresult,X),y)
MLBase.
predict_mode(lr,fitresult,X)
y
