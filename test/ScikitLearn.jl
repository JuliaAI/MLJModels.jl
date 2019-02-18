module TestScikitLearn

using MLJBase
using Test

import MLJModels
import ScikitLearn
using MLJModels.ScikitLearn_
using CategoricalArrays

task = load_iris();
X, y = X_and_y(task)
train, test = partition(eachindex(y), 0.6) # levels of y are split across split

plain_classifier = SVMClassifier()
nu_classifier = SVMNuClassifier()
linear_classifier = SVMLClassifier(max_iter=10000)

plain_regressor = SVMRegressor()
nu_regressor = SVMNuRegressor()
linear_regressor = SVMLRegressor(max_iter=10000)

# Xr = [[0, 0], [1, 1]]
# yr = [0, 1]
# targetr = [[1, 1]]

task = load_boston()
Xr, yr = X_and_y(task)

fitresultC, cacheC, reportC = MLJBase.fit(plain_classifier, 1,
                                          selectrows(X, train), y[train]);
fitresultCnu, cacheCnu, reportCnu = MLJBase.fit(nu_classifier, 1, 
                                          selectrows(X, train), y[train]);
fitresultCL, cacheCL, reportCL = MLJBase.fit(linear_classifier, 1,
                                          selectrows(X, train), y[train]);

fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 1, Xr, yr);
fitresultRnu, cacheRnu, reportRnu = MLJBase.fit(nu_regressor, 1, Xr, yr);
fitresultRL, cacheRL, reportRL = MLJBase.fit(linear_regressor, 1, Xr, yr);

pcpred = predict(plain_classifier, fitresultC, selectrows(X, test));
nucpred = predict(nu_classifier, fitresultCnu, selectrows(X, test));
lcpred = predict(linear_classifier, fitresultCL, selectrows(X, test));

@test levels(pcpred) == levels(y[train])
@test levels(nucpred) == levels(y[train])
@test levels(lcpred) == levels(y[train])

rpred = predict(plain_regressor, fitresultR, targetr);
nurpred = predict(nu_regressor, fitresultRnu, targetr);
Lrpred = predict(linear_regressor, fitresultRL, targetr);


end
true
