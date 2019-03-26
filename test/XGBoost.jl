module TestXGBoosts

using MLJBase
using Test
import MLJModels
import XGBoost
import CategoricalArrays
using MLJModels.XGBoost_

n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = features * weights;


plain_regressor = XGBoostRegressor()


#bst = XGBoost.xgboost(Xmatrix,  y[train])


# test with linear data:
fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 1,
                                          features, labels);

rpred = predict(plain_regressor, fitresultR, features);

@test fitresultR isa MLJBase.fitresult_type(plain_regressor)

info(XGBoostRegressor)


end
true
