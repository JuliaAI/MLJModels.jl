module TestDecisionTree

# using Revise
using Test
import CategoricalArrays
using MLJBase

# load code to be tested:
import MLJModels
import DecisionTree 
using MLJModels.DecisionTree_

# get some test data:
task = load_iris();
X, y = X_and_y(task)       # a table and CategoricalVector

baretree = DecisionTreeClassifier()

baretree.max_depth = 1
fitresult, cache, report = MLJBase.fit(baretree, 1, X, y);
baretree.max_depth = -1 # no max depth
fitresult, cache, report = MLJBase.update(baretree, 1, fitresult, cache, X, y);

# in this case decision tree is a perfect predictor:
yhat = MLJBase.predict_mode(baretree, fitresult, X);
@test yhat == y

# but pruning upsets this:
baretree.post_prune = true
baretree.merge_purity_threshold=0.1
fitresult, cache, report = MLJBase.update(baretree, 2, fitresult, cache, X, y)
yhat = MLJBase.predict_mode(baretree, fitresult, X);
@test yhat != y
yhat = MLJBase.predict(baretree, fitresult, X);

# check preservation of levels:
yyhat = predict_mode(baretree, fitresult, MLJBase.selectrows(X, 1:3))
@test MLJBase.classes(yyhat[1]) == MLJBase.classes(y[1])

info(baretree)

# # testing machine interface:
# tree = machine(baretree, X, y)
# fit!(tree)
# yyhat = predict_mode(tree, MLJBase.selectrows(X, 1:3))
using Random: seed!
seed!(0)

n,m = 10^3, 5 ;
raw_features = rand(n,m);
weights = rand(-1:1,m);
labels = raw_features * weights;
features = MLJBase.table(raw_features);

R1Tree = DecisionTreeRegressor(min_samples_leaf=5, pruning_purity_threshold=0.1)
R2Tree = DecisionTreeRegressor(min_samples_split=5)
model1, = MLJBase.fit(R1Tree,1, features, labels)

vals1 = MLJBase.predict(R1Tree,model1,features)
R1Tree.post_prune = true
model1_prune, = MLJBase.fit(R1Tree,1, features, labels)
vals1_prune = MLJBase.predict(R1Tree,model1_prune,features)
@test vals1 !=vals1_prune

@test DecisionTree.R2(labels, vals1) > 0.8

model2, = MLJBase.fit(R2Tree, 1, features, labels)
vals2 = MLJBase.predict(R2Tree, model2, features)
@test DecisionTree.R2(labels, vals2) > 0.8

info(R1Tree)

end
true
