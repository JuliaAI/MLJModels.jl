module TestDecisionTree

# using Revise
using Test
import CategoricalArrays
using MLJ

# load code to be tested:
import MLJModels 
import DecisionTree # MLJModels.DecisionTree_ now available via lazy-loading
using MLJModels.DecisionTree_ 

# get some test data:
task = load_iris();
X, y = X_and_y(task) # a table and categorical vector
 
baretree = DecisionTreeClassifier(target_type=String)

baretree.max_depth = 1 
fitresult, cache, report = MLJ.fit(baretree, 1, X, y);
baretree.max_depth = -1 # no max depth
fitresult, cache, report = MLJ.update(baretree, 1, fitresult, cache, X, y);
@test fitresult isa MLJ.fitresult_type(baretree)
# in this case decision tree is a perfect predictor:
yhat = MLJ.predict_mode(baretree, fitresult, X);
@test yhat == y

# but pruning upsets this:
baretree.post_prune = true
baretree.merge_purity_threshold=0.1
fitresult, cache, report = MLJ.update(baretree, 2, fitresult, cache, X, y)
yhat = MLJ.predict_mode(baretree, fitresult, X);
@test yhat != y
yhat = MLJ.predict(baretree, fitresult, X);
cross_entropy(y, yhat)

# check preservation of levels:
yyhat = predict_mode(baretree, fitresult, MLJ.selectrows(X, 1:3))
@test CategoricalArrays.levels(yyhat) == CategoricalArrays.levels(y)

info(baretree)

# testing machine interface:
tree = machine(baretree, X, y)
fit!(tree)
yyhat = predict_mode(tree, MLJ.selectrows(X, 1:3))


end
true
