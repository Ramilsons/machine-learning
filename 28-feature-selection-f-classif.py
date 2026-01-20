import pandas as pd;
from sklearn.datasets import load_iris;
from sklearn.feature_selection import SelectKBest;
from sklearn.feature_selection import f_classif;

iris = load_iris();

x = pd.DataFrame(iris.data, columns = [iris.feature_names]);
y = pd.Series(iris.target);

algorithm = SelectKBest(score_func= f_classif, k = 2);
bests_pedicts = algorithm.fit_transform(x, y);

# Results
print('Scores: ', algorithm.scores_);
print('Transform Result:\n', bests_pedicts);