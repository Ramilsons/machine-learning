import pandas as pd;
import numpy as np;
import os;
from sklearn.datasets import load_iris;
from sklearn.model_selection import cross_val_score;
from sklearn.model_selection import StratifiedKFold;
from sklearn.tree import DecisionTreeClassifier; # NEWEST
from sklearn.tree import export_graphviz; # NEWEST
import graphviz; # NEWEST
from sklearn.model_selection import GridSearchCV

dataset = load_iris();

x = pd.DataFrame(dataset.data, columns = [dataset.feature_names]);
y = pd.Series(dataset.target);

skfold = StratifiedKFold(n_splits = 5, random_state = 8, shuffle = True);

# Values to Test
min_split = np.array([2, 3, 4, 5, 6, 7, 8]);
max_level = np.array([3, 4, 5, 6]);
algorithm = ['gini', 'entropy']
grid_values = { 'min_samples_split': min_split, 'max_depth': max_level, 'criterion': algorithm }

model = DecisionTreeClassifier();
result = cross_val_score(model, x, y, cv = skfold);

print('Accuracy', result.mean());

gridDecisionTree = GridSearchCV(estimator = model, param_grid = grid_values, cv = 5);
gridDecisionTree.fit(x, y);

model.fit(x, y);

folder = './decision-tree-builds'
if not os.path.exists(folder):
    os.makedirs(folder)

file_path = os.path.join(folder, 'iris-classify-code-22.dot')

export_graphviz(
    model, 
    out_file=file_path, 
    feature_names=dataset.feature_names,
    class_names=dataset.target_names,
    filled=True, 
    rounded=True
)

with open(file_path, 'r') as f:
    dot_graph = f.read()

h = graphviz.Source(dot_graph)

print('Best minimum split: ', gridDecisionTree.best_estimator_.min_samples_split);
print('Best max level: ', gridDecisionTree.best_estimator_.max_depth);
print('Best algorithm: ', gridDecisionTree.best_estimator_.criterion);
print('Best accuracy: ', gridDecisionTree.best_score_);
