import pandas as pd;

file = pd.read_csv('././data-sets/column_2C_weka.csv');
file.head();

missing = file.isnull().sum();
percentMissingData = (file.isnull().sum() / len(file['pelvic_incidence'])) * 100;

print('Percent Missing');
print(percentMissingData);
print('-----------------');


print('Types Of Data');
print(file.dtypes);
print('-----------------');

# Trasforming Text Data into Number Data (1 and 0)
file['class'] = file['class'].replace('Abnormal', 1);
file['class'] = file['class'].replace('Normal', 0);

y = file['class'];
x = file.drop('class', axis = 1);

import numpy as np;
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

min_split = np.array([2, 3, 4, 5, 6, 7, 8]);
max_level = np.array([3, 4, 5, 6 ])
algorithm = ['gini', 'entropy']
grid_values = { 'min_samples_split': min_split, 'max_depth': max_level, 'criterion': algorithm }

model = DecisionTreeClassifier()

gridDecisionTree = GridSearchCV(estimator = model, param_grid = grid_values, cv = 5);
gridDecisionTree.fit(x, y);

print('Best minimum split: ', gridDecisionTree.best_estimator_.min_samples_split);
print('Best max level: ', gridDecisionTree.best_estimator_.max_depth);
print('Best algorithm: ', gridDecisionTree.best_estimator_.criterion);
print('Best accuracy: ', gridDecisionTree.best_score_)
