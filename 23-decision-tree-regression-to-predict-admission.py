# First Step - Imports
import pandas as pd;
import numpy as np;

from sklearn.tree import DecisionTreeRegressor;
from sklearn.model_selection import cross_val_score;
from sklearn.model_selection import KFold;
from sklearn.model_selection import GridSearchCV;

# Second Step - Prepair Data
pd.set_option('display.max_columns', 9);
file = pd.read_csv('././data-sets/admission_predict.csv');

# removing irrelevants columns (variables)
file.drop('Serial No.', axis=1, inplace=True);


# Third Step - Define Target and Variables
y = file["Chance of Admit "] # Target
x = file.drop('Chance of Admit ', axis=1); # All columns without Chance of Admit will be our X

min_split = np.array([2, 3, 4, 5, 6, 7]);
max_level = np.array([3, 4, 5, 6, 7, 9, 11]);
algorithm = ['squared_error', 'poisson', 'absolute_error' , 'friedman_mse']
grid_values = { 'min_samples_split': min_split, 'max_depth': max_level, 'criterion': algorithm }

model = DecisionTreeRegressor();

gridDecisionTree = GridSearchCV(estimator = model, param_grid = grid_values, cv = 5);
gridDecisionTree.fit(x, y);

print('Best minimum split: ', gridDecisionTree.best_estimator_.min_samples_split);
print('Best max level: ', gridDecisionTree.best_estimator_.max_depth);
print('Best algorithm: ', gridDecisionTree.best_estimator_.criterion);
print('Best accuracy: ', gridDecisionTree.best_score_);


