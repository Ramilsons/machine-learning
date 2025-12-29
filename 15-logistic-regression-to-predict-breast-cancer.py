# Objetivo
# criar uma fórmula matemática (modelo) que, ao receber as medidas de uma nova mama, calcula uma probabilidade e decide o diagnóstico baseando-se no comportamento histórico de milhares de outros exames que ele analisou durante o treino.

import pandas as pd;
import numpy as np;
from sklearn.model_selection import GridSearchCV;
from sklearn.linear_model import LogisticRegression;
from sklearn.datasets import load_breast_cancer;

pd.set_option('display.max_columns', 30);
data = load_breast_cancer();

# Defining the Target and Variables 
x = pd.DataFrame(data.data, columns = [data.feature_names]);
y = pd.Series(data.target);

# Add Values to Test
cValues = np.array([95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]);
regValues = ['l1', 'l2'];

gridValues = {'C': cValues, 'penalty': regValues}

# Create the Model
model = LogisticRegression(solver = 'liblinear', max_iter=10000);

gridLogisticRegression = GridSearchCV(estimator = model,  param_grid = gridValues, cv=5);
gridLogisticRegression.fit(x, y);


# Showing the Accuracy (Result of Model)
print('Best Accuracy', gridLogisticRegression.best_score_);
print('Best Parameter C', gridLogisticRegression.best_estimator_.C);
print('Best Regularization', gridLogisticRegression.best_estimator_.penalty);
