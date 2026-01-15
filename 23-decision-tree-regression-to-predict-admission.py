# First Step - Imports
import pandas as pd;
import numpy as np;

from sklearn.tree import DecisionTreeRegressor;
from sklearn.model_selection import cross_val_score;
from sklearn.model_selection import KFold;

# Second Step - Prepair Data
pd.set_option('display.max_columns', 9);
file = pd.read_csv('././data-sets/admission_predict.csv');

file.head();

# removing irrelevants columns (variables)
file.drop('Serial No.', axis=1, inplace=True);

file.head();

# Third Step - Define Target and Variables
y = file["Chance of Admit "] # Target
x = file.drop('Chance of Admit ', axis=1); # All columns without Chance of Admit will be our X

kfold = KFold(n_splits = 5, random_state = 7, shuffle = True);

model = DecisionTreeRegressor();
result = cross_val_score(model, x, y, cv = kfold);

print('Coeficiente de determinação R2 ', result.mean());


