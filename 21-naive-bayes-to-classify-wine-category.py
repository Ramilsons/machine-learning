import pandas as pd;
from sklearn.model_selection import StratifiedKFold;
from sklearn.model_selection import cross_val_score;
from sklearn.naive_bayes import GaussianNB; # NEWEST

pd.set_option('display.max_columns', 13); 

dataSet = pd.read_csv('././data-sets/wine_dataset.csv');

# Third Step - Define Target and Variables
y = dataSet["style"] # Target
x = dataSet.drop('style', axis=1); # All columns without price will be our X

skfold = StratifiedKFold(n_splits = 3);

model = GaussianNB();

result = cross_val_score(model, x, y, cv = skfold);
print('Accuracy', result.mean());