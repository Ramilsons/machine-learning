import pandas as pd;
from sklearn.model_selection import StratifiedKFold;
from sklearn.model_selection import cross_val_score;
from sklearn.linear_model import LogisticRegression;

pd.set_option('display.max_columns', 64);
file = pd.read_csv('././data-sets/Data_train_reduced.csv');


# PREPAIR THE DATA - FORMAT AND REMOVE UNUSED COLUMNS 
file.dtypes # verify types of columns - if there are columns on string or object we need to change the format for int or float

missingData = file.isnull().sum();
percentMissingData = (file.isnull().sum() / len(file['Product'])) * 100;
print(percentMissingData);

file.drop('q8.20', axis = 1, inplace = True);
file.drop('q8.18', axis = 1, inplace = True);
file.drop('q8.17', axis = 1, inplace = True);
file.drop('q8.8', axis = 1, inplace = True);
file.drop('q8.9', axis = 1, inplace = True);
file.drop('q8.10', axis = 1, inplace = True);
file.drop('q8.2', axis = 1, inplace = True);
file.drop('Respondent.ID', axis = 1, inplace = True);
file.drop('Product', axis = 1, inplace = True);
file.drop('q1_1.personal.opinion.of.this.Deodorant', axis = 1, inplace = True);

# Completing data missing with median 
file['q8.12'].fillna(file['q8.12'].median(), inplace = True);
file['q8.7'].fillna(file['q8.7'].median(), inplace = True);

# Defining the Target and Variables 
y = file['Instant.Liking'];
x = file.drop('Instant.Liking', axis = 1);

stratifiedKFold = StratifiedKFold(n_splits = 5);

# Create the Model
model = LogisticRegression(penalty = 'l2', solver = 'liblinear');
result = cross_val_score(model, x, y, cv = stratifiedKFold);


# Showing the Accuracy (Result of Model)
print('Result: ', result.mean());
