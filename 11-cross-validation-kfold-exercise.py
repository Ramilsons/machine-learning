# First Step - Imports
import pandas as pd;
from sklearn.linear_model import LinearRegression;
from sklearn.linear_model import Ridge;
from sklearn.linear_model import Lasso;
from sklearn.linear_model import ElasticNet;
from sklearn.model_selection import KFold;
from sklearn.model_selection import cross_val_score;

# Second Step - Prepair Data
pd.set_option('display.max_columns', 9);
file = pd.read_csv('././data-sets/admission_predict.csv');

file.head();

print('**************************************');
print(file.dtypes); # Verify Types Data of Each Column - If a Type is a string we need to transform into int or float (number format).

missingData = file.isnull().sum();
print('**************************************');
print(missingData); # Verify if it has a missing data - If it has a missing data we need to calculate how much percent and apply the average or the median

# removing irrelevants columns (variables)
file.drop('Serial No.', axis=1, inplace=True);

file.head();

# Third Step - Define Target and Variables
y = file["Chance of Admit "] # Target
x = file.drop('Chance of Admit ', axis=1); # All columns without Chance of Admit will be our X

# Fourth Step - Split the Data (Train and Test)
kfold = KFold(n_splits = 10, shuffle = True); # Set to split de Data in 5 parts

def regressionModels():
    reg = LinearRegression();
    ridge = Ridge();
    lasso = Lasso();
    elastic = ElasticNet();

    regResult = cross_val_score(reg, x, y, cv = kfold);
    ridgeResult = cross_val_score(ridge, x, y, cv = kfold);
    lassoResult = cross_val_score(lasso, x, y, cv = kfold);
    elasticResult = cross_val_score(elastic, x, y, cv = kfold);

    print('**************************************');
    print('Regressão Linear: ', regResult.mean(), '| Regressão Ridge: ', ridgeResult.mean(), '| Regressão Lasso: ', lassoResult.mean(), '| Regressão Elastic: ', elasticResult.mean());
    print('**************************************');


regressionModels();

