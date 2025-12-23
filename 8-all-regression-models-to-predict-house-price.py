# First Step - Imports
import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression;
from sklearn.linear_model import Ridge;
from sklearn.linear_model import Lasso;
from sklearn.linear_model import ElasticNet;

# Second Step - Prepair Data
pd.set_option('display.max_columns', 12);
file = pd.read_csv('././data-sets/kc_house_data.csv');

file.head();

# removing irrelevants columns (variables)
file.drop('id', axis=1, inplace=True);
file.drop('date', axis=1, inplace=True);
file.drop('zipcode', axis=1, inplace=True);
file.drop('lat', axis=1, inplace=True);
file.drop('long', axis=1, inplace=True);

file.head();

# Third Step - Define Target and Variables
y = file["price"] # Target
x = file.drop('price', axis=1); # All columns without price will be our X

# Fourth Step - Split the Data (Train and Test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=14);

def regressionModels():
    reg = LinearRegression();
    ridge = Ridge();
    lasso = Lasso();
    elastic = ElasticNet();

    reg.fit(x_train, y_train);
    ridge.fit(x_train, y_train);
    lasso.fit(x_train, y_train);
    elastic.fit(x_train, y_train);

    regResult = reg.score(x_test, y_test);
    ridgeResult = ridge.score(x_test, y_test);
    lassoResult = lasso.score(x_test, y_test);
    elasticResult = elastic.score(x_test, y_test);

    print('Regressão Linear: ', regResult, 'Regressão Ridge: ', ridgeResult, 'Regressão Lasso: ', lassoResult, 'Regressão Elastic: ', elasticResult);

regressionModels();
