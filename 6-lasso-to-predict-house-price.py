# Lasso is a function that use L1 Method of regularization. The ridge regression (previous lesson) apply the L2 Method of regularization.
# The regularization is used to give lower weight avoiding distortions

import pandas as pd;
from sklearn.model_selection import train_test_split;

# Linear Regression - It doesn't use any method of regularization
from sklearn.linear_model import LinearRegression; 

# Linear Regression + L2 Method Regularization = Ridge Regression
from sklearn.linear_model import Ridge; # Oldest

# Linear Regression + L1 Method Regulazization = Lasso
from sklearn.linear_model import Lasso; # Newest

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
y = file["price"]; # Target
x = file.drop('price', axis=1); # Variables

# Fourth Step - Split the Data (Train and Test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=14);

# Fifth Step - Apply the Model
modelRegression = LinearRegression();
modelRegression.fit(x_train, y_train);
resultRegression = modelRegression.score(x_test, y_test);

modelRidgeRegression = Ridge(alpha=50);
modelRidgeRegression.fit(x_train, y_train);
resultRidgeRegression = modelRidgeRegression.score(x_test, y_test);

modelLasso = Lasso(alpha=1000, max_iter=1000, tol=0.1); # Change the alpha (0 - infinite) to change result. Alpha 0 is equal regression without ridge. But you can change to verify which value improve your result.

modelLasso.fit(x_train, y_train);
resultLasso = modelLasso.score(x_test, y_test);

# Sixth Step - Result
print('**************************************');
print("Linear Regression Result:");
print(resultRegression); # Close to 1 is the better result

print('**************************************');
print("Ridge Regression Result:"); 
print(resultRidgeRegression); # Close to 1 is the better result

print('**************************************');
print("Lasso Result:"); 
print(resultLasso); # Close to 1 is the better result