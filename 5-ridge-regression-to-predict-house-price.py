# Ridge regression is similar to linear regression but the difference is that ridge avoid big weights - 
# The ridge regression give lower weight avoiding distortions

# First Step - Imports
import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression; # Oldest
from sklearn.linear_model import Ridge; # Newest

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=14) # 30% percent of our data will be used to Test and 70% percent to train. Random state 14 is used to be equals of the Teacher of Course

# Fifth Step - Apply the Model
modelRegression = LinearRegression();
modelRegression.fit(x_train, y_train);
resultRegression = modelRegression.score(x_test, y_test);

modelRidgeRegression = Ridge(alpha=50); # Change the alpha (0 - infinite) to change result. Alpha 0 is equal regression without ridge. But you can change to verify which value improve your result.
modelRidgeRegression.fit(x_train, y_train);
resultRidgeRegression = modelRidgeRegression.score(x_test, y_test);

# Sixth Step - Result
print('**************************************');
print("Linear Regression Result:");
print(resultRegression); # Close to 1 is the better result
print('**************************************');
print("Ridge Regression Result:"); 
print(resultRidgeRegression); # Close to 1 is the better result