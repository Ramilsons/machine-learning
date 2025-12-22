# Elastic Net is a operation that combine the mathematic of linear regression + lasso (l1) + ridge (l2)
# In the previous case we work with L1 and L2 separated, but with Elastic Net we can work L1 + L2 in the same moment

# We can control percent of L1 and Percent of L2 that we need work. The parameter in sklearn algoritm is "l1_ratio"

# First Step - Imports
import pandas as pd;
from sklearn.model_selection import train_test_split;
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

# Fifth Step - Apply the Model
modelElasticNet = ElasticNet(alpha=1, l1_ratio=0.9, tol=0.2, max_iter=5000); # In this case, 90% (0.9) of ElasticNet weight will be L1 Method (Lasso) and 10%(0.10) of weight will be L2 Method (Ridge).
modelElasticNet.fit(x_train, y_train);


# Sixth Step - Result
resultModelElasticNet = modelElasticNet.score(x_test, y_test);

print('**************************************');
print("Linear Regression Result:");
print(resultModelElasticNet); # Close to 1 is the better result
