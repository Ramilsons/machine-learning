# First Step - Imports
import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression;
from sklearn.linear_model import Ridge;
from sklearn.linear_model import Lasso;
from sklearn.linear_model import ElasticNet;

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=45);

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

    print('**************************************');
    print('Regressão Linear: ', regResult, 'Regressão Ridge: ', ridgeResult, 'Regressão Lasso: ', lassoResult, 'Regressão Elastic: ', elasticResult);
    print('**************************************');

    # Simulate a real student to show chance of admit after apply the model
    simulateStudent = pd.DataFrame([{
        'GRE Score': 325,
        'TOEFL Score': 110,
        'University Rating': 4,
        'SOP': 4.5,
        'LOR ': 4.0,
        'CGPA': 9.1,
        'Research': 1
    }]);

    predict = reg.predict(simulateStudent);
    print('Student Chance of Admit: ', predict[0]);


regressionModels();

