# In the previous examples, we need to change manually each parameters (alpha, l1_ratio and etc). And this is not a good practice, becausa it need time a lot
# So, there are the Method RandomizedSearchCV and GridSearchCV where we can test multiple parameters automatically and see the best format and combination of parameters 

import pandas as pd;
from sklearn.model_selection import RandomizedSearchCV;
from sklearn.linear_model import ElasticNet;

pd.set_option('display.max_columns', 9);
file = pd.read_csv('././data-sets/admission_predict.csv');

file.head();

file.drop('Serial No.', axis=1, inplace=True);

file.head();


y = file["Chance of Admit "]
x = file.drop('Chance of Admit ', axis=1);

# Set the values of parameters that will be test
valuesToTest = { 
                'alpha': [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
                'l1_ratio': [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            }

model = ElasticNet();

search = RandomizedSearchCV(estimator = model, param_distributions = valuesToTest, n_iter = 150, cv = 5, random_state = 15);
    # Parameters Explations
        # estimatior -> We need set our model selected;
        # param_distributions -> We need set all values that will be test
        # n_iter -> Number max of combinations to test. So, if you set n_iter=2 the python will test only two combinations. If the value if big, the time of run is big. The number recommend is between 60 and 100. 
        # cv -> RandomizedSearch use KFold insider your function. So, cv=5 say to split de data into 5 fols using cross validation

search.fit(x, y);

print('Melhor Score', search.best_score_);
print('Melhor Alpha', search.best_estimator_.alpha);
print('Melhor l1_ratio', search.best_estimator_.l1_ratio);