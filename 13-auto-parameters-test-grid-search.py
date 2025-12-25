# THE BIG DIFFERENCE BETWENT GRID SEARCH AND RANDOMIZED SEARCH IS THAT THE GRID METHOD WILL TEST ALL POSSIBLES COMBINATIONS. Whereas, the randomized search will test only the number of combinations defined on n_iter parameter

import pandas as pd;
from sklearn.model_selection import GridSearchCV;
from sklearn.linear_model import ElasticNet;

pd.set_option('display.max_columns', 9);
file = pd.read_csv('././data-sets/admission_predict.csv');

file.head();

file.drop('Serial No.', axis=1, inplace=True);

file.head();


y = file["Chance of Admit "]
x = file.drop('Chance of Admit ', axis=1);

valuesToTest = { 
                'alpha': [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
                'l1_ratio': [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            }

model = ElasticNet();

search = GridSearchCV(estimator = model, param_grid = valuesToTest, cv = 5);
    # Parameters Explations
        # estimatior -> We need set our model selected;
        # param_grid -> We need set all values that will be test
        # cv -> RandomizedSearch use KFold insider your function. So, cv=5 say to split de data into 5 fols using cross validation

    # THE BIG DIFFERENCE BETWENT GRID SEARCH AND RANDOMIZED SEARCH IS THAT THE GRID METHOD WILL TEST ALL POSSIBLES COMBINATIONS. Whereas, the randomized search will test only the number of combinations defined on n_iter parameter

search.fit(x, y);

print('Melhor Score', search.best_score_);
print('Melhor Alpha', search.best_estimator_.alpha);
print('Melhor l1_ratio', search.best_estimator_.l1_ratio);