# Objetivo é pegar um data set sobre características da flor Iris e classificar ela em 3 grupos: setosa, versicolor e virginica.


from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import GridSearchCV;

import pandas as pd;
import numpy as np;


pd.set_option('display.max_columns', 4);
data = load_iris();

x = pd.DataFrame(data.data, columns = [data.feature_names]);
y = pd.Series(data.target);

# Normalizing the data
normalizer = MinMaxScaler(feature_range = (0, 1));
x_normalized = normalizer.fit_transform(x);

# Setting the values of parameters that will be test
kValues = np.array([3, 5, 7, 9, 11]);
distanceCalcMethods = ['minkowski', 'chebyshev', 'euclidean'];
pValues = np.array([1, 2, 3, 4, ]);

gridValues = { 'n_neighbors': kValues, 'p': pValues, 'metric': distanceCalcMethods };

x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size = 0.3, random_state = 16);


model = KNeighborsClassifier(n_neighbors = 5); # K = 5

# Creating the Grids
gridKNN = GridSearchCV(estimator = model, param_grid = gridValues, cv = 5);
gridKNN.fit(x_normalized, y);

print('Best Accuracy', gridKNN.best_score_);
print('Best K', gridKNN.best_estimator_.n_neighbors);
print('Best Distance Calculation Method', gridKNN.best_estimator_.metric);
print('Best P', gridKNN.best_estimator_.p);
