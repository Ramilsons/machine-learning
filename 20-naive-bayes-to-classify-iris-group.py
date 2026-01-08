import pandas as pd;
from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split;
from sklearn.naive_bayes import GaussianNB; # NEWEST

iris = load_iris();

x = pd.DataFrame(iris.data, columns = [iris.feature_names]);
y = pd.Series(iris.target);

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 67);

model = GaussianNB();
model.fit(x_train, y_train);

result = model.score(x_test, y_test);
print('Accuracy', result);