# A sigla KNN significa K-Nearest Neighbors, que em português é traduzido como K-Vizinhos Mais Próximos.
# O funcionamento é baseado na ideia de que "objetos semelhantes estão próximos uns dos outros". 
# Quando o algoritmo recebe um novo dado, ele calcula a distância desse ponto em relação aos outros dados conhecidos, identifica os k vizinhos mais próximos e decide a categoria do novo dado com base na maioria.

from sklearn.datasets import load_breast_cancer;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.preprocessing import MinMaxScaler;
import pandas as pd;

pd.set_option('display.max_columns', 30);
dados = load_breast_cancer();

x = pd.DataFrame(dados.data, columns = [dados.feature_names]);
y = pd.Series(dados.target);

# Normalizing the data
normalizer = MinMaxScaler(feature_range = (0, 1));
x_normalized = normalizer.fit_transform(x);

x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size = 0.3, random_state = 16);

model = KNeighborsClassifier(n_neighbors = 5); # K = 5
model.fit(x_train, y_train);

result = model.score(x_test, y_test);
print('Accuracy', result);
