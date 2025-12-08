# 30% dos dados para teste e 70% para treino
# Calcular o quoficiente R2 para os dados de teste que tenta prever os preços das casas
# Dividir as variáveis preditoras e a variavel target que nesse caso é o price

import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression;

pd.set_option('display.max_columns', 21); # To force the prompt (log) show all (21) columns
file = pd.read_csv('././data-sets/kc_house_data.csv');

print('file', file)
file.head();

def excludeIrrelavantsColumns():
    file.drop('id', axis = 1, inplace = True);
    file.drop('date', axis = 1, inplace = True);
    file.drop('zipcode', axis = 1, inplace = True);
    file.drop('lat', axis = 1, inplace = True);
    file.drop('long', axis = 1, inplace = True);


excludeIrrelavantsColumns();

file.head();


# Set "variaveis preditorias" and the "target variable"
y = file['price'];

fileWithoutTarget = file.drop('price', axis = 1);
x = fileWithoutTarget; # All columns will be variavel preditora it's except the price (it's our target variable)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30);

model = LinearRegression(); # To instance the Linear Regression Method;
model.fit(x_train, y_train); # Apply the data into the model

# It's calculating the coefficient R2
result = model.score(x_test, y_test);

print(result);