# Nesse código vamos buscar utilizar os conceitos de AUC (area under curve) e ROC (Receiver Operating Characteristic) 
# para entender a qualidade no threshold (limiar de decisão) que foi utilizado na nossa sigmóide


import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import train_test_split;
from sklearn.datasets import load_breast_cancer;
from sklearn.metrics import confusion_matrix; 
from sklearn.metrics import roc_curve;  # NEWEST

pd.set_option('display.max_columns', 30);
data = load_breast_cancer();

# Defining the Target and Variables 
x = pd.DataFrame(data.data, columns = [data.feature_names]);
y = pd.Series(data.target);

y.value_counts();

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 9);

model = LogisticRegression(solver = 'liblinear', C = 95, penalty = 'l1');
model.fit(x_train, y_train);

accuracy = model.score(x_test, y_test);
print('Accuracy', accuracy);


prediction = model.predict(x_test);
print(confusion_matrix(y_test, prediction)); 

# NEWEST PART
prediction_proba = model.predict_proba(x_test);
# print('Predicion', prediction_proba); # Probabilidade de cada valor fazer parte da classe. Lembrando que o threshold padrão do sklearn é 0.5, ou seja, se a probabilidade de certo valor foi 0.6, então o algortimo irá entender como 1

probs = prediction_proba[:, 1]; # : Para receber todos as linhas, da coluna 
                             # 1 Para receber apenas da primeira colunas. Ou seja, estamos recebendo todas as linhas da coluna 1

fpr, tpr, thresholds = roc_curve(y_test, probs); # Passamos os valores reais e os valores previstos, com isso a função irá calcular e nos retornar o FPR, TPR e os possíveis thresholds testados

print(f"{"FPR         "} | {"TPR         "} | {"Threshhold"}");
for fpr_line, tpr_line, threshhold_line in zip(fpr, tpr, thresholds):
    # Imprimind os valores de FPR e TRP retornado em cada threshold testado
    print(f"{fpr_line:<12.4f} | {tpr_line:<20.4f} | {threshhold_line:<10.4f}"); # Relembrando - quanto MAIOR o TPR MELHOR e quanto MENOR o FPR MELHOR

# Showing on Graph
plt.scatter(fpr, tpr);
plt.show();