# Matriz de Confusão (Confusion Matrix) é uma tabela que mostra exatamente onde o seu modelo de Machine Learning está acertando e, principalmente, onde ele está sendo enganado.
# Para um modelo médico como o nosso, a acurácia de 96% diz que ele é bom, mas a Matriz de Confusão diz quem são os 4% que ele errou.

# Imagine uma tabela 2x2 que cruza o que o modelo previu com o que o paciente realmente tem:
# [[VN  FP]
#  [FN  VP]]  

# Verdadeiros Negativos (VN): O tumor era maligno e o modelo disse que era maligno. (Sucesso)
# Verdadeiros Positivos (VP): O tumor era benigno e o modelo disse que era benigno. (Sucesso)
# Falsos Positivos (FP): O tumor era maligno, mas o modelo disse que era benigno. (Erro Perigoso)
# Falsos Negativos (FN): O tumor era benigno, mas o modelo disse que era maligno. (Erro Conservador)


import pandas as pd;
import numpy as np;
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import train_test_split;
from sklearn.datasets import load_breast_cancer;
from sklearn.metrics import confusion_matrix; # NEWEST

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
# RESULT 
# [59  3] 
# [1  108]

# 59 -> Verdadeiros Negativos (VN): O tumor era maligno e o modelo disse que era maligno. (Sucesso)
# 1 -> Falsos Negativos (FN): O tumor era benigno, mas o modelo disse que era maligno. (Erro Conservador)
# 3 -> Falsos Positivos (FP): O tumor era maligno, mas o modelo disse que era benigno. (Erro Perigoso)
# 108 -> Verdadeiros Positivos (VP): O tumor era benigno e o modelo disse que era benigno. (Sucesso)
