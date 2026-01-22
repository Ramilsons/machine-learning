# Esse método vai eliminando cada feature (variavel) até chegar no número que definimos. 
# Primeiro ele vai eliminar a que tem pior quoficiente (pontuação) para o nosso modelo. Depois vai eliminar a que tem segunda pior e assim por diante. Até chegar no número de feature que decidimos

import pandas as pd;
from sklearn.feature_selection import RFE;
from sklearn.linear_model import Ridge;

pd.set_option("display.max_columns", 7);
pd.set_option("display.width", 320);

file = pd.read_csv('././data-sets/admission_predict.csv');
file.drop('Serial No.', axis = 1, inplace = True);


y = file['Chance of Admit '];
x = file.drop('Chance of Admit ', axis = 1);

model = Ridge()


rfe = RFE(estimator = model, n_features_to_select = 5) # Queremos as 5 melhores features. Portanto, serão excluídas 2. (Total 7)
fit = rfe.fit(x, y);

# Showing the results
print("Número de atributos: ", fit.n_features_)
print("Atributos selecionados: ", fit.support_)
print("Ranking dos atributos: ", fit.ranking_) # Os selecionados são 1 - A pior tem nota maior