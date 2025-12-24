import pandas as pd;
from sklearn.model_selection import KFold;
from sklearn.model_selection import cross_val_score;
from sklearn.linear_model import LinearRegression;

pd.set_option('display.max_columns', 9);
file = pd.read_csv('././data-sets/admission_predict.csv');

file.drop('Serial No.', axis=1, inplace=True);

file.head();

y = file["Chance of Admit "] # Target
x = file.drop('Chance of Admit ', axis=1); # All columns without Chance of Admit will be our X

model = LinearRegression();
kfold = KFold(n_splits = 5, shuffle = True); # Set to split de Data in 5 parts

result = cross_val_score(model, x, y, cv = kfold); # Apply the model into each part of Kfold separation

print('Resultado Separado de Cada Fold (cluster): ', result);
print('Média Resultante: ', result.mean()); # Main result - We need to analysis this result to define next step