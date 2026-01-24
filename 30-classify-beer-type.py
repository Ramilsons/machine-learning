# 2) Remover classes que não tiverem pelo menos 1000 amostras OK
# 3) Verificar se o tipo das variaveis são numericos. Se não, transforma em numerico com one hot encoding OK
# 4) Dados missing - preencher com media ou mediana em casos de falta de preenchimento OK
# 5) Remover variaveis se elas são correlatas OK
# 6) Remover variavel se não fizer sentido OK
# 7) Verificar se existem outliers OK
# 8) Aplicar MinMaxScaler pra padronizar os dados pro KNN OK
# 9) Dividir dados de treino e teste em Kfold OK
# 10) Aplicar gridSearchCv pra testar diferentes parametros 
# 11) Criar função com diversos modelos de classificação OK
# 12) Testar melhor acurácia  OK

import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
from sklearn.feature_selection import SelectKBest;
from sklearn.feature_selection import f_classif;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import KFold;
from sklearn.model_selection import cross_val_score;
from sklearn.linear_model import LogisticRegression;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB; 
from sklearn.tree import DecisionTreeClassifier; 


pd.set_option('display.max_columns', 24);
pd.set_option('display.max_rows', None);

file = pd.read_csv('././data-sets/recipeData.csv',  encoding='latin-1');
file.head();

# step 1 remove lines that class doenst has 1000 samples
counts = file['StyleID'].value_counts();
valid_classes = counts[counts >= 1000].index

file_filtered = file[file['StyleID'].isin(valid_classes)].copy();
# print(len(file_filtered)); Number of lines after filter


# step 3 verify type of variables (columns)

# name, url, style, sugarscale and brew method are in wrong type (object)
# verify which has only two different values (in this cases, we can transform in 0 and 1 without run one hot enconding)
cols_object = file_filtered.select_dtypes(include=['object']).columns

def encode_binary_columns():
    for col in cols_object:
        if file_filtered[col].nunique() == 2:
            labels = file_filtered[col].unique()
            file_filtered[col] = file_filtered[col].map({labels[0]: 0, labels[1]: 1})
            # print(f"Coluna '{col}' convertida: {labels[0]}->0, {labels[1]}->1");

encode_binary_columns();

# Showing the correlations between columns
numeric_data = file_filtered.select_dtypes(include=['number']);
# print(numeric_data.corr());

#plt.figure(figsize=(10,10));
#sns.heatmap(numeric_data.corr());
# plt.show();

# Removing cases with correlation a lot
file_filtered.drop('FG', axis = 1, inplace = True);
file_filtered.drop('SugarScale', axis = 1, inplace = True);
file_filtered.drop('BoilSize', axis = 1, inplace = True);
file_filtered.drop('BoilGravity', axis = 1, inplace = True);


# Removing irrelevants columns
file_filtered.drop('BeerID', axis = 1, inplace = True); 
file_filtered.drop('UserId', axis = 1, inplace = True);
file_filtered.drop('Name', axis = 1, inplace = True);
file_filtered.drop('URL', axis = 1, inplace = True);
file_filtered.drop('Style', axis = 1, inplace = True);

# Percent of data missing
percentOfDataMissingEachColumn = (file_filtered.isnull().sum() / file.shape[0]) * 100;
print(percentOfDataMissingEachColumn);

# verify outliers
# file_filtered.boxplot(column = 'MashThickness')
# plt.show();

# file_filtered.boxplot(column = 'PitchRate')
# plt.show();

# file_filtered.boxplot(column = 'PrimaryTemp')
#plt.show();

# MashThickness > use median because has outliers
# PitchRate > use median because has outliers
# PrimaryTemp > use median because has outliers
# PrimingMethod > remove (muito poluída, com diferentes formatos e muito NA)
# PrimingAmount > remove (muito poluída, com diferentes formatos e muito NA)

file_filtered["MashThickness"] = file_filtered["MashThickness"].fillna(file_filtered["MashThickness"].median());
file_filtered["PrimaryTemp"] = file_filtered["PrimaryTemp"].fillna(file_filtered["PrimaryTemp"].median());
file_filtered["PitchRate"] = file_filtered["PitchRate"].fillna(file_filtered["PitchRate"].median());

file_filtered.drop('PrimingAmount', axis = 1, inplace = True);
file_filtered.drop('PrimingMethod', axis = 1, inplace = True);

percentOfDataMissingEachColumn = (file_filtered.isnull().sum() / file.shape[0]) * 100;
print(percentOfDataMissingEachColumn);

# tranforming the other columns that it isn't a binary column into one hot enconding
def enconde_one_hot():
    columns_target = ['BrewMethod']

    area_encode = pd.get_dummies(file_filtered, columns=columns_target, drop_first=True);
    return area_encode;

new_file_after_one_hot_enconding = enconde_one_hot();


# Defining Predict Values and Target
x = new_file_after_one_hot_enconding.drop("StyleID", axis = 1);
y = new_file_after_one_hot_enconding["StyleID"];


algorithm =  SelectKBest(score_func= f_classif, k = 5);
bests_pedicts = algorithm.fit_transform(x, y)

# Results
# Showing the relavance column score (quanto maior melhor)
for col, score in zip(x.columns, algorithm.scores_):
    print(f"Variável: {col:20} | Score: {score:.4f}")


# Normalizing the data (avoiding outliers)
normalizer = MinMaxScaler(feature_range = (0, 1));
x_normalized = normalizer.fit_transform(x);

kfold = KFold(n_splits = 5, shuffle = True); # Set to split de Data in 5 parts

def test_model():
    # logistic regression
    modelLogistcReg = LogisticRegression(solver = 'liblinear', max_iter=10000);
    scoreLogistcReg = cross_val_score(modelLogistcReg, x_normalized, y, cv=kfold, scoring='accuracy')

    # knn
    modelKNN = KNeighborsClassifier(n_neighbors = 5);
    scoreKNN = cross_val_score(modelKNN, x_normalized, y, cv=kfold, scoring='accuracy')

    # naive bayes
    modelNaiveBayes = GaussianNB();
    scoreNaiveBayes = cross_val_score(modelNaiveBayes, x_normalized, y, cv=kfold, scoring='accuracy')

    # decision tree
    modelDecisionTree = DecisionTreeClassifier(max_depth=10);
    scoreDecisionTree = cross_val_score(modelDecisionTree, x_normalized, y, cv=kfold, scoring='accuracy')

    allScores = [
        {
            "name": "LogisticRegression",
            "score": scoreLogistcReg.mean()
        },
        {
            "name": "KNN",
            "score": scoreKNN.mean()
        },
        {
            "name": "NaiveBayes",
            "score": scoreNaiveBayes.mean()
        },
        {
            "name": "DecisionTree",
            "score": scoreDecisionTree.mean()
        }
    ]

    print(allScores)

    bestScore = 0
    nameOfBestScore = "";
    for index, score in enumerate(allScores):
        if score["score"] > bestScore:
            bestScore = score["score"];
            nameOfBestScore = allScores[index]["name"]

    print("Best Model: ", nameOfBestScore);
    print("Result: ", bestScore);

test_model();