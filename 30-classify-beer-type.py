# 2) Remover classes que não tiverem pelo menos 1000 amostras OK
# 3) Verificar se o tipo das variaveis são numericos. Se não, transforma em numerico com one hot encoding OK
# 4) Dados missing - preencher com media ou mediana em casos de falta de preenchimento
# 5) Remover variaveis se elas são correlatas
# 6) Remover variavel se não fizer sentido
# 7) Verificar se existem outliers
# 8) Aplicar maxClassifier pra padronizar os dados pro KNN
# 9) Dividir dados de treino e teste em Kfold
# 10) Aplicar gridSearchCv pra testar diferentes parametros
# 11) Criar função com diversos modelos de classificação
# 12) Testar melhor acurácia 

import pandas as pd;

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
            print(f"Coluna '{col}' convertida: {labels[0]}->0, {labels[1]}->1");

encode_binary_columns();

# Percent of data missing
percentOfDataMissingEachColumn = (file_filtered.isnull().sum() / file.shape[0]) * 100;
print(percentOfDataMissingEachColumn);

# tranforming the other columns that it isn't a binary column into one hot enconding
def enconde_one_hot():
    columns_target = ['Name', 'URL', 'Style', 'BrewMethod', 'PrimingMethod', 'PrimingAmount']

    area_encode = pd.get_dummies(file_filtered, columns=columns_target);
    concat = pd.concat([file_filtered, area_encode], axis = 1);

    concat.drop('Name', axis = 1, inplace = True);
    concat.drop('URL', axis = 1, inplace = True);
    concat.drop('Style', axis = 1, inplace = True);
    concat.drop('BrewMethod', axis = 1, inplace = True);
    concat.drop('PrimingMethod', axis = 1, inplace = True);
    concat.drop('PrimingAmount', axis = 1, inplace = True);

    return concat;

new_file_after_one_hot_enconding = enconde_one_hot();
#print(new_file_after_one_hot_enconding.dtypes)
