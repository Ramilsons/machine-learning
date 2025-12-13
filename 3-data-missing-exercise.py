import pandas as pd;

pd.set_option('display.max_columns', 42); 
pd.set_option('display.max_rows', None);

file = pd.read_csv('././data-sets/2015-building-energy-benchmarking.csv');

percentOfDataMissingEachColumn = (file.isnull().sum() / file.shape[0]) * 100;
print(percentOfDataMissingEachColumn);


print('**************** BEFORE TREATMENT **********************');
# print(file['ENERGYSTARScore']);

# Changing the "NaN" (null - missing) values to median of this columns (usigin the other rows);
file['ENERGYSTARScore'] = file['ENERGYSTARScore'].fillna(file['ENERGYSTARScore'].median());
print('**************** AFTER TREATMENT **********************');
# print(file['ENERGYSTARScore']);

