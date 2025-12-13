import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;

pd.set_option('display.max_columns', 42); 
pd.set_option('display.max_rows', None);

dataSet = pd.read_csv('././data-sets/2015-building-energy-benchmarking.csv');

# IMPORTANT = It's possible corr method only numbers columns. String isn't possible to correlation
numeric_data = dataSet.select_dtypes(include=['number']);
print(numeric_data.corr());

plt.figure(figsize=(10,10));
sns.heatmap(numeric_data.corr());
plt.show();