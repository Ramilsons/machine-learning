# Outliers are values that there are exceding the normal values. 
# Example: There is a "altura_humana_metros" column and there are values 5, 7, 9...This values are outliers.

# Why we need to do?
# We need use the boxplot to see this outliers and decide remove or not this values. Run one model without this values and another with this values, the best scoree is keeping.

import matplotlib.pyplot as plt;
import pandas as pd;

pd.set_option('display.max_columns', 10);
file = pd.read_csv('././data-sets/traffic-collision-data-from-2010-to-present.csv');

file.boxplot(column = 'Census Tracts')
plt.show();

# Obs: This is not necessary to Decision Trees algorithms, only for another methods