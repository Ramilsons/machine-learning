# What is it? It's a method of pré processing to transform string data into number

import pandas as pd;

pd.set_option('display.max_columns', 10);
file = pd.read_csv('././data-sets/traffic-collision-data-from-2010-to-present.csv');

# In this data, there is a column "Area Name" where its type is String. Like "Olympic", "Foothill", "Southeast"
# We need transform it into Number

# We can't tranform creating only id. Like Olympic -> 1, Foothill -> 4, Southeast -> 8 e etc. Because this affect the mathematic count behind of models.

# We need to use the script below to transform using the best way
area_encode = pd.get_dummies(file['Area Name']);  # Here is happening the trasforming

concat = pd.concat([file, area_encode], axis = 1); # Saving the new columns generated into previou sheet
concat.drop('Area Name', axis = 1, inplace = True); # Removing the oldest column (in the incorrect type)

print('Final After Processing: ', concat);