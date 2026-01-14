import pandas as pd;
import os;
from sklearn.datasets import load_iris;
from sklearn.model_selection import cross_val_score;
from sklearn.model_selection import StratifiedKFold;
from sklearn.tree import DecisionTreeClassifier; # NEWEST
import graphviz; # NEWEST
from sklearn.tree import export_graphviz; # NEWEST

dataset = load_iris();

x = pd.DataFrame(dataset.data, columns = [dataset.feature_names]);
y = pd.Series(dataset.target);

skfold = StratifiedKFold(n_splits = 5, random_state = 8, shuffle = True);

model = DecisionTreeClassifier();
result = cross_val_score(model, x, y, cv = skfold);

print('Accuracy', result.mean());

model.fit(x, y);

folder = './decision-tree-builds'
if not os.path.exists(folder):
    os.makedirs(folder)

file_path = os.path.join(folder, 'iris-classify-code-22.dot')

export_graphviz(
    model, 
    out_file=file_path, 
    feature_names=dataset.feature_names,
    class_names=dataset.target_names,
    filled=True, 
    rounded=True
)

with open(file_path, 'r') as f:
    dot_graph = f.read()

h = graphviz.Source(dot_graph)

