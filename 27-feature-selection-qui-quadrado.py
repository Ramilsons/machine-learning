from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Definindo as variaveis preditoras e target (simulando um dataset)
x = [[12, 2, 30], [15, 11, 6], [16, 8, 90], [5, 3, 20], [4, 14, 5], [2, 5, 70]]
y = [1, 1, 1, 0, 0, 0]

algorithm =  SelectKBest(score_func= chi2, k = 2); # Selecionando duas variveis com o maior chi-quadrado
bests_pedicts = algorithm.fit_transform(x, y)


# Results
print('Scores: ', algorithm.scores_);
print('Transform Result:\n', bests_pedicts); # Dataset apenas com as features selecionadas, removeu a feature que não influenciava no resultado
