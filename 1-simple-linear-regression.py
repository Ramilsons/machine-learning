from sklearn.datasets import make_regression;
import matplotlib.pyplot as plt;

# Generate a random large data
x, y = make_regression(n_samples=200, n_features=1, noise=20);

# Showing in the graph
plt.scatter(x, y);
plt.show();


# ***************** #

from sklearn.linear_model import LinearRegression;
model = LinearRegression();

model.fit(x, y); # All Math Equations was created here. Derivadas, Quoficientes Angulares e etc. All was done automatically using the library.
model.intercept_; # Quoficientes LINEAR --> y = mx + b... O B é o Quoficiente LINEAR, ou seja o intercept_ representa o B da equação.
model.coef_; # Quoficientes ANGULAR --> y = mx + b... O M é o Quoficiente Angular, ou seja o COEF representa o M da equação.

# ***************** #

# Showing the model results'
import numpy as np;

xreg = np.arange(-3, 3, 1); # The arrange its saying that we want to create a array from -3 to 3 with a step of 1.

m_value = model.coef_[0];
b_value = model.intercept_;

plt.scatter(x, y);
plt.plot(xreg, m_value * xreg - b_value, color='purple'); # The second parameter is the equation of the line. y = mx + b.
plt.show();