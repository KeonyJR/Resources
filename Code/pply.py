import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score



# Aplicar regresión polinómica de grado 2 en los datos de entrenamiento
poly_features = PolynomialFeatures(degree=3)
x_train_poly = poly_features.fit_transform(X_train)

model = LinearRegression()
model.fit(x_train_poly, y_train)

# Predicción en el conjunto de validación
x_val_poly = poly_features.transform(X_test)
y_pred_val = model.predict(x_val_poly)

# Calcular el coeficiente de determinación R² en el conjunto de validación
y_val=y_test
r2 = r2_score(y_val, y_pred_val)
print("Coeficiente de determinación R² en validación:", r2)

y_hat =y_pred_val
x1 = np.asanyarray(X_test)
y1 = np.asanyarray(y_test)
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y1) ** 2))


print("R2-score: %.2f" % r2_score(y_hat ,y1) )

for idx, col_name in enumerate(col):
    print("The coefficient for {} is {}".format(col_name, model.coef_[0][idx]))
