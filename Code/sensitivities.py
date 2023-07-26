

#------------------LOGISTIC REGRESSION--------------------------


"""
D

Perform the following sensitivities:
Vary multi_class and use all available options. { ‘ovr’, ‘multinomial’}
Vary 3 solvers and use all available options. ‘lbfgs’ and other 2 : { ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}

"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Training data
X = np.array([[...],   # Insert your training data (features) here
              [...],
              ...
              [...]])
y = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # Insert your corresponding class labels here

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy: {:.2f}".format(accuracy))



#------------------SVM--------------------------

"""
K

Perform the following sensitivities:

Vary kernel and use all available options. {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} 
Vary degree and use 10 values.
Vary C until finding the optimal value.

"""

from sklearn.svm import SVC
# Create the SVM model
model = SVC(kernel='linear')

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy: {:.2f}".format(accuracy))



#------------------ k-NN --------------------------

"""
M

Perform the following sensitivities:

Vary weights and use all available options. {‘uniform’, ‘distance’}
Vary algorithm and use all available options. { ‘ball_tree’, ‘kd_tree’, ‘brute’}
Vary n_neighbors until finding the optimal value.

"""

from sklearn.neighbors import KNeighborsClassifier 
# Create the k-NN model
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy: {:.2f}".format(accuracy))




#------------------ Decision Tree --------------------------



from sklearn.tree import DecisionTreeClassifier


"""
S

Perform the following sensitivities:

Vary criterion and use all available options. ( {“gini”, “entropy”, “log_loss”} , max_depth = 1000)
Vary max_depth and use 10 values.( 10, 20, 40, 80, 160, 320, 660 ,1320 ,2640, 5280)

"""
# Create the Decision Tree model
model = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy: {:.2f}".format(accuracy))