#!/usr/bin/env python
# coding: utf-8

### ==============================================================================
### Logistic Regression from scratch with Gradient Descent
### ==============================================================================

import warnings
warnings.filterwarnings("ignore")

### ==============================================================================
### The Algorithm Steps
### ==============================================================================

# 1. Build the design matrix, optionally adding an intercept column.
# 2. Initialize the coefficient vector beta.
# 3. Compute predicted probabilities: p = sigmoid(X @ beta).
# 4. Compute the gradient: Grad_J = (1 / n) * X.T @ (p -y)
# 5. Update the coefficients: beta = beta - learning_rate * Grad_J.
# 6. Repeat steps 3 to 5 until convergence or until the maximum number of iterations is reached.
# 7. Predict probabilities and classify them using a decision threshold.

import numpy as np

def add_intercept(X):
    return np.c_[np.ones(X.shape[0]), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, beta, learning_rate):
    n = X.shape[0]

    p = sigmoid(X @ beta)
    gradient = (1 / n) * X.T @ (p - y)

    beta = beta - learning_rate * gradient

    return beta


### ==============================================================================
### Test the implementation on a real dataset
### ==============================================================================

# 0. Load the dataset

from sklearn import datasets

df = datasets.load_breast_cancer()
X, y = df.data, df.target

# 0. Train-test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 1. Add intercept term to the features

X_design = add_intercept(X_train)

# 2. Initialize the coefficients (beta) to zeros

beta = np.zeros(X_design.shape[1])

# 3-5. Run gradient descent for a fixed number of iterations

for _ in range(1000):
    beta = gradient_descent(X_design, y_train, beta, learning_rate=0.1)


print("Beta coefficients (first 5):\n", beta[:5].round(4))


# 7. Predict probabilities and classify them using a decision threshold

preds = sigmoid(add_intercept(X_test) @ beta) >= 0.5

# Evaluate the accuracy of the model

accuracy = np.mean(preds == y_test)

print("Accuracy:", accuracy)


### ==============================================================================
### Class implementation
### We can wrap the above code into a class for better modularity and reusability.
### ==============================================================================
import numpy as np

class Logistic_Regression:
    
    def __init__(self, learning_rate=0.01, n_iter=1000):
        '''Initiate the constructor
            INPUT:
                learning_rate: magnitude of the step
                n_iter: number of iterations
        '''
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        '''Train the model
        INPUTS:
            X: the dataset of the features
            y: the target
        OUTPUTS:
            The model
        '''

        self.n_samples, self.p_features = X.shape
        
        # initialize the parameters:
        self.betas = np.zeros(self.p_features)

        self.X = X
        self.y = y

        for _ in range(self.n_iter):
            self.gradient_descent()
        
        return self

    def gradient_descent(self):

        p = self.sigmoid(self.X @ self.betas)
        gradient = (1 / self.n_samples) * self.X.T @ (p - self.y)

        self.betas = self.betas - self.learning_rate * gradient

    def predict(self, X):
        
        p = self.sigmoid(X @ self.betas)
        
        y_pred = np.where(p > 0.5 , 1 , 0)
        
        return y_pred

# Initialize the model

classifier = Logistic_Regression(learning_rate=0.01, n_iter=1000)


# Fit the model

classifier.fit(X_train, y_train)

# Accuracy

from sklearn.metrics import accuracy_score

# Accuracy on the training data

y_train_pred = classifier.predict(X_train)
training_data_accuracy = accuracy_score(y_train, y_train_pred)
print(training_data_accuracy)

# Accuracy on the test data

y_test_pred = classifier.predict(X_test)
test_data_accuracy = accuracy_score(y_test, y_test_pred)
print(test_data_accuracy)


# END