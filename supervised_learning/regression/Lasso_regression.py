#!/usr/bin/env python
# coding: utf-8

# # Lasso example
# 
# Illustration of how the Lasso (Least Absolute Shrinkage and Selection Operator) selects variables depending on their correlation with the residual and the parameter $\lambda$.

# ## Generate the data
# - 15 observations
# - 3 features (X1, X2, X3)
# - 1 target variable y
# - X1 and X2 are correlated and X3 is less correlated with y

# In[1]:


import numpy as np

# Fixons une graine pour la reproductibilité
np.random.seed(42)

# Générons les données
n = 15
X1 = np.random.normal(0, 1, n)
X2 = X1 + np.random.normal(0, 0.5, n)  # X2 corrélé avec X1
X3 = np.random.normal(0, 1, n)  # X3 non corrélé avec X1/X2

# Variable cible : y = 2*X1 + 0.5*X2 + 0.1*X3 + bruit
y = 2 * X1 + 0.5 * X2 + 0.1 * X3 + np.random.normal(0, 0.5, n)

# Matrice X (avec une colonne de 1 pour l'intercept si nécessaire)
X = np.column_stack((X1, X2, X3))


# ## Standardize the data
# 
# Lasso is sensitive to the scale of the variables.

# In[3]:


X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std


# ## 3. Implement a simplifed version of Lasso regression
# 
# Iterative approach based on the KKT conditions to estimate the $\beta$.  
# 
# **Function to update the coefficients**  
# 
# For each $\beta_j$, we update the value using the soft-thresholding rule (derived from KKT conditions)
# 
# $$\beta_j \leftarrow \dfrac{S(\sum_{i=1}^{n}x_{ij}(y_i - \tilde{y}_i^{(j)}), \lambda)}{||x_j||_2^2}$$
# 
# where:
# 
# - $S(z, \lambda) = \text{sgn}(z)(|z|-\lambda)_{+}$ is the operator of soft-thresholding,
# - $\tilde{y}_i^{(j)} = \sum_{k \neq j}x_{ij}\beta_k$ is the prediction without the variable $j$.

# **Algorithm**

# In[ ]:


def soft_threshold(z, lambda_):
    if z > lambda_:
        return z - lambda_
    elif z < -lambda_:
        return z + lambda_
    else:
        return 0.0

def lasso_coordinate_descent(X, y, lambda_, max_iter=100, tol=1e-4):
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # Compute the partial residu without the variable j
            residual = y - X @ beta + X[:, j] * beta[j]
            # Update the beta_j
            z_j = X[:, j].T @ residual
            beta[j] = soft_threshold(z_j, lambda_) / (X[:, j].T @ X[:, j])

        # Verification of the convergence
        if np.linalg.norm(beta - beta_old, ord=np.inf) < tol:
            break

    return beta


# In[6]:


lambda_ = 0.5
beta_hat = lasso_coordinate_descent(X, y, lambda_)
print("Coefficients estimated :", beta_hat)


# - $\beta_1$ is correlated with $y$ and remain in the model.
# - $\beta_2$ is less correlated with $y$ but remain in the model.
# - $\beta_3$ has a weak correlation with $y$ and is eliminated.

# ## Check the KKT conditions
# 
# For every $\beta_j$, check if the KKT conditions are satisfied:

# In[7]:


for j in range(X.shape[1]):
    residual = y - X @ beta_hat
    correlation = X[:, j].T @ residual
    print(f"Variable {j+1}: |x_j^T(y - Xβ)| = {abs(correlation):.3f}, λ/2 = {lambda_/2:.3f}")


# - If $\beta_j = 0$, then $|x_j^T(y−X\beta)|$ must be $\leq \lambda / 2$.
