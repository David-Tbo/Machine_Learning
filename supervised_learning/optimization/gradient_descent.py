import numpy as np
import matplotlib.pyplot as plt

# Univariate gradient descent
def f(x):
    return -3 * x + x ** 2

def gradf(x):
    return -3 + 2 * x

def gradient_descent_univariate(f, gradf, initial_guess, tau=0.3, tol=1e-4, max_iter=1000):
    res = [initial_guess]
    old = initial_guess
    new = initial_guess - tau * gradf(old)

    while abs(old - new) > tol and len(res) < max_iter:
        old = new
        new = old - tau * gradf(old)
        res.append(new)

    return np.array(res)

# Run gradient descent
res_univariate = gradient_descent_univariate(f, gradf, initial_guess=-2)
print(f"\nNumber of iterations (univariate): {len(res_univariate)}")

# Plot
x = np.linspace(-3, 4, 1000)
plt.figure(figsize=(8, 6))
plt.plot(x, f(x), label="f(x) = -3x + x^2")
plt.scatter(res_univariate, f(res_univariate), color="red", label="Gradient Descent Path")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Univariate Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

# Bivariate gradient descent
eta = 4

def f_bivariate(x, y):
    return x ** 2 + eta * y ** 2

def gradf_bivariate(x):
    return np.array([2 * x[0], 2 * eta * x[1]])

# Plot the function
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = f_bivariate(X, Y)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20, cmap="viridis")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Bivariate Function Contour")
plt.grid(True)

# Gradient descent with fixed iterations
def gradient_descent_bivariate_fixed(f, gradf, initial_guess, tau=0.1, niter=100):
    res = np.zeros((niter + 1, 2))
    curr = np.array(initial_guess)
    res[0] = curr

    for i in range(niter):
        curr = curr - tau * gradf(curr)
        res[i + 1] = curr

    return res

res_bivariate_fixed = gradient_descent_bivariate_fixed(f_bivariate, gradf_bivariate, initial_guess=[0.7, -0.4], tau=0.8 / eta, niter=100)
plt.plot(res_bivariate_fixed[:, 0], res_bivariate_fixed[:, 1], marker="o", color="red", label="Gradient Descent Path")
plt.legend()
plt.show()

# Gradient descent with while loop
def gradient_descent_bivariate_while(f, gradf, initial_guess, tau=0.1, tol=1e-8, max_iter=1000):
    res = [initial_guess]
    old = np.array(initial_guess)
    new = old - tau * gradf(old)

    while np.max(np.abs(old - new)) > tol and len(res) < max_iter:
        old = new
        new = old - tau * gradf(old)
        res.append(new)

    return np.array(res)

res_bivariate_while = gradient_descent_bivariate_while(f_bivariate, gradf_bivariate, initial_guess=[0.5, 0.5], tau=0.1 / eta)
print(f"\nNumber of iterations (bivariate while loop): {len(res_bivariate_while)}")

# Plot
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20, cmap="viridis")
plt.plot(res_bivariate_while[:, 0], res_bivariate_while[:, 1], marker="o", color="red", label="Gradient Descent Path")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Bivariate Gradient Descent (While Loop)")
plt.legend()
plt.grid(True)
plt.show()