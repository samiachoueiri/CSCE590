import numpy as np

# Gradient descent function with additional diagnostics
def gradient_descent(A, b, lambda_, learning_rate, max_iter=2000, epsilon=1e-5):
    # Get the number of features
    n_features = A.shape[1]

    # Initialize x with zeros
    x = np.zeros(n_features)

    # Gradient descent iterations
    for iteration in range(max_iter):
        # Compute the gradient
        gradient = 2 * np.dot(A.T, np.dot(A, x) - b) + 2 * lambda_ * x

        # Update the parameter vector x using the gradient
        x_new = x - learning_rate * gradient

        # Convergence condition
        if np.linalg.norm(x_new - x) < epsilon:
            print(f"Converged in {iteration} iterations")
            break

        x = x_new

    else:
        print("Did not converge within the maximum iterations")

    return x

# Cost function
def cost(A, b, x, lambda_):
    error = np.dot(A, x) - b
    cost = np.dot(error, error) + lambda_ * np.dot(x, x)
    return cost

# Normalize data (optional but can help)
def normalize(matrix):
    norm = np.linalg.norm(matrix, axis=0)
    return matrix / norm

# Example matrix A (m x n)
A = np.array([[1, 1, 1], [0, 0, 1], [1, 0, 1], [2, 0, 5], [-7, 8, 0], [1, 2, -1]])
# Example vector b (m x 1)
b = np.array([3, 1, 2, 8, 0, 1])

# Regularization parameter
lambda_ = 0.1

# Learning rate for gradient descent
learning_rate = 0.001  # Decreased learning rate for stability

# Solve for x using gradient descent
x_optimal = gradient_descent(A, b, lambda_, learning_rate)

# Compute the cost for the optimal solution
final_cost = cost(A, b, x_optimal, lambda_)

print("Optimal x:", x_optimal)
print("Final cost:", final_cost)
