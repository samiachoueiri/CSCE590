import numpy as np
import matplotlib.pyplot as plt

# Define a simple quadratic function
def func(x):
    return x**2 + 4 * x + 4

# Gradient of the quadratic function
def grad(x):
    return 2 * x + 4

# ADAM optimizer implementation
def adam_optimize(grad, x0, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iters=20000):
    # Initialize ADAM parameters
    m = 0  # First moment (mean of gradients)
    v = 0  # Second moment (mean of squared gradients)
    t = 0  # Time step
    x = x0  # Initial point
    x_values = [x]

    # Optimization loop
    for i in range(num_iters):
        t += 1  # Increment time step

        # Compute the gradient
        g = grad(x)

        # Update first moment estimate (exponential moving average)
        m = beta1 * m + (1 - beta1) * g

        # Update second moment estimate (exponential moving average of squared gradients)
        v = beta2 * v + (1 - beta2) * (g ** 2)

        # Bias-corrected first and second moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Parameter update with adaptive learning rate
        x -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        # Store the updated x values for visualization
        x_values.append(x)

    return x, x_values

# Initial point
x0 = 10  # Start from a point far from the minimum

# Optimize the function to find the minimum
optimal_x, x_values = adam_optimize(grad, x0)

# Plotting the function and the optimization path
x_range = np.linspace(-10, 10, 100)
y_range = func(x_range)

plt.plot(x_range, y_range, label="f(x) = x^2 + 4x + 4")
plt.plot(x_values, func(np.array(x_values)), 'ro-', label="Optimization path")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("ADAM Optimization Path")
plt.legend()
plt.show()

print("Optimal value of x:", optimal_x)
