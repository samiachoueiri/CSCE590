import numpy as np
import matplotlib.pyplot as plt

# Given data
t_values = np.array([-2, -1, 0, 1, 2])
y_values = np.array([2, -10, 0, 2, 1])

# Define the objective function
def objective_function(x, t_values, y_values):
    ord=len(x)-1
    p_values = x[0]
    for i in range (ord):
        p_values = p_values + x[i+1]*(t_values**(i+1))
    return np.sum((y_values - p_values)**2)

# Define the gradient of the objective function
def gradient(x, t_values, y_values):
    ord=len(x)-1
    p_values = x[0]
    for i in range (ord):
        p_values = p_values + x[i+1]*(t_values**(i+1))
    residuals = y_values - p_values

    grad = [-2 * np.sum(residuals)]
    for i in range (ord):
        grad.append(-2 * np.sum(residuals * t_values**(i+1))) 
    return np.array([grad])

# Gradient descent algorithm
def gradient_descent(x,t_values, y_values, learning_rate=0.001, max_iter=10000, tol=1e-6):
    # Perform gradient descent
    for i in range(max_iter):
        grad = gradient(x, t_values, y_values)[0]
        x -= learning_rate * grad
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
    return x

# Create a figure and axis object
fig, ax = plt.subplots()
kay = []
error = []
for k in range (4):
    order = k+2
    kay.append(order-1)
    # Initialize coefficients
    x = np.zeros(order)
    # Run gradient descent
    coefficients = gradient_descent(x,t_values, y_values)
    cost = objective_function(coefficients, t_values, y_values)
    error.append(cost)
    # print(coefficients , cost)

    # plot output
    t = np.linspace(-2, 2, 100)
    
    ord=len(coefficients)-1
    y = coefficients[0]
    for i in range (ord):
        y = y + coefficients[i+1]*(t**(i+1))
    
    ax.plot(t, y, label=f'k = {order-1}')

t_true = [-2,-1,0,1,2]
y_true = [2,-10,0,2,1]
ax.scatter(t_true, y_true, color='red', linestyle='-')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Best fit function')
ax.legend()
plt.show()

print(kay)
print(error)
plt.plot(kay, error, color='blue', linestyle='-')
plt.xlabel('k')
plt.ylabel('error')
plt.title('k vs error')
plt.xticks([1,2,3,4])
plt.show()
