import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def is_convex_or_concave(func, variables):
    """
    Checks if a 3-dimensional function is convex or concave.
    
    Parameters:
    - func: The function to be checked.
    - variables: The variables of the function (e.g., ['x', 'y', 'z']).

    Returns:
    - 'convex' if the function is convex.
    - 'concave' if the function is concave.
    - 'undetermined' if the convexity/concavity cannot be determined.
    """
    # Define variables
    x, y= variables

    # Define the function symbolically
    import sympy as sp
    expr = func(sp.Symbol(x), sp.Symbol(y))

    # Compute the Hessian matrix
    hessian = sp.hessian(expr, (x, y))

    # Compute eigenvalues of the Hessian matrix
    eigenvalues = hessian.eigenvals()

    # Check if all eigenvalues are non-negative (convex)
    all_positive = all([eig_val >= 0 for eig_val in eigenvalues])

    # Check if all eigenvalues are non-positive (concave)
    all_negative = all([eig_val <= 0 for eig_val in eigenvalues])

    if all_positive:
        return 'convex'
    elif all_negative:
        return 'concave'
    else:
        return 'undetermined'


# Define the range for x1 and x2
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
x1, x2 = np.meshgrid(x1, x2)
z = abs(x1) + abs(x2)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(x1, x2, z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Z')
ax.set_title('Plot of z = |x1| + |x2|')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

# Define the function
def test_function(x1, x2):
    return abs(x1) + abs(x2)
# # Test convexity/concavity
# convexity = is_convex_or_concave(test_function, ['x1', 'x2'])
# print("Convexity/Concavity:", convexity)


# Show the plot
plt.show()
