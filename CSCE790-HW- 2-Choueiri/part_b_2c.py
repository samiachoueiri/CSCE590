import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the range for x1, x2, and x3
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
x, y = np.meshgrid(x, y)

# Calculate z from the equation x1 + x2 + x3 = 1
z = 1 - x - y

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of S2
ax.plot_surface(x, y, z, alpha=0.5, color='blue', label='S2: x1 + x2 + x3 = 1')

# Plot the intersection line of S1 and S2
x_vals = np.linspace(-1, 1, 100)
y_vals = x_vals
z_vals = 1 - x_vals - y_vals
ax.plot(x_vals, y_vals, z_vals, color='red', label='Intersection (S3): x1 = x2 = x3')

# Set labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('Sketch of the intersection set S3: {x1 = x2 = x3, x1 + x2 + x3 = 1}')
ax.legend()

# Show the plot
plt.show()
