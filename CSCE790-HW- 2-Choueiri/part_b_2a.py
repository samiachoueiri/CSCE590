import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create data points for x1, x2, and x3
x = np.linspace(-10, 10, 100)

# Define the line x1 = x2 = x3
x2 = x
x3 = x

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the line x1 = x2 = x3
ax.plot(x, x2, x3, label='x1 = x2 = x3', color='blue')

# Set labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('Sketch of the set S1: {x1 = x2, x2 = x3}')

# Show the plot
plt.legend()
plt.show()

