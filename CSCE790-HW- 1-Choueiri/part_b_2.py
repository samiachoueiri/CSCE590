import numpy as np
import matplotlib.pyplot as plt

# Define the function
def func(x):
    return 3 * x**4 - 4 * x**3 + 1

# Generate x values
x_values = np.linspace(-0.5, 0.5, 100)

# Calculate corresponding y values
y_values = func(x_values)
y_values = np.array(y_values)

# Plot the function
plt.plot(x_values, y_values, label=r'$y = 3x^4 - 4x^3 + 1$')
# Point of inflection
plt.scatter(0.03535353535353536, 0.9998279369024238, color='red', marker='o')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of the function $y = 3x^4 - 4x^3 + 1$')
plt.legend()

# Display and save the plot
plt.savefig('partb_2.png')
plt.show()
