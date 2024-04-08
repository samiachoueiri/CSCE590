import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

def f(x, y, z):
    return 5*x**2 + y**2 + 2*z**2 + 4*x*y - 14*x - 6*y + 20

def gradient_f(x, y, z):
    df_dx = 10*x + 4*y - 14
    df_dy = 2*y + 4*x - 6
    df_dz = 4*z
    return np.array([df_dx, df_dy, df_dz])

def hessian_f(x, y, z):
    return np.array([[10, 4, 0], [4, 2, 0], [0, 0, 4]])

# Implement Newton-Raphson algorithm
def newton_raphson(initial_guess, data, epsilon=1e-6, max_iterations=100):
    x,y,z = initial_guess
    i = 0
    gradient = gradient_f(x,y,z)
    update = 0
    data.append ([i,x,y,z,gradient,update,f(x,y,z)])
    while i < max_iterations:
        i += 1
        gradient = gradient_f(x,y,z)
        hessian_inv = np.linalg.inv(hessian_f(x,y,z))
        update = np.dot(hessian_inv, gradient)
        [x,y,z] = [x,y,z] - update
        data.append ([i,x,y,z,gradient,update,f(x,y,z)])
        # print(i,x,y,z,gradient,update,f(x,y,z),'---------',np.linalg.norm(update),'<????',epsilon)
        if np.linalg.norm(update) < epsilon:
            break
    return [x,y,z]

data = [['i','x','y','z','gradient','update','f(x,y,z)']]
i = 1
initial_guess = [100, 100, 100] # Initial guess

# Find the minimizer using Newton-Raphson algorithm
minimizer = newton_raphson(initial_guess,data)
print("Minimizer found by Newton-Raphson algorithm:", minimizer)
print("Minimum value of f(x) at the minimizer:", f(*minimizer))

# Writing to the CSV file
with open('resultsB1.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)

# Lists to store i and cost
i_values = []
c_values = []
x_values = []
y_values = []
z_values = []


# Read the CSV file and extract data
with open('resultsB1.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        i_values.append(float(row['i']))
        c_values.append(float(row['f(x,y,z)']))
        x_values.append(float(row['x']))
        y_values.append(float(row['y']))
        z_values.append(float(row['z']))


# Plot the values
plt.figure(figsize=(8, 6))
plt.plot(i_values, c_values, color='blue', linestyle='-')
plt.xlabel('Iterations')
plt.ylabel('cost')
plt.title('Plot of cost vs iterations - bisection rule')
plt.legend()
plt.grid(True)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y, 0) 
ax.contour(X, Y, Z, levels=60, linestyles="solid", alpha=0.9, antialiased=True) 
ax.plot(x_values, y_values, z_values, marker='o', linestyle='-', color='black', label='Solution Sequence')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(0, 100)
ax.set_title('Cost Contours and Solution Sequence')
cbar = plt.colorbar(ax.contour(X, Y, Z, levels=60, alpha=0.9), ax=ax)
cbar.set_label('Cost')
plt.show()

