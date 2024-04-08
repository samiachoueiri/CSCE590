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

def alpha_bisection (x,y,z,d,a1,a2):
    arr = np.array([x,y,z])
    d = np.array(d)
    alpha_min = a1
    alpha_max = a2
    alpha_mid = (alpha_min + alpha_max)/2
    
    imp = arr+alpha_mid*d
    h = np.dot(np.transpose(gradient_f(imp[0], imp[1], imp[2])),d)
    if (abs(h) < 10**(-6)) or (alpha_min == alpha_max):
        return np.array([alpha_mid])
    elif h> 0:
        return alpha_bisection(x,y,z,d,alpha_min, alpha_mid)
    else:
        return alpha_bisection(x,y,z,d,alpha_mid, alpha_max)



def steepest_descent(i,initial_point,epsilon,data):
    x, y, z = initial_point
    gradient = gradient_f(x, y, z)
    mag = math.sqrt(gradient[0]**2+gradient[1]**2+gradient[2]**2)
    d = [-gradient[0],-gradient[1],-gradient[2]]
    alpha = alpha_bisection (x,y,z,d,a1,a2)[0]

    data.append([i,x,y,z,gradient[0],gradient[1],gradient[2],mag,alpha,f(x,y,z)])
    # print(i,x,y,z,gradient,mag,alpha,f(x,y,z))

    while True:
        i +=1
        gradient = gradient_f(x, y, z)
        mag = math.sqrt(gradient[0]**2+gradient[1]**2+gradient[2]**2)
        d = [-gradient[0],-gradient[1],-gradient[2]]
        alpha = alpha_bisection (x,y,z,d,a1,a2)[0]

        if np.linalg.norm(gradient) < epsilon:
                break

        x -= alpha * gradient[0]
        y -= alpha * gradient[1]
        z -= alpha * gradient[2]

        data.append([i,x,y,z,gradient[0],gradient[1],gradient[2],mag,alpha,f(x,y,z)])
        # print(i,x,y,z,gradient,mag,alpha,f(x,y,z))
    return x, y, z



data = [['i','x','y','z','direction x','direction y','direction z','||direction||','alpha','f(x,y,z)']]
i = 1
initial_point = [0, 10, 5]  # Initial guess
epsilon = 0.01  # Convergence threshold
a1 = 0
a2 = 1

min_point = steepest_descent(i,initial_point, epsilon,data)
print("Minimum point:", min_point)
print("f(x, y) evaluated at minimum point:", f(*min_point))

# Writing to the CSV file
with open('resultsA1.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)

# Lists to store i and cost
i_values = []
c_values = []
x_values = []
y_values = []
z_values = []


# Read the CSV file and extract data
with open('resultsA1.csv', 'r') as file:
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

