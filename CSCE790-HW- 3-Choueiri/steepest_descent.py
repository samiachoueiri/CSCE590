import numpy as np
import math
import matplotlib.pyplot as plt
import csv

def f(x, y):
    return 5*x**2 + y**2 + 4*x*y - 14*x - 6*y + 20

def gradient_f(x, y):
    df_dx = 10*x + 4*y - 14
    df_dy = 2*y + 4*x - 6
    return np.array([df_dx, df_dy])

def steepest_descent(i,initial_point, alpha, epsilon,data,x_arr,y_arr):
    x, y = initial_point
    gradient = gradient_f(x, y)
    mag = math.sqrt(gradient[0]**2+gradient[1]**2)
    # print(i,x,y,gradient[0],gradient[1],mag,alpha,f(x,y))
    data.append([i,x,y,gradient[0],gradient[1],mag,alpha,f(x,y)])
    x_arr.append(x)
    y_arr.append(y)

    while True:
        i +=1
        if alpha == 0.0866:
            alpha = 2.1800
        else:
            alpha = 0.0866
        gradient = gradient_f(x, y)
        mag = math.sqrt(gradient[0]**2+gradient[1]**2)
        if np.linalg.norm(gradient) < epsilon:
            break
        x -= alpha * gradient[0]
        y -= alpha * gradient[1]
        # print(i,x,y,gradient[0],gradient[1],mag,alpha,f(x,y))
        data.append([i,x,y,gradient[0],gradient[1],mag,alpha,f(x,y)])
        x_arr.append(x)
        y_arr.append(y)
    return x, y

i = 1
x_arr = []
y_arr = []
data = [['i','x','y','direction x','direction y','||direction||','alpha','f(x,y)']]
initial_point = [0, 10]  # Initial guess
alpha = 0.00866  # Step size
epsilon = 1e-6  # Convergence threshold
min_point = steepest_descent(i,initial_point, alpha, epsilon,data,x_arr,y_arr)
print("Minimum point:", min_point)
print("f(x, y) evaluated at minimum point:", f(*min_point))

# Writing to the CSV file
with open('resultsA1.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)

# print('x array',x_arr,'size',len(x_arr))
# print('y array',y_arr,'size',len(y_arr))
# print(data)

# Generate x, y values
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Plot the function
fig = plt.figure(figsize=(12, 6))

# Surface plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(X, Y)')
ax1.set_title('Surface plot of f(X, Y)')

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(X, Y, Z, levels=50, cmap='coolwarm')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour plot of f(X, Y)')
ax2.plot(x_arr,y_arr,color='black')
plt.colorbar(contour, ax=ax2, label='f(X, Y)')

# Show the plots
# plt.tight_layout()
plt.show()