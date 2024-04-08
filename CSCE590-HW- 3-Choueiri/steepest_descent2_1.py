import numpy as np
import math
import matplotlib.pyplot as plt
import csv

def f(x, Q, c):
    return 0.5 * np.dot(np.dot(x.T, Q), x) - np.dot(c.T, x) + 10

def gradient_f(x, Q, c):
    return np.dot(Q, x) - c

def steepest_descent(i,Q, c, initial_point, alpha, epsilon,data):
    x = initial_point
    gradient = gradient_f(x, Q, c)
    mag = math.sqrt(gradient[0]**2+gradient[1]**2)
    # print(i,x,Q,c,gradient[0],gradient[1],mag,alpha,f(x, Q, c))
    data.append([i,x,Q,c,gradient[0],gradient[1],mag,alpha,f(x, Q, c)])
    x_arr.append(x[0])
    y_arr.append(x[1])
    while True:
        i+=1
        gradient = gradient_f(x, Q, c)
        mag = math.sqrt(gradient[0]**2+gradient[1]**2)
        if np.linalg.norm(gradient) < epsilon:
            break
        x = x - (alpha * gradient)
        # print(i,x,Q,c,gradient[0],gradient[1],mag,alpha,f(x, Q, c))
        data.append([i,x,Q,c,gradient[0],gradient[1],mag,alpha,f(x, Q, c)])
        x_arr.append(x[0])
        y_arr.append(x[1])
    return x

i = 1
x_arr = []
y_arr = []
data = [['i','x','Q','c','direction x','direction y','||direction||','alpha','f(x,y)']]
Q = np.array([[20, 5], [5, 2]]) 
c = np.array([14, 16]) 
initial_point = np.array([40, -100])  # Initial guess
alpha = 0.01  # Step size
epsilon = 1e-6  # Convergence threshold
min_point = steepest_descent(i,Q, c, initial_point, alpha, epsilon,data)
print("Minimum point:", min_point)
print("Minimum value of f(x):", f(min_point, Q, c))

# # Writing to the CSV file
# with open('resultsA2_1.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerows(data)

X = np.linspace(-10, 10, 400)
Y = np.linspace(-10, 10, 400)
point = np.array([X, Y])
Z = f(point, Q, c)

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
# ax2.plot(x_arr,y_arr,color='black')
plt.colorbar(contour, ax=ax2, label='f(X, Y)')

# Show the plots
# plt.tight_layout()
plt.show()