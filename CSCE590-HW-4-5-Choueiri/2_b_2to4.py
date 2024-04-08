import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

#k = 2
coef2 = [-3,1,1]
cost2 = 80

#k = 3
coef3 = [-3.,8.08333333,1.,-2.08333333]
cost3 = 17.5

#k = 4
coef4 = [-2.71368151e-28,8.08333333e+00,-5.45833333e+00,-2.08333333e+00,1.45833333e+00]
cost4 = 2.529285277364869e-29


fig, ax = plt.subplots()
t_true = np.array([-2, -1, 0, 1, 2])
y_true = np.array([2, -10, 0, 2, 1])
ax.scatter(t_true, y_true, color='red', linestyle='-')

def p2(t, x):
    x0, x1, x2 = x
    return x0 + x1 * t + x2 * t**2
t = np.linspace(-2, 2, 100)
y2 = p2(t, coef2)
ax.plot(t, y2, color='blue', linestyle='-',label='k=2')

def p3(t, x):
    x0, x1, x2, x3 = x
    return x0 + x1 * t + x2 * t**2 + x3 * t**3
y3 = p3(t, coef3)
ax.plot(t, y3, color='green', linestyle='-',label='k=3')

def p4(t, x):
    x0, x1, x2, x3, x4 = x
    return x0 + x1 * t + x2 * t**2 + x3 * t**3 + x4 * t**4
y4 = p4(t, coef4)
ax.plot(t, y4, color='orange', linestyle='-',label='k=4')

plt.xlabel('t')
plt.ylabel('y')
plt.title('Best fit function')
ax.legend()
plt.show()








kay = [2,3,4]
error = [cost2,cost3,cost4]

print(kay)
print(error)
plt.plot(kay, error, color='blue', linestyle='-')
plt.xlabel('k')
plt.ylabel('error')
plt.title('k vs error')
plt.xticks(kay)
plt.show()