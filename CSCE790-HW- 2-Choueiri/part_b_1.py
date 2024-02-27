import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Define the function
def cost_function(x1, x2):
    return (x1 - 3)**4 + (x2 - 3)**2
def c1_function(x1, x2):
    return 4*x1**2 + 9*x2**2
def c2_function(x1, x2):
    return x1**2 + 3*x2

# Generate data
x1 = np.linspace(-1, 10, 100)
x2 = np.linspace(-1, 10, 100)
# x1 = np.linspace(-5, 5, 100)
# x2 = np.linspace(-5, 5, 100)
x1, x2 = np.meshgrid(x1, x2)
z_cost = cost_function(x1, x2)
z_cost_constrained = np.where(z_cost <= 36, z_cost, np.nan)
z_c1 = c1_function(x1, x2)
z_c1_constrained = np.where(z_c1 <= 36, z_c1, np.nan)
z_c2 = c2_function(x1, x2)
z_c2_constrained = np.where(z_c2 <= 36, z_c2, np.nan)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x1, x2, z_cost, cmap='viridis')
ax.plot_surface(x1, x2, z_cost_constrained, cmap='viridis', alpha=0.8, label='(x1-3)^2 + (x2-3)^2')
ax.plot_surface(x1, x2, z_c1_constrained, cmap='Dark2', alpha=0.8, label='4(x1)^2 + 9(x2)^2 <= 36')
ax.plot_surface(x1, x2, z_c2_constrained, color='orange', alpha=0.5, label='x1^2 + 3x2 = 3')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Z')
plt.title('min (x1-3)^4 + (x2-3)^2')
# fig.colorbar(surf1, shrink=0.5, aspect=5)
ax.legend()

fig = plt.figure()
# plt.style.use(["science", "notebook", "grid"])
ax = plt.axes()
# Data for a three-dimensional line
xline = np.linspace(-3, 3, 1000)
yline = [ (3 - math.sqrt(i**2))/3 for i in xline]
plt.plot(xline,yline)
yline = [ (math.sqrt(36 -  4*i**2))/9 for i in xline]
yline2 = [ -(math.sqrt(36 -  4*i**2))/9 for i in xline]
plt.title('Feasible Region')
plt.plot(xline,yline,color="orange")
plt.plot(xline,yline2,color="orange")
plt.plot([-3, -1.1544], [0, 0.6150], color ='red')
plt.plot([3, 1.1544], [0, 0.6150], color ='red')
plt.show()
# graphically x1=4.49 x2=6.59 z=2.99


