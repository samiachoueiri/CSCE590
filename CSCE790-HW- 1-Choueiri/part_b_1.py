import matplotlib.pyplot as plt
import numpy as np
import math

# Create Plot for Exercise 3 figure
plt.figure(figsize=(8, 6))

# Plot a vertical line at x1=3/2
plt.axvline(x=1.5, label='x1=3/2', color='green')
# Plot line x1+x2=2
plt.plot([0, 2], [2, 0], label='x1+x2=2', color='orange')
# Plot line x1+(8/3)x2=4
plt.plot([0, 4], [(12/8), 0], label='x1+(8/3)x2=4', color='purple')
# Plot optimal solution 2x1+x2=3.5
# plt.plot([0, 3.5], [1.75, 0], label='optimal solution', color='blue', linestyle='--')
plt.plot([0, 1.75], [3.5, 0], label='optimal solution', color='blue', linestyle='--')
plt.annotate('Minimizer\n(1.5, 0.5)', xy = (1.55, 0.5))

# Define the boundaries of the feasible area in the plot
x1_vert = [0, 0.8, 1.5, 1.5, 0]
x2_vert = [1.5, 1.2, 0.5, 0, 0]
plt.fill(x1_vert, x2_vert, 'grey')
plt.text(0.3, 0.5, 'Feasible Region', size = '11')

# Customize the plot
plt.xlabel('x1-axis')
plt.ylabel('x2-axis')
plt.title('Exercise 3')
plt.legend()

# Set x-axis and y-axis values and limits
plt.xticks(np.arange(0, 3.1, 0.5))  # Set x-axis values
plt.yticks(np.arange(0, 3.1, 0.5))  # Set y-axis values
plt.xlim(0, 3)  # Set x-axis limits
plt.ylim(0, 3)  # Set y-axis limits
plt.savefig('ex3.png')
#################################################################################
# Create Plot for Exercise 7-a figure
plt.figure(figsize=(8, 8))

# Plot circle (x+8)^2+(y-9)^2=49
h = 8  # x-coordinate of the center
k = 9  # y-coordinate of the center
r = math.sqrt(49)  # radius
theta = np.linspace(0, 2*np.pi, 100)
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, label='Circle: $(x-8)^2 + (y-9)^2 = 7^2$')
# Plot a vertical line at x=2
plt.axvline(x=2, label='x=2', color='green')
# Plot a vertical line at x=13
plt.axvline(x=13, label='x=13', color='green')
# Plot line x1+x2=24
plt.plot([0, 24], [24, 0], label='x+y=24', color='purple')

# Solution Corner
plt.scatter(16, 14, color='black', marker='o')
plt.annotate('(16,14)', xy = (15.4,14.2))
h = 16  # x-coordinate of the center
k = 14  # y-coordinate of the center
theta = np.linspace(0, 2*np.pi, 100)
r = 1.5  # radius
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, color='black')
r = 4.2  # radius
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, color='black')
r = 7  # radius
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, color='black')

# Optimal Solution
plt.scatter(13, 11, color='red', marker='o')
plt.annotate('Solution\n(13, 11)', xy = (13,11))
plt.text(18, 23, 'Solution Corner Point', size = '11')

# Define the boundaries of the feasible area in the plot
p1x= 2
p1y= 5.4
p2x= 2
p2y= 12.6
c1x= [2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5]
c1y= [12.778,13.417,13.949,14.381,14.814,15.102,15.335,15.567,15.711,15.85,15.92,15.98]
pos1= 2
p3x= 8
p3y= 16
p4x= 13
p4y= 11
p5x= 13
p5y= 4.1
c2x= [1.952,2.134,2.617,3.019,3.382,3.784,4.187,4.67,5.193,5.676,6.159,6.763,7.528,8.092,8.736,9.3,9.903,10.548,11.151,11.795,12.46]
c2x.reverse()
c2y= [5.383,5.139,4.527,4.038,3.711,3.426,3.14,2.855,2.569,2.406,2.243,2.08,1.998,1.998,2.039,2.121,2.284,2.488,2.732,3.14,3.589]
c2y.reverse()
pos2= 17
x_vert = [p1x, p2x, p3x, p4x, p5x]
for i in range(len(c1x)):
	x_vert.insert(i + pos1, c1x[i])
for i in range(len(c2x)):
	x_vert.insert(i + pos2, c2x[i])
y_vert = [p1y, p2y, p3y, p4y, p5y]
for i in range(len(c1y)):
	y_vert.insert(i + pos1, c1y[i])
for i in range(len(c2y)):
	y_vert.insert(i + pos2, c2y[i])
plt.fill(x_vert, y_vert, 'grey')
plt.text(5, 7, 'Feasible Region', size = '11')

# Customize the plot
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Exercise 7 (a,b)=(16,14)')
plt.legend()

# Set x-axis and y-axis values and limits
plt.xticks(np.arange(0, 25.1, 1))  # Set x-axis values
plt.yticks(np.arange(0, 25.1, 1))  # Set y-axis values
plt.xlim(0, 25)  # Set x-axis limits
plt.ylim(0, 25)  # Set y-axis limits

plt.savefig('ex7_a.png')
#################################################################################
# Create Plot for Exercise 7-b figure
plt.figure(figsize=(8, 8))

# Plot circle (x+8)^2+(y-9)^2=49
h = 8  # x-coordinate of the center
k = 9  # y-coordinate of the center
r = math.sqrt(49)  # radius
theta = np.linspace(0, 2*np.pi, 100)
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, label='Circle: $(x-8)^2 + (y-9)^2 = 7^2$')
# Plot a vertical line at x=2
plt.axvline(x=2, label='x=2', color='green')
# Plot a vertical line at x=13
plt.axvline(x=13, label='x=13', color='green')
# Plot line x1+x2=24
plt.plot([0, 24], [24, 0], label='x+y=24', color='purple')

# Solution interior
h = 11  # x-coordinate of the center
k = 10  # y-coordinate of the center
theta = np.linspace(0, 2*np.pi, 100)
r = 1  # radius
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, color='black')
r = 2  # radius
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, color='black')
r = 3  # radius
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, color='black')

# Define the boundaries of the feasible area in the plot
p1x= 2
p1y= 5.4
p2x= 2
p2y= 12.6
c1x= [2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5]
c1y= [12.778,13.417,13.949,14.381,14.814,15.102,15.335,15.567,15.711,15.85,15.92,15.98]
pos1= 2
p3x= 8
p3y= 16
p4x= 13
p4y= 11
p5x= 13
p5y= 4.1
c2x= [1.952,2.134,2.617,3.019,3.382,3.784,4.187,4.67,5.193,5.676,6.159,6.763,7.528,8.092,8.736,9.3,9.903,10.548,11.151,11.795,12.46]
c2x.reverse()
c2y= [5.383,5.139,4.527,4.038,3.711,3.426,3.14,2.855,2.569,2.406,2.243,2.08,1.998,1.998,2.039,2.121,2.284,2.488,2.732,3.14,3.589]
c2y.reverse()
pos2= 17
x_vert = [p1x, p2x, p3x, p4x, p5x]
for i in range(len(c1x)):
	x_vert.insert(i + pos1, c1x[i])
for i in range(len(c2x)):
	x_vert.insert(i + pos2, c2x[i])
y_vert = [p1y, p2y, p3y, p4y, p5y]
for i in range(len(c1y)):
	y_vert.insert(i + pos1, c1y[i])
for i in range(len(c2y)):
	y_vert.insert(i + pos2, c2y[i])
plt.fill(x_vert, y_vert, 'grey')
plt.text(5.5, 3, 'Feasible Region', size = '11')

# Optimal Solution
plt.scatter(11, 10, color='red', marker='o')
plt.annotate('Solution\n(11, 10)', xy = (17,10))
arrow_properties = dict(facecolor='red', edgecolor='red', arrowstyle='->', shrinkA=0, lw=2)
plt.annotate('', xy=(11.3, 10), xytext=(16.7, 10), arrowprops=arrow_properties)
plt.text(16, 18, 'Solution in Interior', size = '11')

# Customize the plot
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Exercise 7 (a,b)=(11,10)')
plt.legend()

# Set x-axis and y-axis values and limits
plt.xticks(np.arange(0, 25.1, 1))  # Set x-axis values
plt.yticks(np.arange(0, 25.1, 1))  # Set y-axis values
plt.xlim(0, 25)  # Set x-axis limits
plt.ylim(0, 25)  # Set y-axis limits

plt.savefig('ex7_b.png')
#################################################################################
# Create Plot for Exercise 7-c figure
plt.figure(figsize=(8, 8))

# Plot circle (x+8)^2+(y-9)^2=49
h = 8  # x-coordinate of the center
k = 9  # y-coordinate of the center
r = math.sqrt(49)  # radius
theta = np.linspace(0, 2*np.pi, 100)
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, label='Circle: $(x-8)^2 + (y-9)^2 = 7^2$')
# Plot a vertical line at x=2
plt.axvline(x=2, label='x=2', color='green')
# Plot a vertical line at x=13
plt.axvline(x=13, label='x=13', color='green')
# Plot line x1+x2=24
plt.plot([0, 24], [24, 0], label='x+y=24', color='purple')

# Solution on boundary
plt.scatter(14, 14, color='black', marker='o')
plt.annotate('(14,14)', xy = (13.3,14.3))
h = 14  # x-coordinate of the center
k = 14  # y-coordinate of the center
theta = np.linspace(0, 2*np.pi, 100)
r = 2  # radius
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, color='black')
r = 2.8  # radius
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, color='black')
r = 4  # radius
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, color='black')
r = 5  # radius
x = h + r * np.cos(theta)
y = k + r * np.sin(theta)
plt.plot(x, y, color='black')

# Define the boundaries of the feasible area in the plot
p1x= 2
p1y= 5.4
p2x= 2
p2y= 12.6
c1x= [2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5]
c1y= [12.778,13.417,13.949,14.381,14.814,15.102,15.335,15.567,15.711,15.85,15.92,15.98]
pos1= 2
p3x= 8
p3y= 16
p4x= 13
p4y= 11
p5x= 13
p5y= 4.1
c2x= [1.952,2.134,2.617,3.019,3.382,3.784,4.187,4.67,5.193,5.676,6.159,6.763,7.528,8.092,8.736,9.3,9.903,10.548,11.151,11.795,12.46]
c2x.reverse()
c2y= [5.383,5.139,4.527,4.038,3.711,3.426,3.14,2.855,2.569,2.406,2.243,2.08,1.998,1.998,2.039,2.121,2.284,2.488,2.732,3.14,3.589]
c2y.reverse()
pos2= 17
x_vert = [p1x, p2x, p3x, p4x, p5x]
for i in range(len(c1x)):
	x_vert.insert(i + pos1, c1x[i])
for i in range(len(c2x)):
	x_vert.insert(i + pos2, c2x[i])
y_vert = [p1y, p2y, p3y, p4y, p5y]
for i in range(len(c1y)):
	y_vert.insert(i + pos1, c1y[i])
for i in range(len(c2y)):
	y_vert.insert(i + pos2, c2y[i])
plt.fill(x_vert, y_vert, 'grey')
plt.text(5.5, 3, 'Feasible Region', size = '11')

# Optimal Solution
plt.scatter(12, 12, color='red', marker='o')
plt.annotate('Solution\n(12, 12)', xy = (17,8))
arrow_properties = dict(facecolor='red', edgecolor='red', arrowstyle='->', shrinkA=0, lw=2)
plt.annotate('', xy=(12.3, 12), xytext=(16.7, 9), arrowprops=arrow_properties)
plt.text(18, 18, 'Solution on Boundary', size = '11')

# Customize the plot
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Exercise 7 (a,b)=(12,12)')
plt.legend()

# Set x-axis and y-axis values and limits
plt.xticks(np.arange(0, 25.1, 1))  # Set x-axis values
plt.yticks(np.arange(0, 25.1, 1))  # Set y-axis values
plt.xlim(0, 25)  # Set x-axis limits
plt.ylim(0, 25)  # Set y-axis limits

plt.savefig('ex7_c.png')

# Show the plot
# plt.show()

