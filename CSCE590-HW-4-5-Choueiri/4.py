import numpy as np
import matplotlib.pyplot as plt

# Given matrix A
A = np.array([[1, -1, 0],
              [1, 1, 2],
              [1, -1, 0],
              [1, 1, 2]])

# (a) Gram-Schmidt Process

# Step 1
u1 = A[:, 0]

# Step 2
u2 = A[:, 1] - (np.dot(u1.T, A[:, 1]) / np.dot(u1.T, u1)) * u1

# Step 3
u3 = A[:, 2] - (np.dot(u1.T, A[:, 2]) / np.dot(u1.T, u1)) * u1 - \
     (np.dot(u2.T, A[:, 2]) / np.dot(u2.T, u2)) * u2

# Step 4
v1 = u1 / np.linalg.norm(u1)
v2 = u2 / np.linalg.norm(u2)
v3 = u3 / np.linalg.norm(u3)

# Calculate the range of A
range_A = np.column_stack((v1, v2, v3))

print("Range of matrix A:")
print(range_A)

# (b) Plotting the column vectors of A

# Plotting the column vectors
for i in range(A.shape[1]):
    plt.plot([0, A[0, i]], [0, A[1, i]], 'b' if i == 0 else 'r' if i == 1 else 'g')

# Plotting the orthonormal vectors
plt.plot([0, v1[0]], [0, v1[1]], 'b--')
plt.plot([0, v1[2]], [0, v1[3]], 'b--')
plt.plot([0, v2[0]], [0, v2[1]], 'r--')
plt.plot([0, v2[2]], [0, v2[3]], 'r--')
plt.plot([0, v3[0]], [0, v3[1]], 'g--')
plt.plot([0, v3[2]], [0, v3[3]], 'g--')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Column Vectors of A and Orthonormal Vectors from Gram-Schmidt')
plt.axis('equal')
plt.show()
