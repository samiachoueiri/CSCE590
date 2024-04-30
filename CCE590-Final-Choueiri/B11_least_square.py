import numpy as np

# Example matrix A (m x n)
A = np.array([[1, 1, 1], [0, 0, 1], [1, 0, 1], [2, 0, 5], [-7, 8, 0], [1, 2, -1]])

# Example vector b (m x 1)
b = np.array([3, 1, 2, 8, 0, 1])

# Compute (AT A)
ATA = np.dot(A.T, A)

# Compute (AT A)^{-1}
ATA_inv = np.linalg.inv(ATA)

# Compute AT
AT = A.T

# Compute x* = (AT A)^{-1} AT b
x_star = np.dot(ATA_inv, np.dot(AT, b))

print("x*:", x_star)
