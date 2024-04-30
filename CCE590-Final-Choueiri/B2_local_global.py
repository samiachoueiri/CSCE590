import sympy as sp

# Define the variable and the function
x = sp.symbols('x')
f = 2*x - x**2

# 1. Find critical points
f_prime = sp.diff(f, x)  # First derivative
critical_points = sp.solve(f_prime, x)

print("Critical Points:", critical_points)

# 2. Determine if these critical points are minima or maxima
f_double_prime = sp.diff(f_prime, x)  # Second derivative
for point in critical_points:
    second_derivative_value = f_double_prime.subs(x, point)
    if second_derivative_value > 0:
        print(f"Point {point} is a local minimum")
    elif second_derivative_value < 0:
        print(f"Point {point} is a local maximum")
    else:
        print(f"Point {point} is a saddle point")

# 3. Check the global optima by evaluating at the endpoints
endpoints = [0, 4]  # Given in the problem statement
endpoint_values = {point: f.subs(x, point) for point in endpoints}

# Compare critical points with endpoint values
critical_point_values = {point: f.subs(x, point) for point in critical_points}
optimal_points = {**critical_point_values, **endpoint_values}

# Find the global maximum and minimum
global_max = max(optimal_points, key=optimal_points.get)
global_min = min(optimal_points, key=optimal_points.get)

print("Global maximum at x =", global_max, "with value =", optimal_points[global_max])
print("Global minimum at x =", global_min, "with value =", optimal_points[global_min])

def plot_function_and_critical_points(f, x_symbol, critical_points, x_range=(0, 4), n_samples=100):
    import numpy as np
    import matplotlib.pyplot as plt
    import sympy as sp

    # Create a numpy function for plotting
    f_np = sp.lambdify(x_symbol, f, 'numpy')

    # Generate data for plotting
    x_vals = np.linspace(x_range[0], x_range[1], n_samples)  # Create a range of x-values
    y_vals = f_np(x_vals)  # Calculate the corresponding y-values

    # Get the critical points' x and y values
    critical_x_vals = [cp.evalf() for cp in critical_points]
    critical_y_vals = [f_np(cp) for cp in critical_x_vals]

    # Plot the function
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='2x - x^2', color='blue')

    # Plot the critical points
    plt.scatter(critical_x_vals, critical_y_vals, color='red', zorder=5, label='Critical Points')

    # Annotate critical points
    for x_val, y_val in zip(critical_x_vals, critical_y_vals):
        plt.annotate(f'({x_val:.2f}, {y_val:.2f})', (x_val, y_val), textcoords="offset points", xytext=(0, 10), ha='center')

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function Plot with Critical Points')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # x-axis
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # y-axis
    plt.legend()
    plt.grid(True)
    plt.show()

# Define the variable and function
x = sp.symbols('x')
f = 2*x - x**2

# Find the critical points
f_prime = sp.diff(f, x)
critical_points = sp.solve(f_prime, x)

# Call the plotting function
plot_function_and_critical_points(f, x, critical_points)
