import numpy as np
from scipy.optimize import bisect, newton, root_scalar, fsolve

# Problem 1: sin(x) - cos(x) = 0.5
def f1(x):
    return np.sin(x) - np.cos(x) - 0.5

# Problem 2: (ln(x))^3 = 0.5
def f2(x):
    return (np.log(x))**3 - 0.5

# Derivative for Newton-Raphson (Problem 1)
def df1(x):
    return np.cos(x) + np.sin(x)

# Derivative for Newton-Raphson (Problem 2)
def df2(x):
    return 3 * (np.log(x))**2 / x

# Function to compare methods
def compare_methods(func, dfunc, interval, initial_guess, problem_name):
    print(f"\n--- {problem_name} ---")

    # Check if Bisection method conditions are met
    if func(interval[0]) * func(interval[1]) > 0:
        print(f"Bisection Method: Cannot apply as f({interval[0]}) and f({interval[1]}) do not have opposite signs.")
    else:
        # Bisection Method
        bisect_result = root_scalar(func, method='bisect', bracket=interval, xtol=1e-6)
        print(f"Bisection Method: Root = {bisect_result.root}, Iterations = {bisect_result.iterations}")

    # Secant Method
    secant_result = root_scalar(func, method='secant', x0=initial_guess[0], x1=initial_guess[1], xtol=1e-6)
    print(f"Secant Method: Root = {secant_result.root}, Iterations = {secant_result.iterations}")

    # Newton-Raphson Method
    newton_result = newton(func, initial_guess[0], fprime=dfunc, tol=1e-6, full_output=True)
    print(f"Newton-Raphson Method: Root = {newton_result[0]}, Iterations = {newton_result[1].iterations}")

# Print name and ID
print('Techin Saetang 6634219123')


# Try adjusting the interval and initial guess for Problem 1
compare_methods(f1, df1, interval=(0, 2), initial_guess=(1, 2), problem_name="Problem 1.1: sin(x) - cos(x) = 0.5")

# For Problem 2, adjust the interval for Bisection method as well
compare_methods(f2, df2, interval=(2, 5), initial_guess=(3, 4), problem_name="Problem 1.9: (ln(x))^3 = 0.5")

############### Q2 ##########################

# Spring constants (in N/mm)
k = 1.0  # N/mm
k_prime = 0.8  # N/mm^3

# Define the force function for the spring
def force(x):
    return -k * x - k_prime * x**3

# Generate an array of forces from 0 to 1 N
F = np.linspace(0, 1, 10)

# Find corresponding displacements (x) for each force using the equation F = -kx - k'x^3
# We solve for x from F = kx + k'x^3. We'll use a numerical solver.

def displacement(F_val):
    # Define the equation F = -kx - k'x^3
    return fsolve(lambda x: force(x) - F_val, 0)[0]  # Initial guess = 0

# Compute corresponding displacements for each force value
x_values = np.array([displacement(F_val) for F_val in F])

# Print the Force-Extension array
force_extension_array = np.column_stack((np.abs(F), x_values))
print('\n')
print('Problem 2')
print("Force-Extension Array (|F|, x):")
print(force_extension_array)