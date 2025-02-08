import numpy as np
from scipy.optimize import bisect, newton, root_scalar

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
compare_methods(f1, df1, interval=(0, 2), initial_guess=(1, 2), problem_name="Problem 1: sin(x) - cos(x) = 0.5")

# For Problem 2, adjust the interval for Bisection method as well
compare_methods(f2, df2, interval=(2, 5), initial_guess=(3, 4), problem_name="Problem 2: (ln(x))^3 = 0.5")