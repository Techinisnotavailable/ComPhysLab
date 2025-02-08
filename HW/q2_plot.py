import numpy as np
import matplotlib.pyplot as plt

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

from scipy.optimize import fsolve

def displacement(F_val):
    # Define the equation F = -kx - k'x^3
    return fsolve(lambda x: force(x) - F_val, 0)[0]  # Initial guess = 0

# Compute corresponding displacements for each force value
x_values = np.array([displacement(F_val) for F_val in F])

# Plot the Force-Extension curve
plt.plot(np.abs(F), x_values, marker='o')
plt.xlabel('|F| (N)')
plt.ylabel('x (mm)')
plt.title('Force-Extension Curve of Nonlinear Spring')
plt.grid(True)
plt.show()