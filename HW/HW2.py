import numpy as np
from scipy.linalg import lu, cholesky

def main():
    print("Name: Techin Saetang")
    print("ID: 663 42191 23")
    X, Y = 9, 1

    # Question 1: Finding z in matrix equation G * l = h
    G = np.array([[1, -1, 1, -1],
                  [-1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [-1, 1, 1, np.nan]])  
    l = np.array([0, -1, 1, 1])
    h = np.array([1, 1, 1, 0])
    
    G[-1, -1] = (h[-1] - np.dot(G[-1, :-1], l[:-1])) / l[-1]
    print("Value of z:", G[-1, -1])
    print("Check np.allclose:", np.allclose(G @ l, h))
    
    # Question 2: Solving A^2 B = I
    A = np.array([[20, Y, 1, -1],
                  [Y, 20, 1, 0],
                  [1, 1, 20, X],
                  [-1, 0, X, 20]])
    I = np.eye(4)
    A2 = np.dot(A, A)
    
    B_solve = np.linalg.solve(A2, I)
    B_inv = np.linalg.inv(A2)
    
    P, L, U = lu(A2)
    B_LU = np.linalg.solve(U, np.linalg.solve(L, I))
    
    try:
        L_chol = cholesky(A2)
        B_chol = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, I))
    except:
        B_chol = "Cholesky decomposition not possible"
    
    print("Matrix B (solve method):\n", B_solve)
    print("Matrix B (inverse method):\n", B_inv)
    print("Matrix B (LU method):\n", B_LU)
    print("Matrix B (Cholesky method):\n", B_chol)
    print("Check np.allclose:", np.allclose(A2 @ B_solve, I))
    
    # Question 3: Iterative Methods
    n = 10
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = X + 10
        if i < n - 1:
            A[i, i + 1] = A[i + 1, i] = -Y / 10
        if i < n - 2:
            A[i, i + 2] = A[i + 2, i] = 1 / 10
    e = np.ones(n)
    b = A @ e
    
    def jacobi(A, b, tol=1e-6, max_iter=1000):
        x = np.zeros_like(b)
        D = np.diag(A)
        R = A - np.diagflat(D)
        for _ in range(max_iter):
            x_new = (b - np.dot(R, x)) / D
            if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                return x_new
            x = x_new
        return x
    
    def gauss_seidel(A, b, tol=1e-6, max_iter=1000):
        x = np.zeros_like(b)
        for _ in range(max_iter):
            x_new = np.copy(x)
            for i in range(len(b)):
                x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
            if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                return x_new
            x = x_new
        return x
    
    def sor(A, b, omega, tol=1e-6, max_iter=1000):
        x = np.zeros_like(b)
        for _ in range(max_iter):
            x_new = np.copy(x)
            for i in range(len(b)):
                x_new[i] = (1 - omega) * x[i] + omega * (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
            if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                return x_new
            x = x_new
        return x
    
    x_jacobi = jacobi(A, b)
    x_gauss = gauss_seidel(A, b)
    omega_values = [0.8, 0.9, 1.0, 1.1, 1.2]
    x_sor = {w: sor(A, b, w) for w in omega_values}
    
    print("Jacobi Solution:", x_jacobi)
    print("Gauss-Seidel Solution:", x_gauss)
    for w, sol in x_sor.items():
        print(f"SOR Solution (omega={w}): {sol}")

if __name__ == "__main__":
    main()
