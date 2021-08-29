import numpy as np


def gauss_seidel(A, b, eps=1e-10):
    for element in np.diagonal(A):
        if element == 0:
            raise Exception("Diagonal equals zero")

    if np.linalg.norm(A, ord=np.inf) < 1:
        A = np.matmul(np.transpose(A), A)
        b = np.matmul(np.transpose(b), b)

    N = np.size(A, 0)
    alpha = np.zeros_like(A)
    beta = np.zeros_like(b)

    for i in range(N):
        beta[i] = b[i] / A[i][i]

    for i in range(N):
        for j in range(N):
            if i != j:
                alpha[i][j] = -A[i][j] / A[i][i]
            else:
                alpha[i][j] = 0

    x = np.zeros_like(b, dtype=np.double)
    output_file = open("output.txt", "a")
    output_file.write(str(alpha))
    output_file.write("\n\n")
    output_file.write(str(beta))
    output_file.write("\n\n")

    while True:
        x_old = x.copy()
        mult = np.matmul(alpha, x)
        x = np.add(beta, mult)

        if np.linalg.norm(x - x_old, ord=np.inf) < (1 - np.linalg.norm(alpha, ord=np.inf)) / np.linalg.norm(alpha, ord=np.inf) * eps:
            break

        output_file.write(str(x))
        output_file.write("\n")

    return x

def gauss_seidel_inv(A, eps=1e-10):
    N = np.size(A, 0)
    E = np.identity(N)
    inverse = np.empty_like(A)
    for i in range(N):
        inverse[:, i] = gauss_seidel(A, E[:, i])

    return inverse
