import numpy as np
import pickle


def forward_sub(L, b):
    y = np.zeros_like(b)

    for i in range(len(y)):
      y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


def backward_sub(U, y):
    x = np.zeros_like(y)

    for i in range(len(x), 0, -1):
      x[i-1] = (y[i-1] - np.dot(U[i-1, i:], x[i:])) / U[i-1, i-1]

    return x


def lu_factor(A):
  L = np.zeros_like(A)
  U = np.zeros_like(A)

  N = np.size(A,0)

  for k in range(N):
    L[k, k] = 1
    U[k, k] = (A[k, k] - np.dot(L[k, :k], U[:k, k]))
    for j in range(k+1, N):
      U[k, j] = (A[k, j] - np.dot(L[k, :k], U[:k, j]))
    for i in range(k+1, N):
      L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]
  return (L, U)


def lu_solve(L, U, b):
    y = forward_sub(L,b)
    x = backward_sub(U,y)

    return x


def lu_decompose(A, b):
    for element in np.diagonal(A):
        if element == 0:
            raise Exception("Diagonal equals zero")

    L, U = lu_factor(A)
    x = lu_solve(L, U, b)

    return L, U, x

def lu_det(A):
    N = np.size(A, 0)
    L, U = lu_factor(A)
    det = 1
    for i in range(N):
        det *= U[i][i]

    return L, U, det

def lu_inv(A):
    N = np.size(A, 0)
    E = np.identity(N)
    L, U = lu_factor(A)
    inverse = np.empty_like(A)
    for i in range(N):
        inverse[:, i] = lu_solve(L, U, E[:, i])

    return L, U, inverse
