from LU_Decomposition import lu_decompose
from LU_Decomposition import lu_det
from LU_Decomposition import lu_inv
from gauss_seidel import gauss_seidel
from gauss_seidel import gauss_seidel_inv
import numpy as np


input_file = open("input.txt", "r")
option = int(input_file.readline())

order = int(input_file.readline())
A = np.loadtxt('input.txt', skiprows=2, usecols=range(order), max_rows=order, delimiter=",")
if option == 1 or option == 4:
    b = np.loadtxt('input.txt', skiprows=3 + order, delimiter=",")
input_file.close()

open('output.txt', 'w').close()
output_file = open("output.txt", "a")

if option == 1:

    answer = lu_decompose(A, b)
    output_file.write(str(answer[0]))
    output_file.write("\n\n")
    output_file.write(str(answer[1]))
    output_file.write("\n\n")
    x = answer[2]

elif option == 2:
    answer = lu_det(A)
    output_file.write(str(answer[0]))
    output_file.write("\n\n")
    output_file.write(str(answer[1]))
    output_file.write("\n\n")
    x = answer[2]

elif option == 3:
    answer = lu_inv(A)
    output_file.write(str(answer[0]))
    output_file.write("\n\n")
    output_file.write(str(answer[1]))
    output_file.write("\n\n")
    x = answer[2]
    b = np.identity(np.size(A, 0))

elif option == 4:
  answer = gauss_seidel(A, b)
  x = answer

elif option == 5:
    answer = gauss_seidel_inv(A)
    x = answer
    b = np.identity(np.size(A, 0))

else:
    raise Exception("Invalid Option")

output_file.write(str(x))
output_file.write("\n\n")
if option != 2:
    error = np.subtract(np.matmul(A, x), b)
    output_file.write(str(error))
    output_file.write("\n\n")
    output_file.write(str(np.linalg.norm(error, ord=np.inf)))

