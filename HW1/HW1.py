'''
ME 598 CUDA
Author: Hien (Ryan) Nguyen
Last modified: 01/16/2018
'''

import numpy as np  # import scientific computing library
import matplotlib.pyplot as plt  # import plotting library
from numba import cuda
import math

# define values for global variables
TPB = 8  # number of threads per block

""" Question 2 function """


def RN_linspace(min, max, size):
    res = np.empty(size)
    for i in range(0, size):
        res[i] = min + i * (max - min) / (size - 1)
    return res

""" Question 3 functions """

def scalar_mult(u, c):  # 3a
    res = np.empty(np.size(u))
    for i in range(np.size(res)):
        res[i] = u[i] * c
    return res


# alternate version that writes results to 'out'
# def scalar_mult(out,u,c): #3a
# 	pass

def add_comp(u, v):  # 3b
    res = np.empty(np.size(u))
    for i in range(np.size(u)):
        res[i] = u[i] + v[i]
    return res


def lerp(x, c, d):  # 3c
    return add_comp(scalar_mult(x, c), d)


# version whose output can be assigned to an array
# call this version as: 'out = mult_comp2(u,v)'
def mult_comp2(u, v):  # 3d - assumes u and v are numpy arrays
    n = u.shape[0]  # number of entries in array
    out = np.zeros(n)
    for i in range(n):
        out[i] = u[i] * v[i]
    return out


# alternate version called with an output array
# calling 'mult_comp3(out,u,v)' stores the results in 'out'
def mult_comp3(out, u, v):  # 3d - assumes u and v are numpy arrays
    n = u.shape[0]  # number of entries in array
    out = np.zeros(n)
    for i in range(n):
        out[i] = u[i] * v[i]
    return out


def inner(u, v):  # 3e
    n = u.shape[0]
    out = 0
    for i in range(n):
        out += u[i] * v[i]
    return out


def norm(u):  # 3f
    n = u.shape[0]
    out = 0
    for i in range(n):
        out += u[i] ** 2
    return math.sqrt(out)


""" Question 4 functions """

# 4a
@cuda.jit
def scalar_mult_kernel(d_u, d_c, d_out):
    n = d_u.shape[0]
    i = cuda.grid(1)
    if i >= n:
        return
    d_out[i] = d_u[i] * d_c[0]


def nu_scalar_mult(u, c):
    n = u.shape[0]
    d_u = cuda.to_device(u)
    d_c = cuda.to_device(np.ones(1)*c)
    d_out = cuda.device_array(n)
    threads = TPB  # excessive use of local variables for clarity
    grids = (n + TPB - 1) // TPB  # ditto
    scalar_mult_kernel[grids, threads](d_u, d_c, d_out)
    return d_out.copy_to_host()

# 4b
@cuda.jit
def add_comp_kernel(d_u, d_v, d_out):
    n = d_u.shape[0]
    i = cuda.grid(1)
    if i >= n:
        return
    d_out[i] = d_u[i] + d_v[i]


def nu_add_comp(u, v):
    n = u.shape[0]
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(n)
    threads = TPB  # excessive use of local variables for clarity
    grids = (n + TPB - 1) // TPB  # ditto
    add_comp_kernel[grids, threads](d_u, d_v, d_out)
    return d_out.copy_to_host()

# 4c
@cuda.jit
def linearF_kernel(d_x, d_c, d_d, d_out):
    n = d_x.shape[0]
    i = cuda.grid(1)
    if i >= n:
        return
    d_out[i] = d_c[0]*d_x[i] + d_d[i]


def nu_linearF(x, c, d):
    n = x.shape[0]
    d_x = cuda.to_device(x)
    d_c = cuda.to_device(np.ones(1)*c)
    d_d = cuda.to_device(d)
    d_out = cuda.device_array(n)
    threads = TPB  # excessive use of local variables for clarity
    grids = (n + TPB - 1) // TPB  # ditto
    linearF_kernel[grids, threads](d_x, d_c, d_d, d_out)
    return d_out.copy_to_host()

# 4d
@cuda.jit
def mult_comp_kernel(d_u, d_v, d_out):
    n = d_u.shape[0]
    i = cuda.grid(1)
    if i >= n:
        return
    d_out[i] = d_u[i] * d_v[i]


def nu_mult_comp2(u, v):
    n = u.shape[0]
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(n)
    threads = TPB  # excessive use of local variables for clarity
    grids = (n + TPB - 1) // TPB  # ditto
    mult_comp_kernel[grids, threads](d_u, d_v, d_out)
    return d_out.copy_to_host()

# alternate version
def nu_mult_comp3(u, v, out):
    n = u.shape[0]
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(n)
    threads = TPB
    grids = (n + TPB - 1) // TPB
    mult_comp_kernel[grids, threads](d_u, d_v, d_out)
    out = d_out.copy_to_host()

# 4e
def nu_inner(u, v):
    out = nu_mult_comp2(u, v)
    res = 0
    for i in range(u.shape[0]):
       res += out[i]
    return res

# 4f
@cuda.jit
def norm_kernel(d_u, d_out):
    n = d_u.shape[0]
    i = cuda.grid(1)
    if i >= n:
        return
    d_out[i] = d_u[i] ** 2


def nu_norm(u):
    n = u.shape[0]
    d_u = cuda.to_device(u)
    d_out = cuda.device_array(n)
    threads = TPB  # excessive use of local variables for clarity
    grids = (n + TPB - 1) // TPB  # ditto
    norm_kernel[grids, threads](d_u, d_out)
    res = 0
    out = d_out.copy_to_host()
    for i in range(n):
        res += out[i]
    return np.sqrt(res)

""" Question 5 functions """

# 5a
def test1(n):
    # i
    v = np.array([1] * n)
    # ii
    u = np.array([1/(n-1)] * n)

    u[0] = 1
    # iii
    z = nu_scalar_mult(u, -1)
    norm = nu_norm(nu_add_comp(u, z))
    print ("Norm is: ", norm)

    # iv
    inner = nu_inner(u, v)
    print ("Inner product is: ", inner)

    # v
    innerReversed = reverseDot(u, v)
    print ("Reversed inner product is: ", innerReversed)

# 5b
def reverseDot(u, v):
    out = nu_mult_comp2(u, v)
    res = 0
    n = u.shape[0]
    for i in range(n):
       res += out[n-i-1]
    return res

# main function to compute requested results
def main():

    # Question 2
    n = 11
    nIndex = range(n)
    array1 = np.linspace(0, 2 * math.pi, 11)
    array2 = RN_linspace(0, 2 * math.pi, 11)

    plt.plot(nIndex, array1, 'r--', label='numpy_linspace')
    plt.plot(nIndex, array2, 'g^', label='custom_linspace')
    plt.legend()
    plt.show()

    # Question 3
    print ("---------- Testing functions of problem 3 ----------")
    array1 = np.array([1, 2, 3])
    array2 = np.array([2, 3, 4])
    c = 5
    print ("Testing scalar multiplication with array ", array1, " and c ", c)
    print(scalar_mult(array1, c))
    print ("Testing component-wise addition with array1 ", array1, " and array2 ", array2)
    print(add_comp(array1, array2))
    print ("Testing linear function with x = ", array1, " c ", c, " and d ", array2)
    print (lerp(array1, c, array2))
    print ("Testing component-wise multiplication with array1 ", array1, " and array2 ", array2)
    print (mult_comp2(array1, array2))
    print ("Testing inner product with array1 ", array1, " and array2 ", array2)
    print (inner(array1, array2))
    print ("Testing norm function with array1 ", array1)
    print (norm(array1))
    print ("\n")

    # Question 4
    print ("---------- Testing functions of problem 4 ----------")
    print ("Testing scalar multiplication with array ", array1, " and c ", c)
    print(nu_scalar_mult(array1, c))
    print ("Testing component-wise addition with array1 ", array1, " and array2 ", array2)
    print(nu_add_comp(array1, array2))
    print ("Testing linear function with x = ", array1, " c ", c, " and d ", array2)
    print (nu_linearF(array1, c, array2))
    print ("Testing component-wise multiplication with array1 ", array1, " and array2 ", array2)
    print(nu_mult_comp2(array1, array2))
    print ("Testing inner product with array1 ", array1, " and array2 ", array2)
    print (nu_inner(array1, array2))
    print ("Testing norm function with array1 ", array1)
    print (nu_norm(array1))
    print ("\n")

    # Question 5
    print ("---------- Testing functions of problem 4 ----------")
    print ("Running first test with n = 5")
    test1(5)
    print()
    print ("Running second test with n = 457000")
    test1(457000)
    print ("\n")
    print ("At ~457000, the result produce a slightly different value for the inner"
    	" product and the reversed inner product. This might be because float operation"
    	" are not necessarily commutative, so at very small float value, the variable"
    	" is round to the nearest value that can be represented in memory. Thefore,"
    	" the order of summation matters")

# call to execute main
if __name__ == '__main__':
    main()
