'''
ME 598 CUDA
Homework 2
Author: Hien (Ryan) Nguyen
Last modified: 01/28/2018
'''

import numpy as np  # import scientific computing library
import matplotlib.pyplot as plt  # import plotting library
from numba import cuda
import math
import time
from mpl_toolkits import mplot3d



''' Question 2 functions '''
def f2D (x, y):
    return math.sin ( np.pi *x)* math.sinh ( np.pi *y)/ math.sinh ( np.pi )

def fArray2D (x, y):
    nx = x.size
    ny = y.size
    f = np.empty ((nx ,ny), dtype = np.float32)
    for i in range (nx):
        for j in range (ny):
            f[i,j] = f2D (x[i], y[j])
        return f

@cuda.jit ( device = True )
def pf2D (x, y):
    return math.sin ( np.pi *x)* math.sinh ( np.pi *y)/ math.sinh ( np.pi )

@cuda.jit ('void (f4 [:] , f4 [:] , f4 [: ,:])')
def pfKernel2D (d_x , d_y , d_f):
    i , j = cuda.grid (2)
    nx , ny = d_f.shape
    if i < nx and j < ny:
        d_f[i,j] = pf2D (d_x[i], d_y[j])

def pfArray2D (x, y, TPBX, TPBY):
    
    nx = x.size
    ny = y.size
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_f = cuda.device_array((nx, ny), dtype=np.float32)
    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)

    startTime = time.time()
    pfKernel2D[gridDims, blockDims](d_x, d_y, d_f)
    kTime = (time.time() - startTime) * 1000
    print ("Kernal call time is: ", kTime)
    return d_f.copy_to_host()

def question2():
    print ()
    print ("---------- Question 2 ----------")

    TPBX = 8
    TPBY = 32
    NX = np.linspace(100, 1000, 10)
    NY = np.linspace(100, 1000, 10)

    sTime = [0]*len(NX)
    pTime = [0]*len(NX)
    accel = [0]*len(NX)

    for i in range(len(NX)):
        print ("Array size: ", NX[i])
        x = np.linspace (0,1,NX[i] , dtype = np.float32)
        y = np.linspace (0,1,NY[i] , dtype = np.float32)
        startTime = time.time()
        fs = fArray2D (x, y)
        sTime[i] = (time.time() - startTime) * 1000
        print ("Series processing time: ", sTime[i])

        startTime = time.time()
        fp = pfArray2D(x, y, TPBX, TPBY)
        pTime[i] = (time.time() - startTime) * 1000
        print ("Parallel processing time: ", pTime[i])

        accel[i] = sTime[i]/pTime[i]
        print ("Accel is: ", accel[i])

    plt.figure(1)
    plt.subplot(211)
    plt.plot(NX, sTime, 'r--', label='series runtime')
    plt.plot(NX, pTime, 'g^', label='parallel_runtime')
    plt.legend()
    plt.title("Series and Parallel Runtime vs Array Size")

    plt.subplot(212)
    plt.plot(NX, accel)
    plt.title("Acceleration vs Array Size")
    plt.show()

''' Question 3 functions '''
def question3():
    print ()
    print ("---------- Question 3 ----------")

    TPBX = 32
    TPBY = 32
    NX = 255
    NY = 255

    x = np.linspace(0, 1, NX, dtype=np.float32)
    y = np.linspace(0, 1, NY, dtype=np.float32)

    fp = pfArray2D(x, y, TPBX, TPBY)
    print ("32 is the largest number of thread a block can have."
        " Anything larger than that will produce the following error:"
        " numba.cuda.cudadrv.driver.CudaAPIError: [1] Call to"
        " cuLaunchKernel results in CUDA_ERROR_INVALID_VALUE")

''' Question 4 functions '''
def question4():
    print ()
    print ("---------- Question 4 ----------")  

    print ("Change in aspect ratio has very little affect on the kernel"
        " execution time or kernal call")

''' Question 5 functions '''
def question5():
    print ()
    print ("---------- Question 5 ----------")

    arrayDimX = 255
    arrayDimY = 255

    array = [[0]*arrayDimX] * arrayDimY
    x = np.linspace(0, 2*math.pi, arrayDimX)
    y = np.linspace(0, 2*math.pi, arrayDimY)
    
    array = make_matrix(x, y)

    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, array)
    plt.show()

    print ("Compute L2:")
    res = pnorm(array, 2)
    print ("Result is: ", res)

    print ("Compute L4:")
    res = pnorm(array, 4)
    print ("Result is: ", res)

    print ("Compute L6:")
    res = pnorm(array, 6)
    print ("Result is: ", res)

    print ("Compute L1000:")
    res = pnorm(array, 1000)
    print ("Result is: ", res)

    print ("The value of norm approaches 1 which is norm infinity as p increases")


def make_matrix(x, y):
    TPBX = 8
    TPBY = 8

    nx = np.array(x).shape[0]
    ny = np.array(y).shape[0]

    d_x = cuda.to_device(np.array(x))
    d_y = cuda.to_device(np.array(y))
    d_out = cuda.device_array((nx, ny))

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)


    make_matrix_kerneld[gridDims, blockDims](d_x, d_y, d_out)
    return d_out.copy_to_host()


@cuda.jit
def make_matrix_kerneld(d_x, d_y, d_out):
    i , j = cuda.grid (2)
    nx = d_x.shape[0]
    ny = d_y.shape[0]
    if i < nx and j < ny:
        d_out[i, j] = math.sin(2*math.pi*d_x[i])*math.sin(2*math.pi*d_y[j])

@cuda.jit
def norm_kernel(d_array, p):
    i , j = cuda.grid (2)
    nx , ny = d_array.shape
    if i < nx and j < ny:
        d_array[i,j] = (d_array[i,j] ** p)



def pnorm(array, p):

    TPBX = 8
    TPBY = 8

    nx, ny = np.array(array).shape

    d_array = cuda.to_device(np.array(array))

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)


    norm_kernel[gridDims, blockDims](d_array, p)
    res = 0
    d_arrayFlat = d_array.copy_to_host().flatten()
    for i in range(d_arrayFlat.shape[0]):
        res += d_arrayFlat[i]
    return res ** (1/p)


''' Question 6 '''
def question6():
    print ()
    print ("---------- Question 6 ----------")

    print ("For IVPs problems, there is no direct way to parallelize "
        "the computation over a grid of time intervals because current "
        "value depends on previous values of each states and thus to get "
        "value at time k, we need to already compute value of all states "
        "at time k-1")

    print ("For IVPs problems, there is a way to parallelize over a "
        "grid of initial conditions because the iteration process for "
        "each group of initial conditions are independent")

    nt = 1000
    t = np.linspace(0, 10, nt)
    dt = 1/nt
    x_i = np.linspace(-3, 3, 50)
    v_i = np.linspace(-3, 3, 50)

    X,V = np.meshgrid(x_i, v_i)

    for subprob in ["6c", "6d", "6e"]:
        print ("Subproblem ", subprob)
        r = iterate(x_i, v_i, dt, nt, subprob)

        fig = plt.figure(6)
        ax = plt.axes(projection='3d')


        ax.scatter3D(X, V, r)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('r')
        plt.show()

    

# 6c
def iterate(x_i, v_i, dt, nt, prob):
    TPBX = 16
    TPBV = 16

    nx = np.array(x_i).shape[0]
    nv = np.array(v_i).shape[0]

    d_xi = cuda.to_device(np.array(x_i))
    d_vi = cuda.to_device(np.array(v_i))
    d_x = cuda.device_array((nx, nv, nt))
    d_v = cuda.device_array((nx, nv, nt))
    d_r = cuda.device_array((nx, nv))

    gridDims = ((nx + TPBX - 1) // TPBX,
                (nv + TPBV - 1) // TPBV)
    blockDims = (TPBX, TPBV)

    if prob == "6c":
        iterate_kernel_6c[gridDims, blockDims](d_xi, d_vi, d_x, d_v, dt, nt, d_r)
    elif prob == "6d":
        iterate_kernel_6d[gridDims, blockDims](d_xi, d_vi, d_x, d_v, dt, nt, d_r)
    elif prob == "6e":
        iterate_kernel_6e[gridDims, blockDims](d_xi, d_vi, d_x, d_v, dt, nt, d_r)
    return d_r.copy_to_host()

# 6d
@cuda.jit
def iterate_kernel_6d(d_xi, d_vi, d_x, d_v, dt, nt, d_r):
    i, j = cuda.grid(2)
    nx = d_xi.shape[0]
    nv = d_vi.shape[0]

    if i < nx and j < nv:
        d_x[i, j, 0] = d_xi[i]
        d_v[i, j, 0] = d_vi[j]
        for k in range(nt-1):
            d_v[i, j, k+1] = d_v[i, j, k] + (- d_x[i, j, k] - 0.1 * d_v[i, j, k]) * dt
            d_x[i, j, k+1] = d_x[i, j, k] + d_v[i, j, k] * dt
            
        d_r[i,j] = (d_v[i, j, nt-1] ** 2 + d_x[i, j, nt-1] **2) ** 0.5 /((d_xi[i]**2 + d_vi[j]**2)**0.5)


# 6c
@cuda.jit
def iterate_kernel_6c(d_xi, d_vi, d_x, d_v, dt, nt, d_r):
    i, j = cuda.grid(2)
    nx = d_xi.shape[0]
    nv = d_vi.shape[0]

    if i < nx and j < nv:
        d_x[i, j, 0] = d_xi[i]
        d_v[i, j, 0] = d_vi[j]
        for k in range(nt-1):
            d_v[i, j, k+1] = d_v[i, j, k] - d_x[i, j, k] * dt
            d_x[i, j, k+1] = d_x[i, j, k] + d_v[i, j, k] * dt
            
        d_r[i,j] = (d_v[i, j, nt-1] ** 2 + d_x[i, j, nt-1] **2) ** 0.5 /((d_xi[i]**2 + d_vi[j]**2)**0.5)

# 6e
@cuda.jit
def iterate_kernel_6e(d_xi, d_vi, d_x, d_v, dt, nt, d_r):
    i, j = cuda.grid(2)
    nx = d_xi.shape[0]
    nv = d_vi.shape[0]

    if i < nx and j < nv:
        d_x[i, j, 0] = d_xi[i]
        d_v[i, j, 0] = d_vi[j]
        for k in range(nt-1):
            d_v[i, j, k+1] = d_v[i, j, k] + (- d_x[i, j, k] + 0.1*(1-d_x[i, j, k]**2) * d_v[i, j, k]) * dt
            d_x[i, j, k+1] = d_x[i, j, k] + d_v[i, j, k] * dt
            
        d_r[i,j] = (d_v[i, j, nt-1] ** 2 + d_x[i, j, nt-1] **2) ** 0.5 /((d_xi[i]**2 + d_vi[j]**2)**0.5)


def main():
    question2()
    question3()
    question4()
    question5()
    question6()

# call to execute main
if __name__ == '__main__':
    main()