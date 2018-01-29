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

    print ("---------- Question 2 ----------")

    TPBX = 8
    TPBY = 32
    NX = np.linspace(100, 3500, 10)
    NY = np.linspace(100, 3500, 10)

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
	print ("---------- Question 4 ----------")	

	print ("Change in aspect ratio has no affect on the kernel"
		" execution time or kernal call")

''' Question 5 functions '''
def question5(p):
	print ("---------- Question 5 ----------")

	arrayDimX = 256
	arrayDimY = 256

	array = [[0]*arrayDimX] * arrayDimY
	x = np.linspace(0, 2*math.pi, arrayDimX)
	for i in range(arrayDimY):
		array[i] = np.sin(np.linspace(x[i], 2*math.pi + x[i], arrayDimX))
	
	res = pnorm(array, p)
	print ("Result is: ", res)

@cuda.jit
def norm_kernel(d_array, out, p):
    i , j = cuda.grid (2)
    nx , ny = d_array.shape
    if i < nx and j < ny:
        out[0] += (d_array[i,j] ** p)


def pnorm(array, p):

    TPBX = 8
    TPBY = 8

    nx, ny = np.array(array).shape

    d_array = cuda.to_device(np.array(array))
    d_out = cuda.to_device(np.zeros(1))

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)


    norm_kernel[gridDims, blockDims](d_array, d_out, p)
    return d_out.copy_to_host()[0] ** (1/p)

''' Question 6 '''
def question6():
    print ("---------- Question 6 ----------")

    t = np.linspace(0, 10, 10000)
    dt = 10/1000
    x_i = np.linspace(0, 3, 4)
    v_i = np.linspace(0, 3, 4)

    iterate(x_i, v_i, dt)

    for i in range(len(t) - 1):
    	[x[i+1], v[i+1]] = iterate(x[i], v[i], dt)

    plt.plot(x, v)
    plt.show()

# 6c
def iterate(x_i, v_i, dt):
    TPBX = 16
    TPBY = 16

    nx, nv = np.array((x_i, v_i)).shape
    d_xi = cuda.to_device(np.array(x_i))
    d_vi = cuda.to_device(np.array(v_i))
    d_out = cuda.device_array((nx, ny), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (nv + TPBV - 1) // TPBV)
    blockDims = (TPBX, TPBV)


    iterate_kernel[gridDims, blockDims](d_xi, d_vi, d_out, dt)
    return d_out.copy_to_host()

@cuda.jit
def iterate_kernel(d_xi, d_vi, d_out, dt):
	i, j = cuda.grid(2)
    nx , nv = d_out.shape

    if i < nx and j < ny:
    	for i in range(100):



# 6d


def main():
    # question2()
    # question3()
    # question4()
    # question5(10000)
    question6(1, 1)

# call to execute main
if __name__ == '__main__':
    main()