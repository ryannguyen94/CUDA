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
    pfKernel2D[gridDims, blockDims](d_x, d_y, d_f)
    return d_f.copy_to_host()

def question2():

    TPBX = TPBY = 16
    NX = [255, 1023, 4095, 16383, 65535]
    NY = [255, 1023, 4095, 16383, 65535]

    sTime = pTime = accel = [0]*len(NX)
    # pTime = sTime
    # accel = sTime
    for i in range(len(NX)):
        x = np.linspace (0,1,NX[i] , dtype = np.float32)
        y = np.linspace (0,1,NY[i] , dtype = np.float32)
        startTime = time.time()
        fs = fArray2D (x, y)
        sTime[i] = time.time() - startTime

        startTime = time.time()
        fp = pfArray2D(x, y, TPBX, TPBY)
        pTime[i] = time.time() - startTime

        accel[i] = pTime[i]/sTime[i]

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

    # X,Y = np.meshgrid (x, y)
    # plt.contourf (X, Y, f)
    # plt.show ()

''' Question 3 functions '''
def question3():

    TPBX = TPBY = 1000
    NX = NY = 65535

    x = np.linspace(0, 1, NX, dtype=np.float32)
    y = np.linspace(0, 1, NY, dtype=np.float32)

    fp = pfArray2D(x, y, TPBX, TPBY)

def main():

if __name__ == '__main__ ':
    main ()