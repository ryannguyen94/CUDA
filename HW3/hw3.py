'''
ME 598 CUDA
Homework 3
Author: Hien (Ryan) Nguyen
Last modified: 01/28/2018
'''

import numpy as np  # import scientific computing library
import matplotlib.pyplot as plt  # import plotting library
from numba import cuda, float32, float64
import math
import time
from mpl_toolkits import mplot3d

TPB_2 = 32
RAD_2B = 1
RAD_2C = 2
NSHARED_2B = TPB_2 + 2*RAD_2B
NSHARED_2C = TPB_2 + 2*RAD_2C

""" Question 1 functions """

def analysisArray(testArray):
    N = testArray.size
    stencilEven = 0.5 * np.array ([0 , 1. , 1. ], dtype = np. float32)
    stencilOdd = 0.5 * np.array ([-1. , 1. , 0 ], dtype = np. float32)

    firstArray = np.zeros(N//2)
    secondArray = np.zeros(N//2)
    for i in range(N):
        if i == 0:
            for j in range(len(stencilEven) - 1):
                firstArray[i] += stencilEven[j+1] * testArray[i+j]
        elif i == (N-1):
            for j in range(len(stencilOdd) - 1):
                secondArray[(N-1)//2] += stencilOdd[j] * testArray[i+j-1]
        elif i%2 == 0:
            for j in range(len(stencilEven)):
                firstArray[i//2] += stencilEven[j] * testArray[i+j-1]
        else:
            for j in range(len(stencilOdd)):
                secondArray[(i-1)//2] += stencilOdd[j] * testArray[i+j-1]
    return [firstArray, secondArray]

def systhesizeArray(firstArray, secondArray):
    interlace = (np.transpose(np.array([firstArray, secondArray]))).flatten()
    stencilEven = np.array([0, 1, -1], dtype = float32)
    stencilOdd = np.array([1, 1, 0], dtype=float32)

    N = np.size(interlace)
    ogArray = np.zeros(N)

    for i in range(N):
        if i == 0:
            for j in range(len(stencilEven) - 1):
                ogArray[i] += stencilEven[j+1] * interlace[i+j]
        elif i == (N-1):
            for j in range(len(stencilOdd) - 1):
                ogArray[N-1] += stencilOdd[j] * interlace[i+j-1]
        elif i%2 == 0:
            for j in range(len(stencilEven)):
                ogArray[i] += stencilEven[j] * interlace[i+j-1]
        else:
            for j in range(len(stencilOdd)):
                ogArray[i] += stencilOdd[j] * interlace[i+j-1]

    return ogArray

def question1():
    x = np.linspace(0, 7, 8)
    testData1 = np.ones(16)
    [firstArray, secondArray] = analysisArray(testData1)
    plt.figure(0)
    plt.plot(x, firstArray, x, secondArray)
    print("Original array is ", testData1)
    print ("Synthesized array is ", systhesizeArray(firstArray, secondArray))

    testData2 = np.linspace(0, 1, 16)
    [firstArray, secondArray] = analysisArray(testData2)
    plt.figure(1)
    plt.plot(x, firstArray, x, secondArray)
    print("Original array is ", testData2)
    print ("Synthesized array is ", systhesizeArray(firstArray, secondArray))

    testData3 = (np.linspace(0, 1, 16))**2
    [firstArray, secondArray] = analysisArray(testData3)
    plt.figure(2)
    plt.plot(x, firstArray, x, secondArray)
    print("Original array is ", testData3)
    print ("Synthesized array is ", systhesizeArray(firstArray, secondArray))

    testData4 = np.random.random(16)
    [firstArray, secondArray] = analysisArray(testData4)
    plt.figure(3)
    plt.plot(x, firstArray, x, secondArray)
    print("Original array is ", testData4)
    print ("Synthesized array is ", systhesizeArray(firstArray, secondArray))

    testData5 = np.zeros(16)
    [firstArray, secondArray] = analysisArray(testData5)
    print("Original array is ", testData5)
    print ("Synthesized array is ", systhesizeArray(firstArray, secondArray))

    plt.show()

""" Question 2 functions """
def question2():

    IC = [0, 1]
    print(LaplaceSolver5(np.zeros(12), IC, 1000))

    # print(pLaplaceSolver(np.zeros(16), IC, 500))

    # # print(psLaplaceSolver(np.zeros(16), IC, 500))

    print(psLaplaceSolver5(np.zeros(10), IC, 500))


def LaplaceSolver(inputArray, initalCondition, iteration):
    N = np.size(inputArray)
    outputArray = inputArray[:]
    for j in range(iteration):
        for i in range(N):
            if i == 0:
                outputArray[i] = initalCondition[0]
            elif i == N-1:
                outputArray[i] = initalCondition[1]
            else:
                outputArray[i] = 0.5 * (outputArray[i-1] + outputArray[i+1])
    return outputArray

def LaplaceSolver5(inputArray, initalCondition, iteration):
    N = np.size(inputArray)
    outputArray = inputArray[:]
    for j in range(iteration):
        for i in range(N):
            if i == 0:
                outputArray[i] = initalCondition[0]
            elif i == N-1:
                outputArray[i] = initalCondition[1]
            elif (i == 1) or (i == N-2):
                outputArray[i] = 0.5 * (outputArray[i-1] + outputArray[i+1])
            else:
                outputArray[i] = outputArray[i - 2] * (-1/30) + outputArray[i - 1] * (8/15) + outputArray[i + 1] * (8/15) + outputArray[i + 2] * (-1/30)
    return outputArray

@cuda.jit
def pLaplaceKernel(d_a, d_IC):
    i = cuda.grid(1)
    n = d_a.size

    if i >= n:
        return

    if (i == 0):
        d_a[i] = d_IC[0]
    elif (i == (n-1)):
        d_a[i] = d_IC[1]
    else:
        d_a[i] = 0.5 * (d_a[i-1] + d_a[i+1])

def pLaplaceSolver(inputArray, initalCondition, iteration):

    N = np.size(inputArray)
    
    d_a = cuda.to_device(np.zeros(N))
    d_IC = cuda.to_device(np.array(initalCondition))

    gridDim = (N + TPB_2 - 1)//TPB_2
    blockDim = TPB_2

    for j in range(iteration):
        pLaplaceKernel[gridDim, blockDim](d_a, d_IC)
    return d_a.copy_to_host()

@cuda.jit
def psLaplaceKernel(d_a, d_IC, d_stencil, rad):
    i = cuda.grid(1)
    n = d_a.size
    if rad == 1: 
        sh_f = cuda.shared.array(shape = (NSHARED_2B), dtype = float64)
    else:
        sh_f = cuda.shared.array(shape = (NSHARED_2C), dtype = float64)

    if i >= n:
        return



    radius = len(d_stencil) // 2
    tIdx = cuda.threadIdx.x
    shIdx = tIdx + radius
    sh_f[shIdx] = d_a[i]



    # if tIdx < radius:
    #     for k in range(radius):
    #         sh_f[tIdx+k] = d_a[i - radius + k]
    # elif tIdx > cuda.blockDim.x - 1 - radius:
    #     for k in range(radius):
    #         sh_f[tIdx + 2*radius - k] = d_a[i + radius - k]

    # if tIdx < radius:
    #   sh_f[tIdx] = d_a[i - radius]
    # elif tIdx > cuda.blockDim.x - 1 - radius:
    #   sh_f[tIdx + 2*radius] = d_a[i + radius]
    # cuda.syncthreads()

    # d_a[i] = 0
    # elif i == 1:
    #   d_a[i] = 0.0667
    # elif i == n-2:
    #   d_a[i] = 0.933
    if i == 0:
        d_a[i] = d_IC[0]
    elif i == n - 1:
        d_a[i] = d_IC[1]
    elif (i == 1) or (i == n-2):
        d_a[i] = 0.5 * (d_a[i-1] + d_a[i+1])
    else:
        d_a[i] = d_a[i - 2] * (-0.0333) + d_a[i - 1] * (0.5333) + d_a[i + 1] * (0.5333) + d_a[i + 2] * (-0.03333)
    # if (i == 1) or (i == n-2):
    #   d_a[i] = 0.5 * (sh_f[shIdx - 1] + sh_f[shIdx + 1])
    # elif (i > 1) and (i < n-2):
    #   d_a[i] = sh_f[shIdx - 2] * (-1/30) + sh_f[shIdx - 1] * (8/15) + sh_f[shIdx + 1] * (8/15) + sh_f[shIdx + 2] * (-1/30)
        # for j in range(len(d_stencil)):
        #   d_a[i] += sh_f[shIdx + j - radius]*d_stencil[j]

def psLaplaceSolver(inputArray, initalCondition, iteration):

    N = np.size(inputArray)
    
    d_a = cuda.to_device(np.array(inputArray))
    d_IC = cuda.to_device(np.array(initalCondition))
    d_stencil = cuda.to_device(np.array([0.5, 0, 0.5]))

    gridDim = (N + TPB_2 - 1)//TPB_2
    blockDim = TPB_2

    prevNorm = nu_norm(d_a, N) 
    for j in range(iteration):
        psLaplaceKernel[gridDim, blockDim](d_a, d_IC, d_stencil, RAD_2B)
        newNorm = nu_norm(d_a, N) 
        if (abs(newNorm - prevNorm) < 0.0001):
            print(j)
            break
        prevNorm = newNorm

    return d_a.copy_to_host()


def psLaplaceSolver5(inputArray, initalCondition, iteration):

    N = np.size(inputArray)
    
    d_a = cuda.to_device(np.array(inputArray))
    d_IC = cuda.to_device(np.array(initalCondition))
    d_stencil = cuda.to_device(np.array([-1/30, 8/15, 0, 8/15, -1/30]))

    gridDim = (N + TPB_2 - 1)//TPB_2
    blockDim = TPB_2

    prevNorm = nu_norm(d_a, N) 
    for j in range(5000):
        psLaplaceKernel[gridDim, blockDim](d_a, d_IC, d_stencil, RAD_2C)
        # newNorm = nu_norm(d_a, N) 
        # if (abs(newNorm - prevNorm) < 0.0001):
        #   print(j)
        #   break
        # prevNorm = newNorm

    return d_a.copy_to_host()

@cuda.jit
def norm_kernel(d_u, d_out):
    n = d_u.shape[0]
    i = cuda.grid(1)
    if i >= n:
        return
    d_out[i] = d_u[i] ** 2


def nu_norm(d_a, n):

    d_out = cuda.device_array(n)
    threads = TPB_2  # excessive use of local variables for clarity
    grids = (n + TPB_2 - 1) // TPB_2  # ditto
    norm_kernel[grids, threads](d_a, d_out)
    res = 0
    out = d_out.copy_to_host()
    for i in range(n):
        res += out[i]
    return np.sqrt(res)

def main():
    # question1()
    question2()

if __name__ == '__main__':
    main()