'''
ME 598 CUDA
Homework 3
Author: Hien (Ryan) Nguyen
Last modified: 01/28/2018
'''

import numpy as np  # import scientific computing library
import matplotlib.pyplot as plt  # import plotting library
# from numba import cuda
import math
import time
from mpl_toolkits import mplot3d

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
    stencilEven = np.array([0, 1, -1], dtype = np.float32)
    stencilOdd = np.array([1, 1, 0], dtype=np.float32)

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
    testArray = np.zeros(16)
    IC = [0, 1]
    print(LaplaceSolver(testArray, IC, 100))

def LaplaceSolver(inputArray, initalCondition, iteration):
    N = np.size(inputArray)
    for j in range(iteration):
        for i in range(N):
            if i == 0:
                inputArray[i] = initalCondition[0]
            elif i == N-1:
                inputArray[i] = initalCondition[1]
            else:
                inputArray[i] = 0.5 * (inputArray[i-1] + inputArray[i+1])
    return inputArray

def pLaplaceKernel(d_out, d_a, d_IC):
    i = cuda.grid(1)
    n = d_a.size

    if i >= n:
        return




def main():
    # question1()
    question2()

if __name__ == '__main__':
    main()