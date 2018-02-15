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

TPB_3C_X = 16
TPB_3C_Y = 16
RAD_3C = 1
NSHARED_3C_X = TPB_3C_X + 2*RAD_3C
NSHARED_3C_Y = TPB_3C_Y + 2*RAD_3C
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
    print(LaplaceSolver(np.zeros(16), IC, 1000))

    print(pLaplaceSolver(np.zeros(16), IC, 500))

    print(psLaplaceSolver(np.zeros(16), IC, 500))

    print(psLaplaceSolver5(np.zeros(16), IC, 1000))


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

    if i == 0:
        d_a[i] = d_IC[0]
    elif i == n - 1:
        d_a[i] = d_IC[1]

    radius = len(d_stencil) // 2
    tIdx = cuda.threadIdx.x
    shIdx = tIdx + radius
    sh_f[shIdx] = d_a[i]

    if tIdx < radius:
        for k in range(radius):
            sh_f[tIdx+k] = d_a[i - radius + k]
    elif tIdx > cuda.blockDim.x - 1 - radius:
        for k in range(radius):
            sh_f[tIdx + 2*radius - k] = d_a[i + radius - k]

    cuda.syncthreads()

    if (i == 1) or (i == n-2):
      d_a[i] = 0.5 * (sh_f[shIdx - 1] + sh_f[shIdx + 1])
    elif (i > 1) and (i < n-2):
        d_a[i] = (1-0.9) * d_a[i]
        for j in range(len(d_stencil)):
            d_a[i] += 0.9 * (sh_f[shIdx + j - radius]*d_stencil[j])

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
    for j in range(iteration):
        psLaplaceKernel[gridDim, blockDim](d_a, d_IC, d_stencil, RAD_2C)
        newNorm = nu_norm(d_a, N) 
        if (abs(newNorm - prevNorm) < 0.0001):
          print(j)
          break
        prevNorm = newNorm

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

""" Question 3 functions """
def question3():
    N = 64
    x = np.linspace(-2, 2, N)
    y = np.linspace(-2, 2, N)

    distance = distance_cal(x, y)

    # fig = plt.figure(4)
    X, Y = np.meshgrid(x, y)
    # plt.contourf(X, Y, np.transpose(distance))

    # fig = plt.figure(5)
    # ax = plt.axes(projection='3d')

    # ax.scatter3D(X, Y, np.transpose(distance))
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('dis')

    startTime = time.time()
    gradient = gradient_cal(x, y, distance)
    pTime = time.time() - startTime

    fig = plt.figure(6)
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, np.transpose(gradient))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('grad')

    startTime = time.time()
    gradient = shGradient_cal(x, y, distance)
    psTime = time.time() - startTime

    fig = plt.figure(7)
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, np.transpose(gradient))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('grad')


    # plt.show()

    print ("Acceleration for shared memory parallel implementation "
        "(shared_mem_parallel/global_mem_parallel) is: ", 
        psTime/pTime)


@cuda.jit
def distance_kernel(d_x, d_y, d_points, d_out):
    nx = d_x.shape[0]
    ny = d_y.shape[0]

    i, j = cuda.grid(2)

    if i < nx and j < ny:
        dis1 = ((d_x[i] - d_points[0][0])**2 + (d_y[j] - d_points[0][1])**2)**0.5
        dis2 = ((d_x[i] - d_points[1][0])**2 + (d_y[j] - d_points[1][1])**2)**0.5
        dis3 = ((d_x[i] - d_points[2][0])**2 + (d_y[j] - d_points[2][1])**2)**0.5

    dis = (dis1, dis2, dis3)
    d_out[i, j] = dis[0]
    for k in range(len(dis)-1):
        if dis[k+1] < d_out[i, j]:
            d_out[i, j] = dis[k+1]

def distance_cal(x, y):

    Nx = x.size
    Ny = y.size

    points = [[1, 0], [-1, 0], [0, 1]]

    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_points = cuda.to_device(np.array(points))
    d_out = cuda.device_array((Nx,Ny))

    TPBX = 8
    TPBY = 8

    gridDims = ((Nx + TPBX - 1) // TPBX,
                (Ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)

    distance_kernel[gridDims, blockDims](d_x, d_y, d_points, d_out)

    return d_out.copy_to_host()

@cuda.jit
def gradient_kernel(d_u, delta_x, delta_y, d_out):
    nx = d_u.shape[0]
    ny = d_u.shape[1]

    i, j = cuda.grid(2)

    if i < nx and j < ny:
        if i == 0:
            gradientX = ((d_u[i, j] + d_u[i+1, j])/(delta_x))**2
        elif i == nx-1:
            gradientX = ((d_u[i-1, j] + d_u[i, j])/(delta_x))**2
        else:
            gradientX = ((d_u[i-1, j] + d_u[i+1, j])/(2*delta_x))**2
        if j == 0:
            gradientY = ((d_u[i, j] + d_u[i, j+1])/(delta_y))**2
        elif j == ny-1:
            gradientY = ((d_u[i, j-1] + d_u[i, j])/(delta_y))**2
        else:
            gradientY = ((d_u[i, j-1] + d_u[i, j+1])/(2*delta_y))**2

        d_out[i, j] = gradientX + gradientY

def gradient_cal(x, y, u):
    Nx = x.size
    Ny = y.size

    delX = x[1] - x[0]
    delY = y[1] - y[0]

    d_u = cuda.to_device(u)
    d_out = cuda.device_array((Nx, Ny))

    TPBX = 8
    TPBY = 8

    gridDims = ((Nx + TPBX - 1) // TPBX,
                (Ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)

    gradient_kernel[gridDims, blockDims](d_u, delX, delY, d_out)
    return d_out.copy_to_host()

@cuda.jit
def shGradient_kernel(d_u, delta_x, delta_y, d_out):

    nx = d_u.shape[0]
    ny = d_u.shape[1]

    i, j = cuda.grid(2)

    sh_f = cuda.shared.array(shape = (NSHARED_3C_X, NSHARED_3C_Y), dtype = float64)
    radius = 1
    tIdx = cuda.threadIdx.x
    tIdy = cuda.threadIdx.y

    shx = tIdx + radius
    shy = tIdy + radius
    sh_f[shx, shy] = d_u[i, j]

    if tIdx < radius:
        sh_f[shx - radius, shy] = d_u[i - radius, j]
    elif tIdx > cuda.blockDim.x - 1 - radius:
        sh_f[shx + radius, shy] = d_u[i + radius, j]

    if tIdy < radius:
        sh_f[shx, shy - radius] = d_u[i, j - radius]
    elif tIdy > cuda.blockDim.y - 1 - radius:
        sh_f[shx, shy + radius] = d_u[i, j + radius]    

    cuda.syncthreads()

    if i < nx and j < ny:
        if i == 0:
            gradientX = ((sh_f[shx, shy] + sh_f[shx+1, shy])/(delta_x))**2
        elif i == nx-1:
            gradientX = ((sh_f[shx-1, shy] + sh_f[shx, shy])/(delta_x))**2
        else:
            gradientX = ((sh_f[shx-1, shy] + sh_f[shx+1, shy])/(2*delta_x))**2
        if j == 0:
            gradientY = ((sh_f[shx, shy] + sh_f[shx, shy+1])/(delta_y))**2
        elif j == ny-1:
            gradientY = ((sh_f[shx, shy-1] + sh_f[shx, shy])/(delta_y))**2
        else:
            gradientY = ((sh_f[shx, shy-1] + sh_f[shx, shy+1])/(2*delta_y))**2

        d_out[i, j] = gradientX + gradientY

def shGradient_cal(x, y, u):
    Nx = x.size
    Ny = y.size

    delX = x[1] - x[0]
    delY = y[1] - y[0]

    d_u = cuda.to_device(u)
    d_out = cuda.device_array((Nx, Ny))

    gridDims = ((Nx + TPB_3C_X - 1) // TPB_3C_X,
                (Ny + TPB_3C_Y - 1) // TPB_3C_Y)
    blockDims = (TPB_3C_X, TPB_3C_Y)

    shGradient_kernel[gridDims, blockDims](d_u, delX, delY, d_out)
    return d_out.copy_to_host()

""" Question 4 functions """
def question4():
    N = 64
    x = np.linspace(-2, 2, N)
    y = np.linspace(-2, 2, N)

    h = x[1] - x[0]

    f = gridCal(x, y)

    fig = plt.figure(7)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, np.transpose(f))

    res = upwindCal(f, h, 10)
    fig = plt.figure(8)
    plt.contourf(X, Y, np.transpose(res))        

    res = upwindCal(f, h, 20)
    fig = plt.figure(9)
    plt.contourf(X, Y, np.transpose(res))

    f = -f

    fig = plt.figure(10)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, np.transpose(f))

    res = upwindCal(f, h, 8)
    fig = plt.figure(11)
    plt.contourf(X, Y, np.transpose(res))        

    res = upwindCal(f, h, 16)
    fig = plt.figure(12)
    plt.contourf(X, Y, np.transpose(res))

    plt.show()

@cuda.jit
def gridCal_kernel(d_x, d_y, d_out):
    nx = d_x.shape[0]
    ny = d_y.shape[0]

    i, j = cuda.grid(2)

    if i < nx and j < ny:
        d_out[i, j] = d_x[i]**2 + d_y[j]**2 - 1

def gridCal(x, y):
    Nx = x.size
    Ny = y.size

    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_out = cuda.device_array((Nx, Ny))

    TPBX = 8
    TPBY = 8

    gridDims = ((Nx + TPBX - 1) // TPBX,
                (Ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)

    gridCal_kernel[gridDims, blockDims](d_x, d_y, d_out)
    return d_out.copy_to_host()

@cuda.jit
def upwind_kernel(d_f, h):
    nx = d_f.shape[0]
    ny = d_f.shape[1]

    i, j = cuda.grid(2)

    if i > nx or j > ny:
        return
    if d_f[i, j] > 0:
        if 0 < i < nx-1 and 0 < j < ny-1:
            T1 = min((d_f[i-1, j], d_f[i+1, j]))
            T2 = min((d_f[i, j-1], d_f[i, j+1]))
            a = 2
            b = -2 * (T1+T2)
            c = T1**2 + T2**2 - h**2
            T = (-b + (b**2 - 4*a*c)**0.5) / (2*a)

            if T > max((T1, T2)):
                d_f[i, j] = T
            elif T2 > T and T > T1:
                d_f[i, j] = T1 + h
            elif T1 > T and T > T2:
                d_f[i, j] = T2 + h

def upwindCal(f, h, iteration):
    Nx = f.shape[0]
    Ny = f.shape[1]

    d_f = cuda.to_device(f)

    TPBX = 8
    TPBY = 8

    gridDims = ((Nx + TPBX - 1) // TPBX,
                (Ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)
    for i in range(iteration):
        upwind_kernel[gridDims, blockDims](d_f, h)
    return d_f.copy_to_host()
            


def main():
    # question1()
    # question2()
    # question3()
    question4()

if __name__ == '__main__':
    main()