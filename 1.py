import numpy as np
# import scipy as sc
import sympy
import time
from sympy import *
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    print 'aaaaa', rank
print comm.Get_size()
def fact(n):
    result = 1
    for i in range(1, n + 1):
        result *=i
    return result

def binom_coef(k, n):
    return float(fact(n))/(float(fact(k)) * float(fact(n - k)))
x = Symbol('x')
y = Symbol('y')
input_function = 'sin(x)*(1+x)**4'
accuracy = 30
X0 = 0
Y0 = 0
x_min = 1
x_max = 2
y_min = 1
y_max = 2
func=sympy.sympify(input_function)
def summa(k):
    sum_part = 0
    i = 1
    # a= diff(func, x, k)
    x_poly = diff(func, x, k).subs({x:X0, y:Y0})
    y_poly = diff(func, y, k).subs({x:X0, y:Y0})
    sum_part =  1.0/fact(k) * (binom_coef(0, k) * (x - X0)**k*x_poly) + 1.0/fact(k) * (binom_coef(0, k)*(y-Y0)**k*y_poly)
    # print sum_part, 'sum'
    k = k 
    while k-1 > 0 and k-i>0:
        # x_poly = diff(func, x, k-i).subs({x:X0, y:Y0})
        # y_poly = diff(func, y, i).subs({x:X0, y:Y0})
        x_poly = diff(func, x, k-i)
        y_poly = diff(x_poly, y, i).subs({x:X0, y:Y0}).n()
        # sum_part = sum_part + 1.0/fact(k) * (binom_coef(i, k) * (x - X0)**(k-i)*(y-Y0)**i*x_poly*y_poly)
        sum_part = sum_part + 1.0/fact(k) * (binom_coef(i, k) * (x - X0)**(k-i)*(y-Y0)**i*y_poly)
        # print 'k-i = ', k-i
        k = k-1
        i = i + 1 
    return sum_part
start_time=time.time()
Teylor = 0
for i in range(1, accuracy + 1):
    Teylor = Teylor + summa(i)
    # print summa(i), i, 'i'
# print Teylor + func.subs({x:X0, y:Y0}).n()
results = Teylor + func.subs({x:X0, y:Y0}).n()
x_massiv = np.linspace(x_min, x_max, 1000)
y_massiv = np.linspace(y_min, y_max, 1000)
func_massiv = np.zeros(len(x_massiv))
for i in range(len(x_massiv)):
    func_massiv[i] = results.subs({x:x_massiv[i], y:y_massiv[i]}).n()
stop_time=time.time() - start_time
print stop_time
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(x_massiv, y_massiv, func_massiv)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
pyplot.show()
    
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
import matplotlib.pyplot as plt
import random

def fun(x, y):
    return x**2 + y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x_massiv, y_massiv)
# zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)

# ax.plot_surface(X, Y, func_massiv)
Z = func_massiv.reshape(1, 1000)
ax.plot_surface(X, Y, Z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()