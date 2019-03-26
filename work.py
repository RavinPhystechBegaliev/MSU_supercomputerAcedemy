import numpy as np
import sympy
from sympy import *
from mpi4py import MPI
import time

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
x_min =1
x_max = 2
y_min =1
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
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() 
Teylorr = 0
parts = None
print 'flag 1'
if rank == 0:
    print "aaaaaa"
    start_time = time.time()
    parts = np.empty((4,2))
    parts[0] = [1, accuracy + 1]
    #parts[1] = [1, accuracy/3]
    #parts[2] = [accuracy/3, 2*accuracy/3]
    #parts[3] = [2*accuracy/3, accuracy + 1]
    parts[1] = [1, accuracy + 1]

    print parts
    print 'flag 0'
    print parts
    # for i in range(1, size):
    #     comm.send(parts[i], dest=i, tag=11)
    comm.send(parts[1], dest=1, tag=11)
    # comm.send(parts[2], dest=2, tag=11)
    # comm.send(parts[3], dest=3, tag=11)
    print parts
elif rank in [1,2,3]:
    data = comm.recv(source=0, tag=11)
    print('rank ={0} , {1}'.format(rank,data))
    i = int(data[0])
    while i <= int(data[1]):
        Teylorr = Teylorr + summa(i)
        i = i +1
    print i
Teylor = comm.reduce(Teylorr, None)
print Teylor
if rank == 0:
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

'''  
if rank in [0,1,2,3]:
    print 'on task', rank
    parts = comm.recv(source=0, tag=11)
    print 'flag'
    #for i in range(int(parts[rank][0]) , int(parts[rank][1])):
    #    Teylorr = Teylorr + summa(i)
#Teylor = comm.reduce(Teylorr, None)
#print Teylor
Teylor = x
print 'aaa'
#if size == 1:
#    Teylor = 0
#    for i in range(1, accuracy + 1):
#        Teylor = Teylor + summa(i)
#    print Teylor + func.subs({x:X0, y:Y0}).n()
'''