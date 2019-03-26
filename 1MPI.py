import numpy as np
# import scipy as sc
import sympy
from sympy import *
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def fact(n):
    result = 1
    for i in range(1, n + 1):
        result *=i
    return result

def binom_coef(k, n):
    return float(fact(n))/(float(fact(k)) * float(fact(n - k)))
x = Symbol('x')
y = Symbol('y')
input_function = raw_input()
accuracy = int(input())
X0 = int(input())
Y0 = int(input())
x_min = int(input())
x_max = int(input())
y_min = int(input())
y_max = int(input())
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

Teylorr = 0
parts = None
print 'flag 1'
if rank == 0:
    print "aaaaaa"
    parts = np.empty((4, 2))
    parts[0] = [1, accuracy + 1]
    parts[1] = [1, 4]
    parts[2] = [5, 7]
    parts[3] = [8, 10]
    #for i in range(1, size):
    #    parts[i] = [1 + int(accuracy/(size - 1))* (i - 1), int(accuracy/(size - 1) + 1) * i]
    #    print i
    #if size>1:
    #    parts[size-1] = [parts[-1][-1], accuracy + 1]
    # print parts
    #for i in range(1, size):
    #    comm.send(parts[i], dest=i, tag=11)
    comm.send(parts[1], dest=1, tag=11)
    #comm.send(parts[2], dest=2, tag=11)
    #comm.send(parts[3], dest=3, tag=11)
    print parts[0]
    
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