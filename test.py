import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
#N = [100,1000,5000, 10000]
n = 100
# send
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() 
ans = 0
data = None
if rank == 0:
    data = np.random.random(10)
    print("sum = {}".format(data.sum()))
    comm.send(data[0:4], dest=1, tag=11)
    comm.send(data[4:7], dest=2, tag=11)
    comm.send(data[7:10], dest=3, tag=11)
elif rank in [1,2,3]:
    print 'on task',rank,'before recv:   data = ',data
    data = comm.recv(source=0, tag=11)
    ans += data.sum()
    print 'on task',rank,'after recv:    data = ',data
answer = comm.reduce(ans, None)
if rank == 0:
    print(answer)

#if rank == 0:
#    start_time = time.time()
#    print("n = {}".format(n))
#    sum_f = Z.sum()
#    print("sum_forward = {}".format(sum_f))
#    print("--- time 1 thread = %s seconds ---" % (time.time() - start_time))
#    time_f.append(time.time() - start_time)


'''
start_time = time.time()
print("n*rank/4 =  {}, n*(rank+1)/4 = {}".format((n*rank/4),(n*(rank+1)/4)-1))
#print(Z)
sum_par = Z[int((n*rank/4)):int((n*(rank+1)/4)-1)].sum()
print("sum_par = {}, rank = {}".format(sum_par,rank))
print("--- time = %s seconds ---" % (time.time() - start_time))
'''