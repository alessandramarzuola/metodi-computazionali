#diffusione 2D simmetrica
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt

#creo funzione che genera passi casuali
def random_walk2d(step, N):
    deltax=np.zeros(N+1)
    deltay=np.zeros(N+1)
    tmpx=0
    tmpy=0
    phi=np.random.uniform(low=0, high=2*np.pi, size=N)
    for i in range(len(phi)):
        tmpx=tmpx+step*np.cos(phi[i])
        tmpy=tmpy+step*np.sin(phi[i])
        deltax[i+1]=tmpx
        deltay[i+1]=tmpy
    return deltax, deltay

for i in range(5):
    x,y= random_walk2d(1,1000)
    plt.plot(x,y)
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'$\Delta y$')
plt.show()

passo10x= np.zeros(1000)
passo10y= np.zeros(1000)
passo100x= np.zeros(1000)
passo100y= np.zeros(1000)
passo1000x= np.zeros(1000)
passo1000y= np.zeros(1000)

for i in range(1000):
    x,y= random_walk2d(1,1000)
    passo10x[i]=x[10]
    passo10y[i]=y[10]
    passo100y[i]=y[100]
    passo100x[i]=x[100]
    passo1000x[i]=x[1000]
    passo1000y[i]=y[1000]

plt.plot(passo1000x, passo1000y,'o',markersize=4,alpha=0.7, color= 'red', label='1000passi')
plt.plot(passo100x, passo100y,'o',markersize=4,alpha=0.7, color= 'slateblue', label='100passi')
plt.plot(passo10x, passo10y,'o',markersize=4,alpha=0.7, color= 'mediumaquamarine', label='10passi')
plt.show()
            




