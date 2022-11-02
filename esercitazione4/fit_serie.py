import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import optimize
dati=pd.read_csv('fit_data.csv')
#print(dati)

#grafico dei dati con errori sulle y

x=dati['x']
y=dati['y']
y_err=np.sqrt(y)


plt.errorbar(x,y,yerr=y_err,fmt='o')
plt.xlabel('x')
plt.ylabel('y')
plt.xscale('log')
plt.show()

#definire funzione lognormale per il fit

def f(x,mu,sigma,A):
    return A*np.exp(-((np.log(x)-mu)**2)/(2*sigma**2))/(np.sqrt(2*math.pi)*sigma*x)

#eseguire il fit con la funzione lognormale

params, params_covariance = optimize.curve_fit(f,x,y,sigma=y_err,absolute_sigma=True)
print('params:',params)
print('params covariance:',params_covariance)

#grafico del fit

yexp=f(x,params[0],params[1],params[2])
plt.errorbar(x,y,yerr=y_err,fmt='o')
plt.errorbar(x,yexp)
plt.xlabel('x')
plt.ylabel('y')
plt.xscale('log')
plt.show()

#chi2

chi2= np.sum((yexp-y)**2/y)
print('chi2:', chi2)


    
