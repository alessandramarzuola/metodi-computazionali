'''
Distanza percorsa in funzione del tempo
1) scaricare il file di dati vel_vs_time.csv contenente dei valori di velocità in funzione del tempo;
2) creare uno script python che:
A. legga il file scaricato;
B. produca un grafico della velocità in funzione del tempo;
C. calcoli la distanza percorsa in funzione del tempo (utilizzando scipy.inetgrate.simpson);
D. produca il grafico della distanza percorsa in funzione del tempo.

SUGGERIMENTO: assicurarsi di comprendere bene il comportamento della funzione scipy.integrate.simpson agli estremi dell'intervallo di intagrazione.
'''
import sys, os
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt

#leggo i dati

vel=pd.read_csv('vel_vs_time.csv')
print(vel.columns)

#faccio il grafico della velocità in funzione del tempo

plt.plot(vel['t'],vel['v'])
plt.xlabel('tempo (s)')
plt.ylabel('velocità (m/s)')
plt.show()

#calcolo la distanza
distanza=np.array([])
for i in range(1,len(vel['v'])+1):
    d=integrate.simpson(vel['v'][:i], dx=0.5)
    distanza=np.append(distanza,d)

#faccio il grafico della distanza in funzione del tempo

plt.plot(vel['t'],distanza)
plt.xlabel('tempo (s)')
plt.ylabel('distanza (m)')
plt.show()

