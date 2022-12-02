import pandas as pd
import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
import corner

#definisco la funzione per il modello teorico f(E)

def flusso (p, E):
    m, b, alpha, mu, sigma=p
    return m*E+b+alpha*np.exp(-((E-mu)**2)/(2*sigma**2))

#definisco la funzione di loglikelihhod

def lnlike(p, E, f, errf):
    return -0.5*np.sum(((fl-flusso(p,E))/err_fl)**2)

#definisco la funzione con il logaritmo della funzione prior

def lnprior(p):
    m, b, alpha, mu, sigma=p
    if(-2<m<0 and 5<b<15 and -6<alpha<-4 and 4<mu<6 and 0<sigma<1):
        return 0.0
    return -np.inf

#definisco la funzione di probabilità logartmica (logprob=logprior+loglikelihood)

def lnprob(p, E, f, errf):
    lp=lnprior(p)
    if np.isfinite(lp):
        return lp+lnlike(p, E, f, errf)
    return -np.inf

#leggo i dati

dati=pd.read_csv('absorption_line.csv')
print(dati)

energia=dati['E'].values
fl=dati['f'].values
err_fl=dati['ferr'].values

#grafico i dati con il modello teroico

par=np.array([-0.2,10,-5.5,5,0.5])
modello_flusso=flusso(par,energia)

plt.errorbar(energia,fl,err_fl)
plt.plot(energia,modello_flusso)
plt.xlabel('Energia')
plt.ylabel('Flusso')
plt.show()

#definisco il numero di walker

nw=32

#definisco la posizione di partenza per tutti

initial_fl=par
ndim_fl=len(initial_fl)
p0=np.array(initial_fl)+0.1*np.random.randn(nw, ndim_fl)
ndim=len(p0)

#definisco un emcee.EnsembleSampler

sampler=emcee.EnsembleSampler(nw, ndim_fl, lnprob, args=(energia, fl, err_fl))

#Produco il grafico in funzione dei passi

print('Running...')
sampler.run_mcmc(p0, 1000, progress=True)

fig, axes = plt.subplots(ndim_fl, figsize=(10, 9), sharex=True)
samples_fl = sampler.get_chain()

for i in range(ndim_fl):
    ax = axes[i]
    ax.plot(samples_fl[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples_fl))
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.show()

#grafico dati con alcuni campionamenti dei parametri

plt.errorbar(energia, fl, yerr=err_fl, fmt="ok", capsize=0)
plt.xlabel('Energia')
plt.ylabel('flusso')

# Grafico 50 campionamenti posterior .
samples= sampler.flatchain
for s in samples[np.random.randint(len(samples), size=50)]:
    plt.plot(energia, flusso(s, energia), color="orange", alpha=0.3)

#escludo i primi 200 passi

flat_samples= sampler.get_chain(discard=200, thin=15, flat=True)
print(samples_fl.shape)

#grafico dati senza i primi 200 passi

plt.errorbar(energia, fl, yerr=err_fl, fmt="ok", capsize=0)
plt.xlabel('energia')
plt.ylabel('flusso')

# Plot 50 posterior samples.
for s in flat_samples[np.random.randint( len(flat_samples), size=50)]:
    plt.plot(energia, flusso(s, energia), color="orange", alpha=0.3)

#produco un corner plot

fig = corner.corner( flat_samples, color='pink')
plt.show()
