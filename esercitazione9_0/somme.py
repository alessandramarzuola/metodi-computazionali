import numpy as np
import sys,os

#definisco un funzione che restituisce la somma dei primi n numeri naturali

def num_nat(n):
    num=np.arange(n+1)
    return np.sum(num)

#definisco una funzione che restiruisce la somma delle radici dei primi n numeri naturali

def rad_num_nat(n):
    num=np.arange(n+1)
    rad=np.sqrt(num)
    return np.sum(rad)
