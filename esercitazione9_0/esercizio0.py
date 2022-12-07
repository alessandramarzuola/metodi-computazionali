#creo uno script che importa il modulo somme creato precedentemente e ne utilizza le funzioni

import sys, os
import somme

a=10
print('Somma dei primi', a,'numeri naturali:', somme.num_nat(a))
print('Somma delle radici dei primi', a,' numeri naturali: ', somme.rad_num_nat(a))

