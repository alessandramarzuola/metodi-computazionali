import numpy as np
import pandas as pd
from scipy import constants, fft
import matplotlib.pyplot as plt
from scipy import optimize

#legge il file
data_sample1=pd.read_csv('data_sample1.csv')
data_sample2=pd.read_csv('data_sample2.csv')
data_sample3=pd.read_csv('data_sample3.csv')
t1=data_sample1['time']
t2=data_sample2['time']
t3=data_sample3['time']
m1=data_sample1['meas']
m2=data_sample2['meas']
m3=data_sample3['meas']

#fa il grafico
plt.plot(t1,m1, color='rebeccapurple')
plt.plot(t2,m2, color='springgreen')
plt.plot(t3,m3, color='orange')
plt.show()

#calcola la trasformata di Fourier e fa il grafico dello spettro di potenza
tf1= fft.rfft(m1.values)
tf2= fft.rfft(m2.values)
tf3= fft.rfft(m3.values)
plt.plot(np.absolute(tf1[:tf1.size//2])**2, 'o', markersize=4, color='rebeccapurple')
plt.plot(np.absolute(tf2[:tf2.size//2])**2, 'o', markersize=4, color='springgreen')
plt.plot(np.absolute(tf3[:tf3.size//2])**2, 'o', markersize=4, color='orange')
plt.xscale('log')
plt.yscale('log')
plt.show()

#recupero frequenze
tf1f= 0.5*fft.rfftfreq(tf1.size, d=1)
tf2f= 0.5*fft.rfftfreq(tf2.size, d=1)
tf3f= 0.5*fft.rfftfreq(tf3.size, d=1)

plt.plot(tf1f[:int(tf1.size/2)],np.absolute(tf1[:tf1.size//2])**2, 'o', markersize=4, color='rebeccapurple')
plt.plot(tf2f[:int(tf2.size/2)],np.absolute(tf2[:tf2.size//2])**2, 'o', markersize=4, color='springgreen')
plt.plot(tf3f[:int(tf3.size/2)],np.absolute(tf3[:tf3.size//2])**2, 'o', markersize=4, color='orange')
plt.xscale('log')
plt.yscale('log')
plt.show()

#fa il fit degli spettri e identifica il tipo di rumore
def s(f,beta,a):
    return a/(f**beta)
pstart=np.array([1,10])


params1, params_covariance1 = optimize.curve_fit(s,tf1f[1:int(tf1.size/2)],np.absolute(tf1[1:tf1.size//2])**2)
s1exp=s(tf1f[1:int(tf1.size/2)],params1[0],params1[1])
print('params1:',params1)
print('params covariance1:',params_covariance1)

params2, params_covariance2 = optimize.curve_fit(s,tf2f[1:int(tf2.size/2)],np.absolute(tf2[1:tf2.size//2])**2)
s2exp=s(tf2f[1:int(tf2.size/2)],params2[0],params2[1])
print('params2:',params2)
print('params covariance2:',params_covariance2)

params3, params_covariance3 = optimize.curve_fit(s,tf3f[5:int(tf3.size/2)],np.absolute(tf3[5:tf3.size//2])**2)
s3exp=s(tf3f[5:int(tf3.size/2)],params3[0],params3[1])
print('params3:',params3)
print('params covariance3:',params_covariance3)

#grafico con fit
plt.plot(tf1f[1:int(tf1.size/2)],np.absolute(tf1[1:tf1.size//2])**2, 'o', markersize=4, color='rebeccapurple')
plt.plot(tf1f[1:int(tf1.size/2)], s1exp, 'o', markersize=4, color='red')
plt.xscale('log')
plt.yscale('log')
plt.show()


plt.plot(tf2f[1:int(tf2.size/2)],np.absolute(tf2[1:tf2.size//2])**2, 'o', markersize=4, color='springgreen')
plt.plot(tf2f[1:int(tf2.size/2)], s2exp, 'o', markersize=4, color='red')
plt.xscale('log')
plt.yscale('log')
plt.show()


plt.plot(tf3f[5:int(tf3.size/2)],np.absolute(tf3[5:tf3.size//2])**2, 'o', markersize=4, color='orange')
plt.plot(tf3f[5:int(tf3.size/2)], s3exp, 'o', markersize=4, color='red')
plt.xscale('log')
plt.yscale('log')
plt.show()

