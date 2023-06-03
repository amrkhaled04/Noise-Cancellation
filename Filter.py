import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import math
from scipy.fftpack import fft 
'''fast fourier transform'''

t = np.linspace(0 , 3 , 12 * 1024)
def j(F,f,ti,T):
    return ((np.sin((2*np.pi*F)*t)+np.sin((2*np.pi*f)*t)) * ((t >= ti) & (t <= (ti+T))))
F = np.array([164.81,220,246.93,130.81,220,164.81])
f = np.array([329.63,440,493.88,261.63,440,329.63])
ti = np.array([0, 0.5, 1, 1.5,2.5,3])
T = np.array([0.3, 0.3, 0.3, 0.3, 0.3,0.3])
i = 0
x = 0
while(i<6):
    x = x + j(F[i],f[i],ti[i],T[i])
    i+=1
#plt.plot(t,x)
#sd.play(x,3*1024)

  
N = 3*1024
'''sample size'''
f = np. linspace(0 , 512 , int(N/2))
xf = fft(x)
xf = 2/N*np.abs(xf[0:np.int(N/2)])
'''convertion'''
fn1,fn2 = np.random.randint(0,512,2)
n = np.sin(2*fn1*np.pi*t)+np.sin(2*fn2*np.pi*t)
xn = x+n
xnf = fft(xn)
xnf = 2/N*np.abs(xnf[0:np.int(N/2)])
z = np.where(xnf>math.ceil(np.max(x)))
'''Tuple'''
i1 = z[0][0]
i2 = z[0][1]
rem1 = int(f[i1])
rem2 = int(f[i2])
xFiltered = xn - (np.sin(2*rem1*np.pi*t)+np.sin(2*rem2*np.pi*t))
sd.play(xFiltered, 4*1024)
xFiltered_f = fft(xFiltered)
xFiltered_f = 2/N*np.abs(xFiltered_f[0:np.int(N/2)])

plt.figure()
plt.subplot(3,1,1)
plt.plot(t,x)
plt.subplot(3,1,2)
plt.plot(t,xn)
plt.subplot(3,1,3)
plt.plot(t,xFiltered)

plt.figure()
plt.subplot(3,1,1)
plt.plot(f,xf)
plt.subplot(3,1,2)
plt.plot(f,xnf)
plt.subplot(3,1,3)
plt.plot(f,xFiltered_f)