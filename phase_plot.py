# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:07:14 2021

@author: pc_clement
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.integrate as sci
from mpmath import mp, nstr

''' De-comment to get the figure for the oscillating case and non oscillating case'''


#parameters oscillating
'''
far =  mp.mpf(0.5) #between 0 and 1
fap = 1-far
gammar =  mp.mpf(10) #/h
gammap =  mp.mpf(10) #/h
betar =  mp.mpf(1)
KD = mp.mpf( 0.5) #no value
K =  mp.mpf(1)
cr =  mp.mpf(far)
cp =  mp.mpf(fap)
alpha0 = mp.mpf( 1)
U= mp.mpf(2)
N=20
Xmin=0.13
Xmax=0.36
Ymin=0.58
Ymax=1.15
T = 8
stepODE=0.001
mp.dps = 50

Est0=KD/(far*gammar/(betar+U)-1)
Ini_1=np.array([0.35,1,Est0])
Ini_2=np.array([0.19,1,Est0])
Ini_3=np.array([0.14,1,Est0])
'''


#parameters non oscillating
far =  mp.mpf(0.5) #between 0 and 1
fap = 1-far
gammar =  mp.mpf(10) #/h
gammap =  mp.mpf(10) #/h
betar =  mp.mpf(1)
KD = mp.mpf( 0.5) #no value
K =  mp.mpf(1)
cr =  mp.mpf(far)
cp =  mp.mpf(fap)
alpha0 = mp.mpf( 1)
U= mp.mpf(1)
N=20
Xmin=0.08
Xmax=0.58
Ymin=0.21
Ymax=0.75
T =5
stepODE=0.001
mp.dps = 50

Est0=KD/(far*gammar/(betar+U)-1)
Ini_1=np.array([0.57,1,Est0])
Ini_2=np.array([0.19,1,Est0])
Ini_3=np.array([0.10,1,Est0])

# End of parameters to comment/de-comment
DenominaR=cr*(betar+U)/far+cp*gammap*(betar+U)/(far*gammar)-K*U
Est0=KD/(far*gammar/(betar+U)-1)
Rst0=alpha0/DenominaR
print(Est0,Rst0)



#Calculating the eigenvalues of the Jacobian matrix at steady state
b=Rst0*(cp*gammap/(Est0+KD)**2+cr*gammar/(Est0+KD)**2)
c=alpha0*KD*far*gammar/(Est0+KD)**2
if b**2-4*c>=0:
        ra1=(-b-math.sqrt(b**2-4*c))/2
        ra2=(-b+math.sqrt(b**2-4*c))/2
        print("Non oscilliating system, eigenvalues of the matrix:",ra1,ra2)
else:
        ra1=(-b-1j*math.sqrt(-b**2+4*c))/2
        ra2=(-b+1j*math.sqrt(-b**2+4*c))/2
        print("Oscilliating system, eigenvalues of the matrix:",ra1,ra2)



X, Y = np.meshgrid(np.arange(Xmin,Xmax,(Xmax-Xmin)/N), np.arange(Ymin,Ymax,(Ymax-Ymin)/N))

x_shape = X.shape

W = np.zeros(x_shape)
L = np.zeros(x_shape)
Rn1=np.arange(Xmin,Xmax+1/(Xmax-Xmin),(Xmax-Xmin)/N)
Rn3=np.arange(Xmin,Rst0+(Rst0-Xmin)/N,(Rst0-Xmin)/N)
En1=np.zeros(len(Rn1))
En3=np.zeros(len(Rn3))
for i in range(len(Rn1)):
    En1[i]=Est0
for i in range(len(Rn3)):
    En3[i]=Est0
En2=np.arange(Ymin,Ymax,(Ymax-Ymin)/(N*4))
Rn2=np.zeros(len(En2))
for i in range(len(Rn2)):
        Rn2[i]=alpha0/(cr*gammar*En2[i]/(En2[i]+KD)+cp*gammap*En2[i]/(En2[i]+KD)-K*U)



for i in range(x_shape[0]):
    for j in range(x_shape[1]):
        a=far*gammar*Y[i,j]/(KD+Y[i,j])*X[i,j]-(betar+U)*X[i,j]
        b=alpha0-X[i,j]*(cr*gammar*Y[i,j]/(KD+Y[i,j])+cp*gammap*Y[i,j]/(KD+Y[i,j]))+K*U*X[i,j]
        c=a*(Ymax-Ymin)/(Xmax-Xmin)
        W[i,j]=a/(b**2+a**2)**0.2*(Ymax-Ymin)/(Xmax-Xmin)*1.3*2
        L[i,j]=b/(b**2+a**2)**0.2*2

fig, ax = plt.subplots()
q = ax.quiver(X, Y, W, L, scale=30, color='red')
# De-comment the following lines to be able to reproduce fig 5
#plt.plot(Rn3,En3, color='green', linestyle='-', label='S')
plt.plot(Rn1,En1, color='black', linestyle=':')#, label=r'$\mathcal{E}_0$'

plt.plot(Rn2,En2, color='blue', linestyle=':' )#,label=r'$\mathcal{R}_0$'
plt.legend()
#plt.text(0.184, 0.4, r'$\mathcal{Q}_1$')
#plt.text(0.386, 0.37, r'$\mathcal{Q}_2$')
#plt.text(0.18, 0.26, r'$\mathcal{Q}_4$')
#plt.text(0.48, 0.27, r'$\mathcal{Q}_3$')

def f(t,X):
    A=np.zeros(3)
    A[0]=far*gammar*X[2]/(KD+X[2])*X[0]-(betar+U)*X[0]
    A[1]=fap*gammap*X[2]/(KD+X[2])*X[0]
    A[2]=alpha0-X[0]*(cr*gammar*X[2]/(KD+X[2])+cp*gammap*X[2]/(KD+X[2]))+X[0]*U*K
    return A
''' '''
N2=int(math.ceil(T/stepODE)+1)
Z1=sci.solve_ivp(f,[0,T*2],Ini_1, method='RK45',t_eval=np.linspace(0,T*2,N2*2))#[0.35,1,Est0]
plt.plot(Z1.y[0,:],Z1.y[2,:],color='black')
Z2=sci.solve_ivp(f,[0,T*2],Ini_2, method='RK45',t_eval=np.linspace(0,T*2,N2*2))
plt.plot(Z2.y[0,:],Z2.y[2,:],color='green')
Z3=sci.solve_ivp(f,[0,T*2],Ini_3, method='RK45',t_eval=np.linspace(0,T*2,N2*2))#[0.14,1,Est0]
plt.plot(Z3.y[0,:],Z3.y[2,:],color='blue')

plt.xlim(Xmin,Xmax)
plt.ylim(Ymin,Ymax)
plt.xlabel("R")
plt.ylabel("E")


plt.savefig('system_without_osc.eps', format='eps', dpi=1200)#qi_region.eps

plt.show()
''''''
fig3, axs3 = plt.subplots(3, sharex=True)
axs3[0].plot(np.linspace(0,T,N2),Z1.y[0,:N2],color='black')
axs3[0].plot(np.linspace(0,T,N2),Z2.y[0,:N2],color='green')
axs3[0].plot(np.linspace(0,T,N2),Z3.y[0,:N2],color='blue')

P1=np.array([gammap*Z1.y[2,i]/(KD+Z1.y[2,i])*Z1.y[0,i] for i in range(N2)])
P2=np.array([gammap*Z2.y[2,i]/(KD+Z2.y[2,i])*Z2.y[0,i] for i in range(N2)])
P3=np.array([gammap*Z3.y[2,i]/(KD+Z3.y[2,i])*Z3.y[0,i] for i in range(N2)])
axs3[1].plot(np.linspace(0,T,N2),P1/2,color='black')
axs3[1].plot(np.linspace(0,T,N2),P2/2,color='green')
axs3[1].plot(np.linspace(0,T,N2),P3/2,color='blue')

axs3[2].plot(np.linspace(0,T,N2),Z1.y[2,:N2],color='black')
axs3[2].plot(np.linspace(0,T,N2),Z2.y[2,:N2],color='green')
axs3[2].plot(np.linspace(0,T,N2),Z3.y[2,:N2],color='blue')

axs3[0].set( ylabel=r'$R(t)$')
axs3[2].set(xlabel='t', ylabel=r'$E(t)$')
axs3[1].set( ylabel=r'$P(t)$')
plt.savefig('plot2Dnonosc.eps', format='eps', dpi=1200)
plt.show()
