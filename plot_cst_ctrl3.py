#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:26:29 2022

@author: c.soubrier
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
import scipy.optimize as opti
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter


""" Parameters """


#parameters
N=1000
gap=0.01
Peri=24 #oscillation period
T =72 #h
gammar = 15 #/h
gammap =300#/h
betar = 0.006
KD = 0.72 #
cu = 77.36*0.9
cr = 77.36
cp =3
alpha0 = 2000
alphamax = 2500
U1 =8
V1=0.791
U0=4
V0=0.6
guess=0.791

DenominaR=cr*(betar+U1)+cp*(1-V0)*gammap*(betar+U1)/(V0*gammar)-cu*U1
Est0=KD/(V0*gammar/(betar+U1)-1)
Rst0=alpha0/DenominaR

Rini=Rst0
Eini=Est0

P0=(alpha0*gammap*(betar+U1)*(1-V0))/(gammar*V0*cr*(betar+U1)+cp*gammap*(1-V0)*(betar+U1)-cu*U1*gammar*V0)
def Tildef(V):
    num=-V*betar*(gammar*cr*P0-cp*gammap*P0+alphamax*gammap)-betar*(cp*gammap*P0-alphamax*gammap)
    den=cp*gammap*P0-alphamax*gammap+V*(gammar*cr*P0-cp*gammap*P0-gammar*cu*P0+alphamax*gammap)
    if den!=0 and num/den>0:
        return num/den
    else:
        return 0

def Psi(U,V,alpha):
    if betar+U<V*gammar:
        alpha_osc=(4*betar**2*KD*gammar*V)/(gammar*V-betar)**2
        R_star=alpha/(cr*(betar+U)+cp*gammap*(betar+U)*(1-V)/(V*gammar)-cu*U)
        E_star=KD/(V*gammar/(betar+U)-1)
        if alpha>=alpha_osc:
            U_osc=(alpha**0.5*(gammar*V-betar)/(4*KD*gammar*V)**0.5-betar)/(1+alpha**0.5/(4*KD*gammar*V)**0.5-gammar*V*cu/(cp*gammap*(1-V)+cr*gammar*V))
            if U<=U_osc:
                part=1-4*alpha*V*gammar*(KD+E_star)**2/(KD*R_star**2*(cp*gammap*(1-V)+cr*gammar*V)**2)
                return R_star*KD*(cp*gammap*(1-V)+cr*gammar*V)/(2*(KD+E_star)**2)*(1-part**0.5)
            else:
                return R_star*KD*(cp*gammap*(1-V)+cr*gammar*V)/(2*(KD+E_star)**2)
        else:
            return R_star*KD*(cp*gammap*(1-V)+cr*gammar*V)/(2*(KD+E_star)**2)
        
        
#evolution function of the differential equation
def evol(t,X,U_0,U_1,V_0,V_1,alpha_min,alpha_max,T_t):
    A=np.zeros(2)
    b=alpha_max
    U=U_1
    V=V_0
    if t%Peri>Peri*T_t:
        b=alpha_min
    else :
        V=Vtrans
        U=Tildef(V)
    A[0]=V*gammar*X[1]/(KD+X[1])*X[0]-(betar+U)*X[0]
    A[1]=b-X[0]*(V*cr*gammar*X[1]/(KD+X[1])+(1-V)*cp*gammap*X[1]/(KD+X[1]))+X[0]*U*cu
    return A




#evolution fuction with a fixed U=U0

def optifunc(V):
    if V>V1 or V<V0:
        return 0
    elif Tildef(V)< U0 or Tildef(V)>U1 : #
        return 0
    else:
        return -Psi(Tildef(V),V,alpha0)


Vtrans=opti.minimize(optifunc,guess,method='Powell').x
print(Vtrans,Tildef(Vtrans))
plt.figure()
plt.plot(np.linspace(V0,V1,1000),[Tildef(V) for V in np.linspace(V0,V1,1000)])

plt.figure()
plt.plot(np.linspace(V0,V1,1000),[optifunc(V) for V in np.linspace(V0,V1,1000)])







Vi=np.linspace(betar/gammar,1,N)
Ui=np.linspace(0,gammar-betar,N)
V_osc=[]
U_osc=[]
V_tilde=[]
U_tilde=[]





Z1=sci.solve_ivp(evol,[0,T],[Rini,Eini], method='Radau',t_eval=np.linspace(Peri,T,N-int(Peri/T*N)),args=(U0,U1,V0,V1,alpha0,alphamax,0.5))
Protprod=np.zeros(N-int(Peri/T*N))
Cont_U=np.zeros(N-int(Peri/T*N))
Cont_V=np.zeros(N-int(Peri/T*N))
alpha=np.zeros(N-int(Peri/T*N))
print(Z1)
for i in range(N-int(Peri/T*N)):
    U=U1
    V=V0
    if Z1.t[i]%Peri>Peri*0.5:
        alpha[i]=alpha0
        
    else:
        alpha[i]=alphamax
        V=Vtrans
        U=Tildef(V)
    Cont_V[i]=V
    Cont_U[i]=U
    Protprod[i]=(1-V)*gammap*Z1.y[1,i]/(KD+Z1.y[1,i])*Z1.y[0,i]
    
 
    
 
    
U0cst=4


def evol2(t,X,U_0,U_1,V_0,V_1,alpha_min,alpha_max,T_t):
    A=np.zeros(2)
    b=alpha_max
    if t%Peri>Peri*T_t:
        b=alpha_min
        U=U_0
        V=V_0
    else :
        V=Vtrans2
        U=U_0
    A[0]=V*gammar*X[1]/(KD+X[1])*X[0]-(betar+U)*X[0]
    A[1]=b-X[0]*(V*cr*gammar*X[1]/(KD+X[1])+(1-V)*cp*gammap*X[1]/(KD+X[1]))+X[0]*U*cu
    return A

P2=(alpha0*gammap*(betar+U0cst)*(1-V0))/(gammar*V0*cr*(betar+U0cst)+cp*gammap*(1-V0)*(betar+U0cst)-cu*U0cst*gammar*V0)
def Tildef2(V):
    num=-V*betar*(gammar*cr*P2-cp*gammap*P2+alphamax*gammap)-betar*(cp*gammap*P2-alphamax*gammap)
    den=cp*gammap*P2-alphamax*gammap+V*(gammar*cr*P2-cp*gammap*P2-gammar*cu*P2+alphamax*gammap)
    if den!=0 and num/den>0:
        return num/den
    else:
        return 0



def optifunc2(V):
    return abs(Tildef2(V)-U0cst)

Vtrans2=opti.minimize(optifunc2,0.7,method='Powell').x
plt.figure()
plt.plot(np.linspace(V0,V1,50),[optifunc2(V) for V in np.linspace(V0,V1,50)])



Z2=sci.solve_ivp(evol2,[0,T],[Rini,Eini], method='Radau',t_eval=np.linspace(Peri,T,N-int(Peri/T*N)),args=(U0cst,U0cst,V0,V1,alpha0,alphamax,0.5))

Protprod2=np.zeros(N-int(Peri/T*N))
Cont_U2=np.zeros(N-int(Peri/T*N))
Cont_V2=np.zeros(N-int(Peri/T*N))

for i in range(N-int(Peri/T*N)):
    if Z2.t[i]%Peri>Peri*0.5:
        alpha[i]=alpha0
        U=U0cst
        V=V0
    else:
        alpha[i]=alphamax
        V=Vtrans2
        U=Tildef2(V)
    Cont_V2[i]=V
    Cont_U2[i]=U
    Protprod2[i]=(1-V)*gammap*Z2.y[1,i]/(KD+Z2.y[1,i])*Z2.y[0,i]



fig = plt.figure()
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
 box.width, box.height * 0.9])
ax.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Z1.y[1,:], label=r'$E(10^6AA/\mu m^3)$', color="g")
ax.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),alpha/50, label=r'$\alpha(5\cdot 10^6AA/h/\mu m^3)$', color="k")
ax.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Cont_U, label='U', color="r")
ax.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Cont_V, label='V ', color="b")
ax.set_xlabel('Time(h)')

ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=4)






fig = plt.figure()
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
 box.width, box.height * 0.9])

plt.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Z1.y[0,:]*10, label=r'$R(10^{3}/\mu m^{3})$', color="b")
plt.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),alpha/10, label=r'$\alpha(10^6AA/h/\mu m^3)$', color="k")
plt.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Protprod, label=r'$\rho(10^{4}/h/\mu m^{3})$', color="r")
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=3)
ax.annotate(r'$P_{score}$='+str(round(sum(Protprod)))+r'$\cdot10^{4}/\mu m^{3}$', [0,0.3])
ax.set_xlabel('Time(h)')




plt.rcParams.update({'font.size': 20})

fig, axs = plt.subplots(2, 2,figsize=(25, 15),sharex=True)

axs[0,0].plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),alpha/10,  color="k")   
axs[0,0].set_ylabel(r'$\alpha\;(10^6 h^{-1}\mu m^{-3})$')


axs[1,1].plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Cont_U, color="r")
axs[1,1].set_ylabel(r'$U\;(h^{-1})$')
axs[1,1].set_ylim(3.5, 9)
axs[1,1].annotate(r'$U_s$', [0,5])
axs[1,1].annotate(r'$V_s$', [0,8.2])
axs[1,1].annotate(r'$U_{\max}$', [13,7.5])
axs[1,1].annotate(r'$V_{\min}$', [13,4])
axs[1,1].tick_params(axis='y', labelcolor='r')
ax2 = axs[1,1].twinx()  
ax2.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Cont_V,  color="b")
ax2.tick_params(axis='y', labelcolor="b")
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.set_ylabel(r'$V$')
axs[1,0].set_xlabel('Time(h)')
axs[1,1].set_xlabel('Time(h)')

axs[0,1].tick_params(axis='y', labelcolor="r")
axs[0,1].plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Z1.y[0,:], color="r")
axs[0,1].set_ylim(0, 30)
axs[0,1].set_ylabel(r'$R\;(10^{3}\;\mu m^{-3})$')
ax2 = axs[0,1].twinx() 
ax2.tick_params(axis='y', labelcolor="b")
ax2.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Z1.y[1,:]*10,  color="b")
ax2.set_ylabel(r'$E\;(10^5\; \mu m^{-3})$')


axs[1,0].plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Protprod/100,  color="k")#
axs[1,0].set_ylim(2, 10)
axs[1,0].set_ylabel(r'$\rho\;(10^{6}\; h^{-1}\mu m^{-3})$')

plt.tight_layout()
plt.savefig('Figurcst_final.eps', format='eps',bbox_inches='tight',dpi=1200)
plt.show()

plt.rcParams.update({'font.size': 10})

print(P0,P2,Rst0*gammap*Est0/(KD+Est0)*(1-V0))
print(Psi(U0,V1,alpha0),Psi(U1,V0,alpha0),Psi(U0,V1,alphamax),Psi(U1,V0,alphamax))



















def evol_smooth(t,X,U_0,U_1,V_0,V_1,alpha_min,alpha_max,T_t,smooth_par, logis_par):
    A=np.zeros(2)
    b,U,V = smooth_func(t,T_t,alpha_min,U_1,V_0,alpha_max,smooth_par, logis_par)
    A[0]=V*gammar*X[1]/(KD+X[1])*X[0]-(betar+U)*X[0]
    A[1]=b-X[0]*(V*cr*gammar*X[1]/(KD+X[1])+(1-V)*cp*gammap*X[1]/(KD+X[1]))+X[0]*U*cu
    return A




def smooth_func(t,T_t,alpha_min,U_1,V_0,alpha_max,smooth_par, logis_par):
    if t%Peri>Peri*3/4:
        b=(alpha_max-alpha_min)/(1+np.exp(logis_par*(Peri-t%Peri)))+alpha_min
        U=(Tildef(Vtrans)-U_1)/(1+np.exp(logis_par*(Peri-t%Peri)))+U_1
        V=(Vtrans-V_0)/(1+np.exp(logis_par*(Peri-t%Peri)))+V_0
    elif 0<= t%Peri<Peri*1/4:
        b=(alpha_max-alpha_min)/(1+np.exp(-logis_par*(t%Peri)))+alpha_min
        U=(Tildef(Vtrans)-U_1)/(1+np.exp(-logis_par*(t%Peri)))+U_1
        V=(Vtrans-V_0)/(1+np.exp(-logis_par*(t%Peri)))+V_0

    else:
        b=(alpha_min-alpha_max)/(1+np.exp(-logis_par*(t%Peri-Peri/2)))+alpha_max
        U=(U_1-Tildef(Vtrans))/(1+np.exp(-logis_par*(t%Peri-Peri/2)))+Tildef(Vtrans)
        V=(V_0-Vtrans)/(1+np.exp(-logis_par*(t%Peri-Peri/2)))+Vtrans
    return b,U,V





Vtrans=opti.minimize(optifunc,guess,method='Powell').x
print(Vtrans,Tildef(Vtrans))




Vi=np.linspace(betar/gammar,1,N)
Ui=np.linspace(0,gammar-betar,N)
V_osc=[]
U_osc=[]
V_tilde=[]
U_tilde=[]





Z1=sci.solve_ivp(evol_smooth,[0,T],[Rini,Eini], method='Radau',t_eval=np.linspace(Peri,T,N-int(Peri/T*N)),args=(U0,U1,V0,V1,alpha0,alphamax,0.5,0.07,3))
Protprod=np.zeros(N-int(Peri/T*N))
Cont_U=np.zeros(N-int(Peri/T*N))
Cont_V=np.zeros(N-int(Peri/T*N))
alpha=np.zeros(N-int(Peri/T*N))
print(Z1)
for i in range(N-int(Peri/T*N)):
    b,U,V = smooth_func(Z1.t[i],0.5,alpha0,U1,V0,alphamax,0.07,3)
    alpha[i]=b
    Cont_V[i]=V
    Cont_U[i]=U
    Protprod[i]=(1-V)*gammap*Z1.y[1,i]/(KD+Z1.y[1,i])*Z1.y[0,i]
    
 
    
 
    



fig = plt.figure()
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
 box.width, box.height * 0.9])
ax.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Z1.y[1,:], label=r'$E(10^6AA/\mu m^3)$', color="g")
ax.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),alpha/50, label=r'$\alpha(5\cdot 10^6AA/h/\mu m^3)$', color="k")
ax.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Cont_U, label='U', color="r")
ax.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Cont_V, label='V ', color="b")
ax.set_xlabel('Time(h)')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=4)





fig = plt.figure()
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
 box.width, box.height * 0.9])
plt.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Z1.y[0,:]*10, label=r'$R(10^{3}/\mu m^{3})$', color="b")
plt.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),alpha/10, label=r'$\alpha(10^6AA/h/\mu m^3)$', color="k")
plt.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Protprod, label=r'$\rho(10^{4}/h/\mu m^{3})$', color="r")#str(round(sum(Protprod)))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=3)
ax.annotate(r'$P_{score}$='+str(round(sum(Protprod)))+r'$\cdot10^{4}/\mu m^{3}$', [0,0.3])
ax.set_xlabel('Time(h)')



plt.rcParams.update({'font.size': 20})

fig, axs = plt.subplots(2, 2,figsize=(25, 15),sharex=True)

axs[0,0].plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),alpha/10,  color="k")   #label=r'$\alpha\;(10^6AA/h/\mu m^3)$',
axs[0,0].set_ylabel(r'$\alpha\;(10^6 h^{-1}\mu m^{-3})$')

axs[1,1].plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Cont_U, color="r")# label=r'$U\;(h^{-1})$',
axs[1,1].set_ylabel(r'$U\;(h^{-1})$')
axs[1,1].set_ylim(3.5, 9)
axs[1,1].annotate(r'$U_s$', [0.5,5])
axs[1,1].annotate(r'$V_s$', [0.5,8.2])
axs[1,1].annotate(r'$U_{\max}$', [13.5,7.5])
axs[1,1].annotate(r'$V_{\min}$', [13.5,4])
axs[1,1].tick_params(axis='y', labelcolor='r')
ax2 = axs[1,1].twinx()  
ax2.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Cont_V,  color="b")#label=r'$V$',
ax2.tick_params(axis='y', labelcolor="b")
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.set_ylabel(r'$V$')
axs[1,0].set_xlabel('Time(h)')
axs[1,1].set_xlabel('Time(h)')

axs[0,1].tick_params(axis='y', labelcolor="r")
axs[0,1].plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Z1.y[0,:], color="r")
axs[0,1].set_ylim(0, 30)
axs[0,1].set_ylabel(r'$R\;(10^{3}\;\mu m^{-3})$')
ax2 = axs[0,1].twinx() 
ax2.tick_params(axis='y', labelcolor="b")
ax2.plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Z1.y[1,:]*10,  color="b")
ax2.set_ylabel(r'$E\;(10^5\; \mu m^{-3})$')


axs[1,0].plot(np.linspace(0,T-Peri,N-int(Peri/T*N)),Protprod/100,  color="k")

axs[1,0].set_ylim(2, 10)
axs[1,0].set_ylabel(r'$\rho\;(10^{6}\; h^{-1}\mu m^{-3})$')

plt.tight_layout()
plt.savefig('Figurcst_final_smooth.eps', format='eps',bbox_inches='tight',dpi=1200)
plt.show()

plt.rcParams.update({'font.size': 10})

print(P0,P2,Rst0*gammap*Est0/(KD+Est0)*(1-V0))
print('Minimal value of Psi is min of :',Psi(U0,V1,alpha0),Psi(U1,V0,alpha0),Psi(U0,V1,alphamax),Psi(U1,V0,alphamax))

