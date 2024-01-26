#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:41:30 2021

@author: pc_clement
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
from matplotlib import colors



""" Parameters """
#fixed parameters
gammar = 15 #/h
gammap =50#/h
betar = 2
KD = 2 #
cu = 7.736
cr = 7.736
cp =2
alpha0 = 50
alphamax = 200
N=1000


#new parameters
gammar = 15#/h
gammap =300#/h
betar = 2
KD = 720000 #
cu = 7736
cr = 7736
cp =300
alpha0 = 50000000
alphamax = 400000000

U1 = 10
V1=0.9
U0=0.5
V0=0.3

P0=(alpha0*gammap*(betar+U1)*(1-V0))/(gammar*V0*cr*(betar+U1)+cp*gammap*(1-V0)*(betar+U1)-cu*U1*gammar*V0)

def Tildef(V):
    num=-V*betar*(gammar*cr*P0-cp*gammap*P0+alphamax*gammap)-betar*(cp*gammap*P0-alphamax*gammap)
    den=cp*gammap*P0-alphamax*gammap+V*(gammar*cr*P0-cp*gammap*P0-gammar*cu*P0+alphamax*gammap)
    if den!=0:
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


Vi=np.linspace(betar/gammar,1,N)
Ui=np.linspace(0,gammar-betar,N)
V_osc=[]
U_osc=[]
V_tilde=[]
U_tilde=[]

for i in range(N):
    if betar<Vi[i]*gammar:
        alpha_osc=(4*betar**2*KD*gammar*Vi[i])/(gammar*Vi[i]-betar)**2
        if alpha0>=alpha_osc:
            Us=(alpha0**0.5*(gammar*Vi[i]-betar)/(4*KD*gammar*Vi[i])**0.5-betar)/(1+alpha0**0.5/(4*KD*gammar*Vi[i])**0.5-gammar*Vi[i]*cu/(cp*gammap*(1-Vi[i])+cr*gammar*Vi[i]))
            V_osc.append(Vi[i])
            U_osc.append(Us)
    if betar+Tildef(Vi[i])<Vi[i]*gammar:
        if V0<=Vi[i]<=V1 and U0<=Tildef(Vi[i])<=U1 :
            V_tilde.append(Vi[i])
            U_tilde.append(Tildef(Vi[i]))
            
solosc=np.ones(len(U_osc))         

Psi_plot=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        Psi_plot[i,j]=Psi(Ui[i],Vi[j],alpha0)

fig, ax = plt.subplots(figsize=(5, 3.5))

c = ax.pcolormesh(Vi, Ui, Psi_plot,norm=colors.PowerNorm(gamma=1), cmap='gist_gray')
line, = ax.plot(V_osc,U_osc,c='green',linestyle='dotted')
line.set_label(r'$U_{\mathrm{osc}}$'+'(V)')

line2, =ax.plot([(betar+Ui[N//3])/gammar,1],[Ui[N//3],Ui[N//3]], c="red" )
line3, =ax.plot([Vi[N//2],Vi[N//2]],[0,gammar*Vi[N//2]-betar], c="blue" )

line4, =ax.plot([Vi[N//2],Vi[N//2],Vi[N//2]+0.2,Vi[N//2]+0.2,Vi[N//2]],[Ui[N//3],Ui[N//3]+1,Ui[N//3]+1,Ui[N//3],Ui[N//3]], c="orange" ,linestyle='dashed')
line4.set_label('Optimization domain')


plt.plot(Vi,Ui,c='black')
# set the limits of the plot to the limits of the data
ax.axis([ Vi.min(), Vi.max(),Ui.min(), Ui.max()])
fig.colorbar(c, ax=ax, extend='max',label=r'$\Psi\;(h^{-1})$')
ax.set_ylabel(r'$U\;(h^{-1})$')
ax.set_xlabel("V")
ax.legend()
plt.tight_layout()

plt.savefig('heatmap_Phase.jpg', format='jpg',dpi=500)
plt.show()




fig=plt.figure(figsize=(5, 3.5))
ax=fig.add_subplot(211, label="1")

ax2=fig.add_subplot(212, label="3")

ax.plot(Vi,Psi_plot[N//3,:], color="red")
plt.tight_layout()
ax.set_xlabel("V")
ax.set_ylabel(r'$\Psi(V)\;(h^{-1})$')
ax.tick_params(axis='x')
ax.tick_params(axis='y')

ax2.plot(Ui,Psi_plot[:,N//2], color="blue")

ax2.set_xlabel(r'$U\;(h^{-1})$') 
ax2.set_ylabel(r'$\Psi(U)\;(h^{-1})$')     
plt.tight_layout()




plt.savefig('heatmap_Phase_UV.eps', format='eps', dpi=20)
plt.savefig('heatmap_Phase_UV.jpg', format='jpg',dpi=200)
plt.show()

