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
import matplotlib as mpl
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import PolyCollection

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










def Tildef2(V,alpha):
    P0=(alpha0*gammap*(betar+U0)*(1-V1))/(gammar*V1*cr*(betar+U0)+cp*gammap*(1-V1)*(betar+U0)-cu*U0*gammar*V1)
    num=-V*betar*(gammar*cr*P0-cp*gammap*P0+alpha*gammap)-betar*(cp*gammap*P0-alpha*gammap)
    den=cp*gammap*P0-alpha*gammap+V*(gammar*cr*P0-cp*gammap*P0-gammar*cu*P0+alpha*gammap)
    if den!=0:
        return num/den
    else:
        return 0

alpha_list=[1410,1450,1500,1600,1750,1900][::-1]
facecolors = plt.colormaps['viridis'](np.linspace(0, 1, len(alpha_list)))[::-1]


fig=plt.figure(figsize=(5, 3.5))


ax =fig.add_subplot()#projection='3d'

ax.set(xlim=((betar+U0)/gammar,V1+0.03), ylim=(U0-0.03,gammar*V1-betar),
       xlabel=r'$V_{\min}$', ylabel=r'$U_{\max}\;(h^{-1})$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
polygones=[]
for i in range(len(alpha_list)):
    val=[]
    absc=[]
    for v in np.linspace(0,V1,N):
        if betar+Tildef2(v,alpha_list[i])<=gammar*v and Tildef2(v,alpha_list[i])>0:
            val.append(Tildef2(v,alpha_list[i]))
            absc.append(v)
        elif betar+Tildef2(v,alpha_list[i])>gammar*v:
            val.append(gammar*v-betar)
            absc.append(v)
            break

    absc=np.array([betar/gammar]+absc)
    val=np.array([0]+val)

    ax.plot(absc,val,color=facecolors[i])
    verts=np.column_stack((absc,val))
    polygones.append(verts)
    

print(polygones[0].shape)


poly = PolyCollection(polygones,facecolors=facecolors,alpha=0.3)
alpha_list=np.array(alpha_list)/1000
ax.add_collection(poly)
ax.plot([betar/gammar,V1],[0,gammar*V1-betar],c='grey',label=r'$\beta_R+U=\gamma_{r,\max}V$')
ax.plot([(betar+U0)/gammar,V1],[U0,U0],c='r',label=r'$U_{\max}\geq U_{\min}$')
ax.plot([V1,V1],[0,gammar*V1-betar],c='k',label=r'$V_{\min}\leq V_{\max}$')


plt.legend()
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.linspace(0, 1, len(alpha_list)),ax=ax,label=r'$\alpha_{\min}\;(10^{8} h^{-1}\mu m^{-3})$')#
cbar.set_ticklabels(alpha_list[::-1])
plt.tight_layout()
plt.savefig('Figure_2d.svg', format='svg')







def comp_U1(U_0,alpha):
    x=-alpha*V1*(1-V0)/(cu*alphamax*V0*(1-V1))*(cr+cp*gammap/gammar*(1-V1)/V1-cu*U_0/(betar+U_0))+(cr+cp*gammap/gammar*(1-V0)/V0)/cu
    return betar*x/(1-x)



alpha_list=np.linspace(1950,2002,30)#
alpha_list= [2002.8,2002,2000,1995,1975,1900,510,508]
print(alpha_list)
col_list=["violet","b","c","g","y","orange","r"]
facecolors = plt.colormaps['viridis'](np.linspace(0, 1, len(alpha_list)))
fig=plt.figure(figsize=(5, 3.5))


ax =fig.add_subplot()

ax.set(xlim=(-0.1,gammar*V0-betar+0.1), ylim=(0,gammar*V0-betar+0.1),
       xlabel=r'$U_{\min}\;(h^{-1})$', ylabel=r'$U,U_{\max}\;(h^{-1})$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

polygones=[]
i=0
for j in range(len(alpha_list)):
    alpha =alpha_list[j]
    val=[]
    absc=[]
    for u in np.linspace(0,gammar*V0-betar,N):
            if u<=comp_U1(u,alpha)<=gammar*V0-betar:

                val.append(comp_U1(u,alpha ))
                absc.append(u)
    absc.append(absc[-1])       
    val.append(gammar*V0-betar)
    
    ax.plot(absc,val,color=facecolors[j])
    absc=np.array(absc+[absc[0],absc[0]])
    val=np.array(val+[gammar*V0-betar,val[0]])
    verts=np.column_stack((absc,val))
    polygones.append(verts)

print(len(polygones))

poly = PolyCollection(polygones,facecolors=facecolors,alpha=0.3)
alpha_list=np.array(alpha_list)/1000
ax.add_collection(poly)

ax.plot([0,gammar*V0-betar],[0,gammar*V0-betar],c='k',label=r'$U_{\min}\leq U$')

ax.plot([0,gammar*V0-betar],[gammar*V0-betar,gammar*V0-betar],c='grey',label=r'$\beta_R+U=\gamma_{r,\max}V_{\min}$')


cmap = mpl.cm.viridis_r
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.linspace(0, 1, len(alpha_list)),ax=ax,label=r'$\alpha_{\min}\;(10^{8} h^{-1}\mu m^{-3})$')#
cbar.set_ticklabels(alpha_list[::-1])

plt.legend()
plt.tight_layout()
plt.savefig('Figure_UU1_2d.svg', format='svg')
plt.show()

