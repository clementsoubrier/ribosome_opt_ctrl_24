#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: c.soubrier
"""

import numpy as np
import scipy.integrate as sci
from multiprocessing import Pool

""" Parameters """


#parameters
N=1000
N_test=100
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




def evol(t,X,U,V,alpha):
    A=np.zeros(2)
    A[0]=V*gammar*X[1]/(KD+X[1])*X[0]-(betar+U)*X[0]
    A[1]=alpha-X[0]*(V*cr*gammar*X[1]/(KD+X[1])+(1-V)*cp*gammap*X[1]/(KD+X[1]))+X[0]*U*cu
    return A


DenominaR_list=[cr*(betar+U1)+cp*(1-V0)*gammap*(betar+U1)/(V0*gammar)-cu*U1,cr*(betar+U0)+cp*(1-V1)*gammap*(betar+U0)/(V1*gammar)-cu*U0]

# Rst0=alpha/DenominaR

Eini_list=[KD/(V0*gammar/(betar+U1)-1),KD/(V1*gammar/(betar+U0)-1)]





def compute(U):
    res =[]
    for V in np.linspace(V0,V1,N_test):
        for Eini in Eini_list:
            for Den in DenominaR_list:
                
                alpha = alpha0
                Rini = alphamax/Den
                #steady states
                Est0 = KD/(V*gammar/(betar+U)-1)
                Rst0 = alpha0/(cr*(betar+U)+cp*(1-V)*gammap*(betar+U)/(V*gammar)-cu*U)
                #simu
                thres_val = np.linalg.norm(np.array([Rini-Rst0,Eini-Est0]))
                if thres_val >=0.1:
                    Z1=sci.solve_ivp(evol,[0,T],[Rini,Eini], method='Radau',t_eval=np.linspace(0,T,N),args=(U,V,alpha))
                    conv_list = (np.absolute(Z1.y[0,:]-Rst0)**2+np.absolute(Z1.y[1,:]-Est0)**2)**0.5
                
                    conv = np.nonzero(conv_list >=(0.05*thres_val))[0][-1]

                res.append(conv*T/N)
                
                alpha = alphamax
                Rini = alpha0/Den
                #steady states
                Est0 = KD/(V*gammar/(betar+U)-1)
                Rst0 = alphamax/(cr*(betar+U)+cp*(1-V)*gammap*(betar+U)/(V*gammar)-cu*U)
                #simu
                thres_val = np.linalg.norm(np.array([Rini-Rst0,Eini-Est0]))
                if thres_val >=0.1:
                    Z1=sci.solve_ivp(evol,[0,T],[Rini,Eini], method='Radau',t_eval=np.linspace(0,T,N),args=(U,V,alpha))
                    conv_list = (np.absolute(Z1.y[0,:]-Rst0)**2+np.absolute(Z1.y[1,:]-Est0)**2)**0.5
                
                    conv = np.nonzero(conv_list >=(0.05*thres_val))[0][-1]

                res.append(conv*T/N)
    return res

res =[]
with Pool(processes=8) as pool:
    for subres in pool.imap_unordered(compute, np.linspace(U0,U1,N_test)):
        res.extend(subres)

    


print(f'Maximum convergence time (95 percent) of {max(res)} hours compared to the assumed period fo {Peri} hours.')
        
