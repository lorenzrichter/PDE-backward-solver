#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:32:29 2019

@author: sallandt

Calcululates H1 orthonormal polynomials of order min_pol_deg to max_pol_deg on the interval (a, b). Useful for TT Ansatz
"""

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

def calc_pol(a,b, max_pol_deg, min_pol_deg=0):
    _str = 'H2'
    # print('orth. w.r.t.', _str)
    def eval_SP(p1, p2, _a, _b):
        dp1 = np.polyder(p1)
        dp2 = np.polyder(p2)
    
        ddp1 = np.polyder(dp1)
        ddp2 = np.polyder(dp2)

        int_p = np.polyint(p1*p2)
        int_dp = np.polyint(dp1*dp2)
        int_ddp = np.polyint(ddp1*ddp2)
        if _str == 'H1':
            return int_p(_a) - int_p(_b) + int_dp(_a) - int_dp(_b)
        elif _str == 'H2':
            return int_p(_a) - int_p(_b) + int_dp(_a) - int_dp(_b) + int_ddp(_a) - int_ddp(_b) 
        else:
            return int_p(_a) - int_p(_b)
        return 0
    #    return int_dp(_a) - int_dp(_b)

    
    
    #max_pol_deg = 16
    
    pol_deg = max_pol_deg + 1
    polynomials = []
    for i0 in range(0, pol_deg):
        polynomials.append([1])
    
    for i0 in range(0,pol_deg):
        for i1 in range(min_pol_deg):
            polynomials[i0].append(0)
    #    polynomials[i0].append(0)
        for i1 in range(0, i0):
            polynomials[i0].append(0)
    
    pol = []
    
    for i0 in range(pol_deg):
        pol.append(np.poly1d(polynomials[i0]))
    
    for i0 in range(pol_deg):
        temp_pol = 1*pol[i0]
        for i1 in range(i0):
            temp_pol = temp_pol - eval_SP(pol[i0], pol[i1], a, b) / eval_SP(pol[i1], pol[i1], a, b) * pol[i1]    
        temp_pol = temp_pol / np.sqrt(eval_SP(temp_pol, temp_pol, a, b))
        pol[i0] = 1* temp_pol
    for i0 in range(len(pol)):
        pol[i0] /= pol[0](0)
    #pol[0][0] = 1
    dpol = []
    
    # hier sind die listen der koeffizienten!
    for i0 in range(pol_deg):
        dpol.append(np.polyder(pol[i0]))
    return pol, dpol
