#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 03:31:15 2019

@author: gionuno
"""

import numpy as np;
import numpy.random as rd;


def prox_dlinf_n(s,v,dn,lam,t_idx,t_dim,i=0,d=0):
    idx = 0;
    if d>0:
        idx = t_idx[d-1];
    idx = idx + i;
    a = np.abs(v[idx]);
    if d<len(t_dim):
        for j in range(t_dim[d]):
            k = t_dim[d]*i+j;
            a += prox_dlinf_n(s,v,dn,lam,t_idx,t_dim,k,d+1);
    dn[idx] = (a > s*lam)*(1.0-s*lam/(a+1e-14));
    return dn[idx]*a;

def prox_dlinf_s(s,v,dn,t_idx,t_dim,i=0,d=0):
    idx = 0;
    if d>0:
        idx = t_idx[d-1];
    idx = idx + i;
    t       = s*dn[idx];
    v[idx] *= t;
    if d<len(t_dim):
        for j in range(t_dim[d]):
            k = t_dim[d]*i+j;
            prox_dlinf_s(t,v,dn,t_idx,t_dim,k,d+1);

def prox_dlinf(s,u,lam,t_idx,t_dim):
    v = u*(u>0.0);
    dn = np.zeros(v.shape);
    prox_dlinf_n(  s,v,dn,lam,t_idx,t_dim);
    prox_dlinf_s(1.0,v,dn,    t_idx,t_dim);
    return v;

def proj_l2(A):
    B = (A>0)*A;
    for k in range(B.shape[1]):
        l2 = np.linalg.norm(B[:,k]);
        if l2 > 1.0:
            B[:,k] /= l2;
    return B;

def esti_leig(AtA,I=10):
    
    u = rd.rand(AtA.shape[1],1);
    m = -1e3;
    for i in range(I):
        AtAu = np.matmul(AtA,u);
        m = np.max(np.abs(AtAu));
        u = AtAu/m;
    return m;

def admm(A,b,T,lam,t_idx,t_dim,p0):
    AtA = np.matmul(A.T,A);
    Atb = np.matmul(A.T,b);
    K = Atb.shape[0];

    x = rd.rand(K,1);
    u = np.zeros((K,1));
    z = np.zeros((K,1));
    p = p0;
    
    f_x  = 0.5*np.linalg.norm(np.matmul(A,x)-b)**2.0;
    
    f_xb = f_x;
    xb = 1.0*x;
    ub = 1.0*u;
    zb = 1.0*z;
    
    for t in range(T+1):
        
        gp = np.matmul(AtA,x)+p*x+u*(u>0.0);
        gn = Atb+p*z-u*(u<0.0);
        xn = x*(gn+1e-14)/(gp+1e-14);#np.linalg.solve(AtA+p*np.eye(K),Atb+p*z-u);
        zn = prox_dlinf(1./p,xn-u/p,lam,t_idx,t_dim);
        un = u+p*(xn-zn);
        f_xn = 0.5*np.linalg.norm(np.matmul(A,xn)-b)**2.0;
        
        x = xn;
        z = zn;
        u = un;
        f_x = f_xn;
        
        if 1e4 <= f_x:
            x = xb + 1e-1*rd.rand(K,1);
            u = np.zeros((K,1));
            z = np.zeros((K,1));
            f_x = 0.5*np.linalg.norm(np.matmul(A,x)-b)**2.0;
        
        if f_x <= f_xb:
            f_xb = f_x;
            xb = np.copy(x);
            ub = np.copy(u);
            zb = np.copy(z);
        
        #if t % 10 == 0:
        #   print(str(t)+" "+str(f_x)+" "+str(f_xb));
        

    return f_xb,xb;

def get_tidx(t_dim):
    t_idx = (len(t_dim)+1)*[0];
    t_idx[0] = 1;
    for d in range(1,len(t_idx)):
        p = 1;
        for e in range(d):
            p *= t_dim[e];
        t_idx[d] = t_idx[d-1] + p;
    return t_idx;

