#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:00:17 2019

@author: gionuno
"""

from admm import *;
import matplotlib.pyplot  as plt;
import matplotlib.pyplot  as img;
import matplotlib.gridspec as gs;

g_dim = [10,2,2,2];
g_idx = get_tidx(g_dim);
K = g_idx[-1];

I = img.imread("lena.jpg")/255.0;
I = (I[:,:,0]+I[:,:,1]+I[:,:,2])/3.0;
psize = 8;
bsize = 100;

N = psize*psize;

R = 100;
T = 10;

A = rd.rand(N,K);
A = proj_l2(A);

lam = N*1e-2/K;
gam = 1e-1;
XXt = 1e-10*np.eye(K,K);
BXt = np.zeros((N,K));
for r in range(R+1):
    print("Round "+str(r));
    XXt *= gam;
    BXt *= gam;
    leig = esti_leig(np.matmul(A.T,A));
    print(leig);
    for b in range(bsize):
        
        p_i = rd.randint(I.shape[0]-psize);
        p_j = rd.randint(I.shape[1]-psize);

        aux_b = I[p_i:p_i+psize,p_j:p_j+psize].reshape((-1,1));
        #aux_b /= max(1.0,np.linalg.norm(aux_b));
        aux_fx,aux_x = admm(A,aux_b,T,lam,g_idx,g_dim,1e-2*leig);
        print("Patch "+str(b)+" "+str(aux_fx));
        XXt += np.outer(aux_x,aux_x)/bsize;
        BXt += np.outer(aux_b,aux_x)/bsize;
    
    AXXt = np.matmul(A,XXt);
    aux_A = (A*BXt/AXXt);
    for k in range(K):
        aux_A[:,k] /= XXt[k,k];
    A = proj_l2(aux_A);
    
f   = plt.figure();
gs_ = gs.GridSpec(9,20);

a_min = np.min(A);
a_max = np.max(A);
aux_a =  A[:,0];
aux_a -= a_min;
aux_a /= (a_max-a_min+1e-14);
aux_a = aux_a.reshape([psize,psize]);

aux_ax = f.add_subplot(gs_[0,0]);
aux_ax.imshow(aux_a,cmap='gray');
aux_ax.set_xticklabels([]);
aux_ax.set_yticklabels([]);
aux_ax.grid(False);

for i in range(1,11):
    aux_a =  A[:,i];
    aux_a -= a_min;
    aux_a /= (a_max-a_min+1e-14);
    aux_a = aux_a.reshape([psize,psize]);

    aux_ax = f.add_subplot(gs_[1,i-1]);
    aux_ax.imshow(aux_a,cmap='gray');
    aux_ax.set_xticklabels([]);
    aux_ax.set_yticklabels([]);
    aux_ax.grid(False);

plt.show();

for j in range(20):
    aux_a =  A[:,11+j];
    aux_a -= a_min;
    aux_a /= (a_max-a_min+1e-14);
    aux_a = aux_a.reshape([psize,psize]);

    aux_ax = f.add_subplot(gs_[2,j]);
    aux_ax.imshow(aux_a,cmap='gray');
    aux_ax.set_xticklabels([]);
    aux_ax.set_yticklabels([]);
    aux_ax.grid(False);

for i in range(2):
    for j in range(20):
        aux_a =  A[:,31+2*j+i];
        aux_a -= a_min;
        aux_a /= (a_max-a_min+1e-14);
        aux_a = aux_a.reshape([psize,psize]);

        aux_ax = f.add_subplot(gs_[3+i,j]);
        aux_ax.imshow(aux_a,cmap='gray');
        aux_ax.set_xticklabels([]);
        aux_ax.set_yticklabels([]);
        aux_ax.grid(False);

for i in range(4):
    for j in range(20):
        aux_a =  A[:,71+4*j+i];
        aux_a -= a_min;
        aux_a /= (a_max-a_min+1e-14);
        aux_a = aux_a.reshape([psize,psize]);

        aux_ax = f.add_subplot(gs_[5+i,j]);
        aux_ax.imshow(aux_a,cmap='gray');
        aux_ax.set_xticklabels([]);
        aux_ax.set_yticklabels([]);
        aux_ax.grid(False);

plt.show();
