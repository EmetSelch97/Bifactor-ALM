# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:58:53 2024

@author: 888
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as rgt
from scipy.optimize import minimize
from multiprocessing import Pool
import time
from tqdm import tqdm
import pandas as pd

def stan_trans(x,G):
    if len(x) != int(G*(G-1)/2):
        raise NotImplementedError

    h = np.zeros([G,G])
    for i in range(1,G):
        h[0:i,i] = x[int((i-1)*i/2):int((i+1)*i/2)]

    #Z = (np.exp(2*h)-1) / (np.exp(2*h) + 1)
    Z = np.tanh(h)
    
    U = np.zeros([G,G])
    U[0,0] = 1
    U[0,1:G] = Z[0,1:G]
    for i in range(1,G):
        U[i,i] = U[i-1,i]*np.sqrt(1-Z[i-1,i]**2) / Z[i-1,i]
        U[i,(i+1):G] = Z[i,(i+1):G]*U[i-1,(i+1):G]*np.sqrt(1 - Z[i-1,(i+1):G]**2) / Z[i-1,(i+1):G] 
    return U

def cons_fun(x,J,G,Pair_cons):
    p = len(Pair_cons)
    if p != int((G-1)*G/2):
        raise NotImplementedError

    cons_val = np.zeros([p,J])
    for i in range(p):
        Z = Pair_cons[i]
        cons_val[i,:] = x[int(G*(G-1)/2 + Z[0]*J): int(G*(G-1)/2 + (Z[0]+1)*J)] * x[int(G*(G-1)/2 + Z[1]*J): int(G*(G-1)/2 + (Z[1]+1)*J)]
    return cons_val

def con_gd(L,J,G,Pairs,gamma,rho):
    L2 = L**2
    grad_L = np.zeros([J,(G+1)])
    for i in range(len(Pairs)):
        Z = Pairs[i]
        a = Z[0]
        b = Z[1]
        gm = gamma[i,:]
        grad_L[:,a] = grad_L[:,a] + gm*L[:,b] + rho*L[:,a]*L2[:,b]
        grad_L[:,b] = grad_L[:,b] + gm*L[:,a] + rho*L[:,b]*L2[:,a]
    grad = np.zeros(J*(1+G) + int(G*(G-1)/2) + J)
    for i in range(1+G):
        grad[int(G*(G-1)/2 + i*J): int(G*(G-1)/2 + (i+1)*J)] = grad_L[:,i]
    return grad

def gd_rp(x,dU,G):
    if len(x)!=int(G*(G-1)/2):
        raise NotImplementedError
    if len(dU) != int((G-1)*(G+2)/2):
        raise NotImplementedError
    #Z = (np.exp(2*x) - 1) / (np.exp(2*x) + 1) 
    Z = np.tanh(x)
    dZ = 1-Z**2
    A = np.zeros([int(G*(G-1)/2),int((G-1)*(G+2)/2)])

    for i in range(1,G):
        y = Z[int(i*(i-1)/2): int(i*(i+1)/2)]
        sy = np.sqrt(1-y**2)
        inv_sy = 1/(sy + 1e-15)
        #inv_sy = 1/sy
        scaler_full = np.ones_like(y)
        scaler_full[1:] = sy[:-1]
        scaler_full = np.cumprod(scaler_full)

        A_sub = np.zeros([i,i+1])
        for j in range(i):
            A_sub[j,j] = 1
            r_1 = np.ones(i-j)
            r_1[:-1] = y[j+1:]
            r_2 = np.ones(i-j)
            r_2[1:] = sy[j+1:]
            r_2 = np.cumprod(r_2)
            A_sub[j,j+1:] = -r_1*r_2*inv_sy[j]*y[j]
        A_sub = np.diag(scaler_full) @ A_sub
        A[int(i*(i-1)/2): int(i*(i+1)/2), int(i*(i+1)/2-1): int(i*(i+3)/2)] = A_sub   
    gd = (A @ dU)* dZ
    return gd

def alm_gd(x,*args):
    J,G,S,n,gamma,rho,Pairs = args
    if len(x) != J*(1+G) + int(G*(G-1)/2) + J:
        raise NotImplementedError

    Phi = np.zeros([(1+G),(1+G)])
    Phi[0,0] = 1
    Phi[1:(G+1),1:(G+1)] = stan_trans(x[0:int(G*(G-1)/2)],G)

    L = np.zeros([J,(G+1)])
    for i in range(G+1):
        L[:,i] = x[int(G*(G-1)/2 + i*J): int(G*(G-1)/2 + (i+1)*J)]
    
    D = np.diag(x[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)]**2)    
    d = x[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)]
    
    Psi = Phi.T @ Phi
    Cov = L @ (Psi @ L.T) + D

    inv_Cov = np.linalg.solve(Cov,np.identity(J))

    Sdw = inv_Cov @ (S @ (inv_Cov))
    LP = L @ Psi
    PL = Phi @ L.T

    cons_grad = con_gd(L,J,G,Pairs,gamma,rho)
    
    nll_grad = np.zeros_like(x)
    
    dL = n * inv_Cov @ LP - n * Sdw @ LP
    for i in range(G+1):
        nll_grad[int(G*(G-1)/2 + i*J): int(G*(G-1)/2 + (i+1)*J)] = dL[:,i]
    
    dd = n*np.diag( inv_Cov - Sdw ) * d
    nll_grad[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)] = dd

    dPhi = n * PL @ (inv_Cov @ L) - n * PL @ (Sdw @ L)
    dU = np.zeros(int((G-1)*(G+2)/2))
    for i in range(G-1):
        dU[int(i*(i+3)/2) : int(i*(i+5)/2 + 2)] = dPhi[1:(i+3),i+2]

    dphi = gd_rp(x[0:int(G*(G-1)/2)],dU,G)
    nll_grad[0:int(G*(G-1)/2)] = dphi
    full_grad = nll_grad + cons_grad
    return full_grad

def objective_function(x,*args):
    J,G,S,n,gamma,rho,Pairs = args
    if len(x) != J*(1+G) + int(G*(G-1)/2) + J:
        raise NotImplementedError
    Phi = np.zeros([(1+G),(1+G)])
    Phi[0,0] = 1
    #for i in range(G):
    #    Phi[1:(2+i),(1+i)] = x[int((i+1)*i/2):int((i+3)*i/2+1)]
    Phi[1:(G+1),1:(G+1)] = stan_trans(x[0:int(G*(G-1)/2)],G)
    
    L = np.zeros([J,(G+1)])
    for i in range(G+1):
        L[:,i] = x[int(G*(G-1)/2 + i*J): int(G*(G-1)/2 + (i+1)*J)]
    D = np.diag(x[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)]**2)

    Psi = Phi.T @ Phi
    Cov = L @ (Psi @ L.T) + D
    R = np.linalg.solve(Cov,S)

    loss_part = n*np.log(np.linalg.det(2*np.pi*Cov))/2 + n*np.trace(R)/2

    cons_val = cons_fun(x,J,G,Pairs)
    pen_part = np.sum(gamma*cons_val) + 0.5*rho*np.sum(cons_val**2)

    loss = loss_part + pen_part
    return loss

def para_decompose(x,J,G):
    if len(x) != J*(1+G) + int(G*(G-1)/2) + J:
        raise NotImplementedError
    Phi = np.zeros([(1+G),(1+G)])
    Phi[0,0] = 1
    Phi[1:(G+1),1:(G+1)] = stan_trans(x[0:int(G*(G-1)/2)],G)
    L = np.zeros([J,(G+1)])
    for i in range(G+1):
        L[:,i] = x[int(G*(G-1)/2 + i*J): int(G*(G-1)/2 + (i+1)*J)]
    D = np.diag(x[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)]**2)
    
    d = x[int(G*(G-1)/2 + (G+1)*J): int(G*(G-1)/2 + (G+2)*J)]**2
    
    Psi = Phi.T @ Phi
    Cov = L @ (Psi @ L.T) + D
    
    return L,Psi,d,Cov

def Q_mat(J,G):
    Q_true = np.zeros([J,(1+G)])
    Q_true[:,0] = np.ones(J)
    for i in range(G):
        Q_true[int(i*J/G):int((i+1)*J/G),(i+1)] = np.ones(int(J/G))
    return Q_true

def generator(J,G,Q,n):
    #D_true = np.diag(1+2*rgt.rand(J))
    D_true = np.identity(J)
    
    L_true = 2*rgt.rand(J,(1+G))*Q*np.sign(rgt.randn(J,(1+G)))
    L_true[:,0] = np.abs(L_true[:,0])
    L_true = L_true + 0.2* np.sign(L_true)
    
    Phi_true = np.zeros([1+G,1+G])
    Phi_true[0,0] = 1
    tmp = rgt.randn(int(G*(G-1)/2))
    
    Phi_true[1:(G+1),1:(G+1)] = stan_trans(tmp,G)
    Psi_true = Phi_true.T @ Phi_true
    Cov_true = L_true @ (Psi_true @ L_true.T) + D_true
    
    mean = np.zeros(J)
    samples = np.random.multivariate_normal(mean, Cov_true, size=n)
    S = samples.T @ samples /n
    
    return D_true,L_true,Phi_true,Psi_true,Cov_true,S

def nll(Psi,L,D,S,n):
    Cov = L @ (Psi @ L.T) + D
    R = np.linalg.solve(Cov,S)
    nll = n*np.log(np.linalg.det(2*np.pi*Cov))/2 + n*np.trace(R)/2
    return nll

def alm_solve(x_init,J,G,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter):
    p = len(x_init)
    x_old = x_init.copy()
    
    gamma = gamma_0.copy()
    rho = rho_0
       
    L_old,Psi_old,d_old,Cov_old = para_decompose(x_old,J,G)
    cons_val_old = cons_fun(x_old,J,G,Pair)
    
    dist_val = np.max(np.sort(np.abs(L_old[:,1:]),axis=1)[:,-2])
    
    dist_para = np.linalg.norm(x_old)/np.sqrt(p)
    #print(dist_val)
    
    iter_num = 0
    while max(dist_val,dist_para) > tol and iter_num < max_iter:
        result = minimize(objective_function,x_old,args=(J,G,S,n,gamma,rho,Pair),method = 'L-BFGS-B',jac = alm_gd)
        x_new = result.x
        
        dist_para = np.linalg.norm(x_old - x_new)/np.sqrt(p)
        
        cons_val_new =  cons_fun(x_new,J,G,Pair)
        gamma = gamma + rho *cons_val_new
        if np.linalg.norm(cons_val_old,ord = 'fro') > theta*np.linalg.norm(cons_val_new,ord = 'fro'):
            rho = rho*rho_sigma
            
        x_old = x_new.copy()
        L_old,Psi_old,d_old,Cov_old = para_decompose(x_old,J,G)
        dist_val = np.max(np.sort(np.abs(L_old[:,1:]),axis=1)[:,-2])
        #print(dist_val)
        iter_num = iter_num + 1
    return x_old,iter_num,dist_val

def init_value(J,G):
    x = np.zeros(J*(1+G) + int(G*(G-1)/2) + J)
    x[0:int(G*(G-1)/2)] = rgt.randn(int(G*(G-1)/2))
    
    x[J*(1+G) + int(G*(G-1)/2) : J*(1+G) + int(G*(G-1)/2) + J] = 1 + rgt.rand(J)
    
    x[int(G*(G-1)/2): int(G*(G-1)/2) + J] = rgt.rand(J)
    
    x[int(G*(G-1)/2)+J : int(G*(G-1)/2) + J*(1+G)] = 0.1*rgt.randn(G*J)
    
    return x

def init_value_v2(J,G):
    x = np.zeros(3*J + int((G-1)*G/2))
    x[0:int(G*(G-1)/2)] = rgt.randn(int(G*(G-1)/2))
    
    x[J*2 + int(G*(G-1)/2) : J*3 + int(G*(G-1)/2) + J] = 1 + rgt.rand(J)
    
    x[int(G*(G-1)/2): int(G*(G-1)/2) + J] = 1 + rgt.rand(J)
    
    x[int(G*(G-1)/2)+J : int(G*(G-1)/2) + J*2] = 0.1*rgt.randn(J)
    
    return x
        

def uncons_nll_fun(x,*args):
    J,G,S,n,Loc_list,cum_sum_ary = args
    if len(x) != 3*J + int((G-1)*G/2):
        raise NotImplementedError
    Phi = np.zeros([(1+G),(1+G)])
    Phi[0,0] = 1
    Phi[1:(G+1),1:(G+1)] = stan_trans(x[0:int(G*(G-1)/2)],G)

    L = np.zeros([J,(G+1)])
    L[:,0] = x[int(G*(G-1)/2):int(G*(G-1)/2+J)]
    for i in range(1,G+1):
        L[Loc_list[i-1],i] = x[int(G*(G-1)/2+J + cum_sum_ary[i-1]) : int(G*(G-1)/2+J + cum_sum_ary[i])]

    D = np.diag(x[int(G*(G-1)/2 + 2*J): int(G*(G-1)/2 + 3*J)]**2)

    Psi = Phi.T @ Phi
    Cov = L @ (Psi @ L.T) + D
    R = np.linalg.solve(Cov,S)
    
    n = S.shape[0]

    loss = n*np.log(np.linalg.det(2*np.pi*Cov))/2 + n*np.trace(R)/2
    return loss

def para_decompose_v2(x,J,G,Q_list,Q_cumsum):
    Phi = np.zeros([(1+G),(1+G)])
    Phi[0,0] = 1
    Phi[1:(G+1),1:(G+1)] = stan_trans(x[0:int(G*(G-1)/2)],G)

    L = np.zeros([J,(G+1)])
    L[:,0] = x[int(G*(G-1)/2):int(G*(G-1)/2+J)]
    for i in range(1,G+1):
        L[Q_list[i-1],i] = x[int(G*(G-1)/2+J + Q_cumsum[i-1]) : int(G*(G-1)/2+J + Q_cumsum[i])]

    D = np.diag(x[int(G*(G-1)/2 + 2*J): int(G*(G-1)/2 + 3*J)]**2)
    d = x[int(G*(G-1)/2 + 2*J): int(G*(G-1)/2 + 3*J)]**2
    Psi = Phi.T @ Phi
    Cov = L @ (Psi @ L.T) + D
    
    return L,Psi,d,Cov

def uncon_gd(x,*args):
    J,G,S,n,Q_list,Q_cumsum = args
    if len(x) != J*3 + int(G*(G-1)/2):
        raise NotImplementedError
    
    Phi = np.zeros([(1+G),(1+G)])
    Phi[0,0] = 1
    Phi[1:(G+1),1:(G+1)] = stan_trans(x[0:int(G*(G-1)/2)],G)

    L = np.zeros([J,(G+1)])
    L[:,0] = x[int(G*(G-1)/2):int(G*(G-1)/2+J)]
    for i in range(1,G+1):
        L[Q_list[i-1],i] = x[int(G*(G-1)/2+J + Q_cumsum[i-1]) : int(G*(G-1)/2+J + Q_cumsum[i])]

    D = np.diag(x[int(G*(G-1)/2 + 2*J): int(G*(G-1)/2 + 3*J)]**2)
    d = x[int(G*(G-1)/2 + 2*J): int(G*(G-1)/2 + 3*J)]
    Psi = Phi.T @ Phi
    Cov = L @ (Psi @ L.T) + D
    
    inv_Cov = np.linalg.solve(Cov,np.identity(J))

    Sdw = inv_Cov @ (S @ (inv_Cov))
    LP = L @ Psi
    PL = Phi @ L.T

    nll_grad = np.zeros_like(x)
    
    dL = n * inv_Cov @ LP - n * Sdw @ LP
    
    nll_grad[int(G*(G-1)/2) : int(G*(G-1)/2) + J] = dL[:,0]
    for i in range(1,G+1):
        nll_grad[int(G*(G-1)/2+J + Q_cumsum[i-1]) : int(G*(G-1)/2+J + Q_cumsum[i])] = dL[Q_list[i-1],i]
    
    dd = n*np.diag( inv_Cov - Sdw ) * d
    nll_grad[int(G*(G-1)/2 + 2*J): int(G*(G-1)/2 + 3*J)] = dd

    dPhi = n * PL @ (inv_Cov @ L) - n * PL @ (Sdw @ L)
    dU = np.zeros(int((G-1)*(G+2)/2))
    for i in range(G-1):
        dU[int(i*(i+3)/2) : int(i*(i+5)/2 + 2)] = dPhi[1:(i+3),i+2]

    dphi = gd_rp(x[0:int(G*(G-1)/2)],dU,G)
    nll_grad[0:int(G*(G-1)/2)] = dphi
    
    return nll_grad

def Q_rd(L,J,G):
    max_mat = np.max(np.abs(L[:,1:(G+1)]),axis=1).reshape(-1,1) @ np.ones([1,G])
    Q = np.ones([J,(G+1)])
    Q[:,1:(G+1)] = np.where(np.abs(L[:,1:(G+1)]) == max_mat, 1, 0)
    return Q

def refit_process_full(x_init_alm,rf_init,J,g,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter):
    x_alm,iter_alm,dist_val = alm_solve(x_init_alm,J,g,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
    
    L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,g)
    
    Q_alm = Q_rd(L_alm,J,g)
    col_sum = np.sum(Q_alm,axis=0)
    col_sum[0] = 0
    Q_alm_cumsum = np.cumsum(col_sum)
    Q_alm_list = []
    for i in range(1,g+1):
        Q_alm_list.append(np.where(Q_alm[:,i] == 1)[0])
    result = minimize(uncons_nll_fun,rf_init,args=(J,g,S,n,Q_alm_list,Q_alm_cumsum),method = 'L-BFGS-B',jac =uncon_gd)
    
    x_rf = result.x
    L_rf,Psi_rf,d_rf,Cov_rf=para_decompose_v2(x_rf,J,g,Q_alm_list,Q_alm_cumsum)
    
    nl = nll(Psi_rf,L_rf,np.diag(d_rf),S,n)
    return nl

def warm_start(x_init,J,G):
    if len(x_init) != int(J*G + int((G-1)*(G-2)/2) + J):
        raise NotImplementedError
    
    x = np.zeros(J*(1+G) + int(G*(G-1)/2) + J)
    x[0:int((G-1)*(G-2)/2)] = x_init[0:int((G-1)*(G-2)/2)]
    
    x[int((G-1)*(G-2)/2) : int(G*(G-1)/2)] = rgt.randn(G-1)
    
    x[int(G*(G-1)/2) : int(G*(G-1)/2) + J*G] = x_init[int((G-1)*(G-2)/2): int((G-1)*(G-2)/2) + J*G]
    x[int(G*(G-1)/2) + J*G : int(G*(G-1)/2) + J*(G+1)] = 0.1*rgt.randn(J)
    
    x[J*(1+G) + int(G*(G-1)/2): J*(1+G) + int(G*(G-1)/2) + J] = x_init[J*G + int((G-1)*(G-2)/2) : J*G + int((G-1)*(G-2)/2) + J]
    
    return x
if __name__ == '__main__':
    rgt.seed(2024)
    IPIP_E_df = pd.read_csv('ALMBF/IPIP120_UK_adult_2530.csv')

    X_raw_df = IPIP_E_df[IPIP_E_df.columns[2:]]
    Label = IPIP_E_df[IPIP_E_df.columns[1]]

    X_raw = np.array(X_raw_df)
    
    X  = X_raw - np.mean(X_raw,axis=1).reshape(-1,1) @ np.ones([1,X_raw.shape[1]])
    
    J = X.shape[0]
    n = X.shape[1]
    
    S = X @ X.T / n
    
    Epoch= 100
    tol =1e-4
    rho_0 = 1
    rho_sigma = 10
    theta = 0.25
    max_iter = 1000
    low_r = 2
    up_r = 12
    Bic_list = []
    
    for g in tqdm(range(low_r,up_r+1)):
        print(g)
        #bounds = []
        #for i in range(J*(1+g) + int(g*(g-1)/2) + J):
        #    bounds.append((B_low,B_up))
        
        Pair = []
        for i in range(1,(g+1)):
            for j in range(i+1,(g+1)):
                Pair.append(np.array([i,j]))
        
        gamma_0 = np.ones([int((g-1)*g/2),J])
        #time_start = time.time()
        Init_alm = []
        x_list = []
        NLL_list = []
        Niter_list = []
        L_list = []
        Psi_list = []
        D_list = []
        
        best_x = rgt.randn(int(J*g + int((g-1)*(g-2)/2) + J))
        
        for rd in range(2):
            if g ==2:
                Init_alm = []
                for epoch in range(Epoch):
                    Init_alm.append(init_value(J,g))
            else:
                Init_alm = []
                for epoch in range(Epoch):
                    Init_alm.append(warm_start(best_x,J,g))
        #params_list = [(Init_alm[i],J,g,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter,bounds) for i in range(Epoch)]
            params_list = [(Init_alm[i],J,g,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter) for i in range(Epoch)]
        

            with Pool(processes= Epoch) as p:
                results = p.starmap(alm_solve, params_list)
            
            for x_alm,iter_num,dist_val in results:
                L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,g)
                nl = nll(Psi_alm,L_alm,np.diag(d_alm),S,n)
                NLL_list.append(nl)
                Niter_list.append(iter_num)
                L_list.append(L_alm)
                Psi_list.append(Psi_alm)
                D_list.append(d_alm)
                x_list.append(x_alm)
        print(len(NLL_list))
        NLL_list = np.array(NLL_list)
        Niter_list = np.array(Niter_list)
        
        if len(np.where(Niter_list<max_iter)[0])>0:
            best_val = np.min(NLL_list[np.where(Niter_list<max_iter)[0]])
        else:
            best_val = np.min(NLL_list)
            print('Random starts do not converge')
        print(f'The best nll is {best_val}')
        best_loc = np.where(NLL_list == best_val)[0][0]
        L_best = L_list[best_loc]
        L_best_df = pd.DataFrame(L_best)
        L_dir_name = 'ALMBF/UK_adult_2530_L_' + str(int(g)) + '.csv'
        L_best_df.to_csv(L_dir_name)
        
        Psi_best = Psi_list[best_loc]
        Psi_best_df = pd.DataFrame(Psi_best)
        Psi_dir_name = 'ALMBF/UK_adult_2530_Psi_' + str(int(g)) + '.csv'
        Psi_best_df.to_csv(Psi_dir_name)
        
        D_best = D_list[best_loc]
        D_best_df = pd.DataFrame(D_best)
        D_dir_name = 'ALMBF/UK_adult_2530_D_' + str(int(g)) + '.csv'
        D_best_df.to_csv(D_dir_name)
        print(NLL_list)
        print(Niter_list)
        #print(Iter_alm)
        bic =  2*best_val+ (3*J + g*(g-1)/2)*np.log(n)
        Bic_list.append(bic)
        
        #print(x_list[best_loc])
        best_x = x_list[best_loc].copy()
        #print(best_x)
  
        
    print(Bic_list)
    Bic_np = np.array(Bic_list)
    print(min(Bic_np))