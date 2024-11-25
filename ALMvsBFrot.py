# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 11:50:47 2024

@author: 888
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as rgt
from scipy.optimize import minimize
from tqdm import tqdm

from Permutation import BF_permutation,Compare_Q,Average_Cr,Exact_Cr
from multiprocessing import Pool

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects import r, globalenv,numpy2ri

def othog_gen(G):
    x = rgt.randn((1+G),(1+G))
    y = (x+x.T)/2
    Q,R = np.linalg.qr(y)
    return Q

def efa_decom(x,J,C):
    if len(x) != J*(C+1)- int(C*(C-1)/2):
        raise  NotImplementedError
    d = x[J*C-int(C*(C-1)/2) : J*(C+1)-int(C*(C-1)/2)]
    D = np.diag(d**2)
    L = np.zeros([J,C])
    for i in range(C):
        L[i:,i] = x[int(i*J - i*(i-1)/2) : int((i+1)*J - i*(i+1)/2)]

    return L,D   

def efa_nll(x,*args):
    J,C,S,n = args
    L,D = efa_decom(x,J,C)

    Cov = L @ L.T + D
    R = np.linalg.solve(Cov,S)

    loss = n*np.log(np.linalg.det(2*np.pi*Cov))/2 + n*np.trace(R)/2
    return loss

def efa_grad(x,*args):
    J,C,S,n = args
    L,D = efa_decom(x,J,C)
    
    Cov = L @ L.T + D
    inv_Cov = np.linalg.solve(Cov,np.identity(J))

    Sdw =  inv_Cov @ (S @ inv_Cov)

    diff = inv_Cov -Sdw
    
    grad_L = n*diff @ L
    grad_D = n*diff/2

    grad_x = np.zeros_like(x)
    grad_x[J*C-int(C*(C-1)/2) : J*(C+1)-int(C*(C-1)/2)] = 2*np.diag(grad_D)*x[J*C-int(C*(C-1)/2) : J*(C+1)-int(C*(C-1)/2)]

    for i in range(C):
        grad_x[int(i*J - i*(i-1)/2) : int((i+1)*J - i*(i+1)/2)] = grad_L[i:,i]
    return grad_x

def efa_init(J,C):
    x_init = np.zeros(J*(C+1)- int(C*(C-1)/2))
    x_init[:J*C- int(C*(C-1)/2)] = rgt.randn(J*C- int(C*(C-1)/2))
    x_init[J*C- int(C*(C-1)/2): J*(C+1)- int(C*(C-1)/2)] = 1 + rgt.rand(J)

    return x_init

def Q_mat(J,G):
    Q = np.zeros([J,(1+G)])
    Q[:,0] = np.ones(J)
    for i in range(int(J/G)):
        Q[i*G:(i+1)*G, 1: (G+1)] = np.identity(G)
    return Q

def stan_trans(x,G):
    if len(x) != int(G*(G-1)/2):
        raise NotImplementedError

    h = np.zeros([G,G])
    for i in range(1,G):
        h[0:i,i] = x[int((i-1)*i/2):int((i+1)*i/2)]
    Z = np.tanh(h)
    #Z = (np.exp(2*h)-1) / (np.exp(2*h) + 1)

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

def init_value(J,G):
    x = np.zeros(J*(1+G) + int(G*(G-1)/2) + J)
    x[0:int(G*(G-1)/2)] = rgt.randn(int(G*(G-1)/2))
    
    x[J*(1+G) + int(G*(G-1)/2) : J*(1+G) + int(G*(G-1)/2) + J] = 1 + rgt.rand(J)
    
    x[int(G*(G-1)/2): int(G*(G-1)/2) + J] = rgt.rand(J)
    
    x[int(G*(G-1)/2)+J : int(G*(G-1)/2) + J*(1+G)] = 0.1*rgt.randn(G*J)
    
    return x

def alm_solve(x_init,J,G,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter):
    x_old = x_init.copy()
    
    gamma = gamma_0.copy()
    rho = rho_0
       
    L_old,Psi_old,d_old,Cov_old = para_decompose(x_old,J,G)
    cons_val_old = cons_fun(x_old,J,G,Pair)
    
    dist_val = np.max(np.sort(np.abs(L_old[:,1:]),axis=1)[:,-2])
 
    
    iter_num = 0
    while dist_val > tol and iter_num < max_iter:
        result = minimize(objective_function,x_old,args=(J,G,S,n,gamma,rho,Pair),method = 'L-BFGS-B',jac = alm_gd)
        x_new = result.x
            
        cons_val_new =  cons_fun(x_new,J,G,Pair)
        gamma = gamma + rho *cons_val_new
        if np.linalg.norm(cons_val_old,ord = 'fro') > theta*np.linalg.norm(cons_val_new,ord = 'fro'):
            rho = rho*rho_sigma
            
        x_old = x_new.copy()
        L_old,Psi_old,d_old,Cov_old = para_decompose(x_old,J,G)
        dist_val = np.max(np.sort(np.abs(L_old[:,1:]),axis=1)[:,-2])

        iter_num = iter_num + 1
    return x_old,iter_num,dist_val

def nll(Psi,L,D,S,n):
    Cov = L @ (Psi @ L.T) + D
    R = np.linalg.solve(Cov,S)
    nll = n*np.log(np.linalg.det(2*np.pi*Cov))/2 + n*np.trace(R)/2
    return nll

def generator(J,G,Q,n):
    D_true = np.identity(J)
    
    X = 2*rgt.randn(J,G)
    L_true = np.zeros([J,(1+G)])
    L_true[:,0] = rgt.rand(J)
    L_true[:,1:(G+1)] = np.sign(X)*(0.1+np.abs(X))
    L_true = L_true*Q
    
    Phi_true = np.zeros([1+G,1+G])
    Phi_true[0,0] = 1
    tmp = 0.5*rgt.randn(int(G*(G-1)/2))
    
    Phi_true[1:(G+1),1:(G+1)] = stan_trans(tmp,G)
    Psi_true = Phi_true.T @ Phi_true
    Cov_true = L_true @ (Psi_true @ L_true.T) + D_true
    
    
    return D_true,L_true,Phi_true,Psi_true,Cov_true

def sampling_process(J,Cov,n):
    mean = np.zeros(J)
    samples = np.random.multivariate_normal(mean, Cov, size=n)
    S = samples.T @ samples /n
    return S

def Compare_process(J,G,S,n,rho_0,rho_sigma,theta,tol,max_iter,Pair
                    ,P_list,Q_true,L_true,Psi_true,Cov_true,D_true,Delta_list):
    print('Start')
    C=G+1
    gamma_0 = np.ones([int((G-1)*G/2),J])
    X_efa_list = []
    Nll_efa_list = np.zeros(Repeat)
    
    Nll_alm_list = []
    Unfinished_alm = []

    X_alm_list = []
    for i in range(Repeat):
        # efa estimation
        x_init_efa = efa_init(J, C)
        efa_result = minimize(efa_nll,x_init_efa,args = (J,C,S,n),method = 'L-BFGS-B',jac= efa_grad)
        X_efa_list.append(efa_result.x)
        Nll_efa_list[i] = efa_result.fun
        
        # alm estimation
        x_init_alm = init_value(J,G)
        x_alm,iter_num,dist_val =  alm_solve(x_init_alm,J,G,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
        
        if iter_num < max_iter:
            L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,G)
            Nll_alm_list.append(nll(Psi_alm,L_alm,np.diag(d_alm),S,n))
            X_alm_list.append(x_alm)
        else:
            Unfinished_alm.append(x_alm)
   
    restart_num = 0 
    while len(Nll_alm_list)< int(Repeat/2) and restart_num<5:
            #print('Encounter restart')
        restart_num = restart_num + 1
        Repeat_UF = []
        for i in range(len(Unfinished_alm)):
            x_alm,iter_num,dist_val = alm_solve(Unfinished_alm[i],J,G,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
            if iter_num < max_iter:   
                L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,G)
                nll_alm = nll(Psi_alm,L_alm,np.diag(d_alm),S,n)
                Nll_alm_list.append(nll_alm)
                X_alm_list.append(x_alm)
            else: 
                Repeat_UF.append(x_alm)
        Unfinished_alm = Repeat_UF
    # select alm estimator    
    if len(Nll_alm_list)>0:
        UF_id = 0
        Nll_alm_list = np.array(Nll_alm_list)
        best_alm_val = np.min(Nll_alm_list)
        best_alm_loc = np.where(Nll_alm_list == best_alm_val)[0][0]
        x_alm_est = X_alm_list[best_alm_loc]
        L_alm_est,Psi_alm_est,d_alm_est,Cov_alm_est = para_decompose(x_alm_est,J,G)
    else:
        UF_id = 1
        for i in range(len(Unfinished_alm)):
            x_alm =  Unfinished_alm[i]
            L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,G)
            nll_alm = nll(Psi_alm,L_alm,np.diag(d_alm),S,n)
            Nll_alm_list.append(nll_alm)
        Nll_alm_list = np.array(Nll_alm_list)
        best_alm_val = np.min(Nll_alm_list)
        best_alm_loc = np.where(Nll_alm_list == best_alm_val)[0][0]
        x_alm_est = Unfinished_alm[best_alm_loc]
        L_alm_est,Psi_alm_est,d_alm_est,Cov_alm_est = para_decompose(x_alm_est,J,G)
        
    Lr_true = Cov_true - D_true
        
    # estimation error of alm
    Lr_alm_est = L_alm_est @ (Psi_alm_est @ L_alm_est.T)
        
    d_alm_err = np.linalg.norm(d_alm_est - np.diag(D_true))**2/J
    Lr_alm_err = np.linalg.norm(Lr_alm_est-Lr_true,ord = 'fro')**2/J**2 
        

    L_alm_err_list = np.zeros(len(P_list))
    Psi_alm_err_list = np.zeros(len(P_list))
    for i in range(len(P_list)):
        Per_m = np.zeros([(1+G),(1+G)])
        Per_m[0,0] = 1
        Per_m[1:(1+G),1:(1+G)] = P_list[i] 
        L_rot = L_true @ Per_m
        S_rot = np.diag(np.diag(np.sign(L_alm_est.T @ L_rot)))
        L_alm_err_list[i] = np.linalg.norm(L_alm_est - L_rot @ S_rot,ord='fro')**2/J
            
        Psi_true_rot  = S_rot @ (Per_m.T @ (Psi_true @ (Per_m @ S_rot)))
        Psi_alm_err_list[i] = np.linalg.norm(Psi_alm_est -Psi_true_rot ,ord='fro')/(1+G)**2
    L_alm_err = np.min(L_alm_err_list) 
    loc_alm_err = np.where(L_alm_err_list == L_alm_err)[0][0]
    Psi_alm_err = Psi_alm_err_list[loc_alm_err]
        
    # recovery of bi-factor of alm
    Q_alm_raw = np.where(np.abs(L_alm_est)>tol,1,0)
    Q_alm_raw[:,0] = np.ones(J)
        
    Exact_cover_alm = Exact_Cr(Q_true,Q_alm_raw,P_list)
    Average_cover_alm = Average_Cr(Q_true,Q_alm_raw,P_list)
            
    # select efa estimator
    efa_best_nll = np.min(Nll_efa_list)
    efa_best_loc = np.where(Nll_efa_list == efa_best_nll)[0][0]
    x_efa_est = X_efa_list[efa_best_loc]
    L_efa,D_efa = efa_decom(x_efa_est,J,C)
        
    Lr_efa = L_efa @ L_efa.T
        
    d_efa_err = np.linalg.norm(np.diag(D_efa)-np.diag(D_true))**2/J
    Lr_efa_err = np.linalg.norm(Lr_efa-Lr_true,ord='fro')**2/J**2
        
    ncol_A = L_efa.shape[1]
    efa_matrix = r.matrix(FloatVector((L_efa.T).flatten()), nrow=L_efa.shape[0], ncol=L_efa.shape[1])   
    L_bf_list = []
    Psi_bf_list = []
    for i in range(Repeat_rf):
        try: 
            if i==0:
                Tmat = r.matrix(FloatVector(np.eye(ncol_A).flatten()), nrow=ncol_A, ncol=ncol_A)
            else:
                T_init =  othog_gen(G)
                Tmat = r.matrix(FloatVector(T_init.T.flatten()), nrow=ncol_A, ncol=ncol_A)
            
            result = GPArotation.bifactorQ(efa_matrix,Tmat=Tmat,normalize=normalize,eps=eps,maxit=maxit,randomStarts=randomStarts)
            L_bf = np.array(result[0])
            Psi_bf = np.array(result[1])
            
            L_bf_list.append(L_bf)
            Psi_bf_list.append(Psi_bf)
            
        except Exception as e:
            print(f"An error occurred during iteration {i + 1}: {e}")
            continue  
        
    if len(L_bf_list)>0:
        L_bf_err_list = []
        Psi_bf_err_list = []
        Exact_cover_efa_list = np.zeros([len(L_bf_list),len(Delta_list)])
        Average_cover_efa_list = np.zeros([len(L_bf_list),len(Delta_list)])
        
        # recovery of bi factor structure
        for i in range(len(L_bf_list)):
            L_bf_temp = L_bf_list[i]
            for j in range(len(Delta_list)):
                delta = Delta_list[j]
                Q_efa_raw = np.where(np.abs(L_bf_temp)>=delta,1,0)
                Q_efa_raw[:,0] = np.ones(J)
                Exact_cover_efa_list[i,j] = Exact_Cr(Q_true,Q_efa_raw,P_list)
                Average_cover_efa_list[i,j] = Average_Cr(Q_true,Q_efa_raw,P_list)
        Exact_cover_efa = np.max(Exact_cover_efa_list,axis=0) 
        Average_cover_efa = np.max(Average_cover_efa_list,axis=0) 
        
        # estimation error of efa + bi rot
        for i in range(len(L_bf_list)):
            L_bf_temp = L_bf_list[i]
            Psi_bf_temp = Psi_bf_list[i]
            for j in range(len(P_list)):
                Per_m = np.zeros([(1+G),(1+G)])
                Per_m[0,0] = 1
                Per_m[1:(1+G),1:(1+G)] = P_list[j] 
                L_rot = L_true @ Per_m
                S_rot = np.diag(np.diag(np.sign(L_bf_temp.T @ L_rot)))
                L_bf_err_list.append(np.linalg.norm(L_alm_est - L_rot @ S_rot,ord='fro')**2/J)
                
                Psi_true_rot  = S_rot @ (Per_m.T @ (Psi_true @ (Per_m @ S_rot)))
                Psi_bf_err_list.append(np.linalg.norm(Psi_bf_temp -Psi_true_rot ,ord='fro')/(1+G)**2)
        L_bf_err_list = np.array(L_bf_err_list)
        Psi_bf_err_list = np.array(Psi_bf_err_list)
        
        L_efa_err = np.min(L_bf_err_list)
        L_efa_in_loc = np.where(L_bf_err_list == L_efa_err)[0][0]
        Psi_efa_err = Psi_bf_err_list[L_efa_in_loc]
        
    else:
        L_efa_err = -1
        Psi_efa_err = -1

        Exact_cover_efa = -np.ones(len(Delta_list))
        Average_cover_efa = -np.ones(len(Delta_list))
    print('Finished')
    return L_alm_err,Psi_alm_err,Lr_alm_err,d_alm_err,Exact_cover_alm,Average_cover_alm,L_efa_err,Psi_efa_err,Lr_efa_err,d_efa_err,Exact_cover_efa,Average_cover_efa,UF_id
        
        
        
        
    
if __name__ == '__main__':
    rgt.seed(2024)
    robjects.r['set.seed'](2024)
    GPArotation = importr('GPArotation')
    normalize = False
    eps = 1e-5
    maxit = 1000
    randomStarts = 0
    
    Delta_list = [0.2,0.4,0.6,0.8,1]
    
    J = 30
    G = 5
    C = G+1
    n=2000
    tol = 1e-2
    max_iter = 1000 
    rho_0 = 100
    rho_sigma = 10
    theta = 0.25
    gamma_0 = np.ones([int((G-1)*G/2),J])
    
    Epoch = 100
    Repeat = 50
    Repeat_rf = 50
    
    Q = Q_mat(J,G)
    
    Permutation_list = BF_permutation(G)
    
    Pair = []
    for i in range(1,(G+1)):
        for j in range(i+1,(G+1)):
            Pair.append(np.array([i,j]))
    S_list = [] 
    D_true,L_true,Phi_true,Psi_true,Cov_true = generator(J,G,Q,n)    
    print(Psi_true)
    for epoch in tqdm(range(Epoch)):
        S = sampling_process(J,Cov_true,n)
        S_list.append(S)
        
    params_list = [(J,G,S_list[i],n,rho_0,rho_sigma,theta,tol,max_iter,Pair
                    ,Permutation_list,Q,L_true,Psi_true,Cov_true,D_true,Delta_list) for i in range(Epoch)]
    
    with Pool(processes= Epoch) as p:
        results = p.starmap(Compare_process, params_list)
        
    L_alm_err_list = []
    Psi_alm_err_list = []
    Lr_alm_err_list = []
    d_alm_err_list = []
    Exact_cover_alm_list = []
    Average_cover_alm_list = []
    L_efa_err_list = []
    Psi_efa_err_list = []
    Lr_efa_err_list = []
    d_efa_err_list = []
    Exact_cover_efa_list = []
    Average_cover_efa_list = []
    UF_id_list = []
    
    for L_alm_err,Psi_alm_err,Lr_alm_err,d_alm_err,Exact_cover_alm,Average_cover_alm,L_efa_err,Psi_efa_err,Lr_efa_err,d_efa_err,Exact_cover_efa,Average_cover_efa,UF_id in results:
        L_alm_err_list.append(L_alm_err)
        Psi_alm_err_list.append(Psi_alm_err)
        Lr_alm_err_list.append(Lr_alm_err)
        d_alm_err_list.append(d_alm_err)
        Exact_cover_alm_list.append(Exact_cover_alm)
        Average_cover_alm_list.append(Average_cover_alm)
        
        L_efa_err_list.append(L_efa_err)
        Psi_efa_err_list.append(Psi_efa_err)
        Lr_efa_err_list.append(Lr_efa_err)
        d_efa_err_list.append(d_efa_err)
        Exact_cover_efa_list.append(Exact_cover_efa)
        Average_cover_efa_list.append(Average_cover_efa)
        UF_id_list.append(UF_id)
        
    
    L_alm_err_list = np.array(L_alm_err_list)
    Psi_alm_err_list = np.array(Psi_alm_err_list)
    Lr_alm_err_list = np.array(Lr_alm_err_list)
    d_alm_err_list = np.array(d_alm_err_list)
    Exact_cover_alm_list = np.array(Exact_cover_alm_list)
    Average_cover_alm_list = np.array(Average_cover_alm_list)
    L_efa_err_list = np.array(L_efa_err_list)
    Psi_efa_err_list = np.array(Psi_efa_err_list)
    Lr_efa_err_list = np.array(Lr_efa_err_list)
    d_efa_err_list = np.array(d_efa_err_list)
    Exact_cover_efa_list = np.vstack(Exact_cover_efa_list)
    Average_cover_efa_list = np.vstack(Average_cover_efa_list)
    UF_id_list = np.array(UF_id_list)
    
    print(L_alm_err_list)
    print(Psi_alm_err_list)
    print(Lr_alm_err_list)
    print(d_alm_err_list)
    print(Exact_cover_alm_list)
    print(Average_cover_alm_list)
    print(L_efa_err_list)
    print(Psi_efa_err_list)
    print(Lr_efa_err_list)
    print(d_efa_err_list)
    #print(Exact_cover_efa_list)
    #print(Average_cover_efa_list)
    print(UF_id_list)
    
    print(np.mean(L_alm_err_list))
    print(np.mean(Psi_alm_err_list))
    print(np.mean(Lr_alm_err_list))
    print(np.mean(d_alm_err_list))
    print(np.mean(Exact_cover_alm_list))
    print(np.mean(Average_cover_alm_list))
    print(np.mean(L_efa_err_list))
    print(np.mean(Psi_efa_err_list))
    print(np.mean(Lr_efa_err_list))
    print(np.mean(d_efa_err_list))
    print(np.mean(Exact_cover_efa_list,axis=0))
    print(np.mean(Average_cover_efa_list,axis=0))
    print(np.mean(UF_id_list))
    
    L_alm_err_list_name = 'ALMBFrot/' + 'L_alm_err_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Psi_alm_err_list_name = 'ALMBFrot/' + 'Psi_alm_err_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Lr_alm_err_list_name = 'ALMBFrot/' + 'Lr_alm_err_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    d_alm_err_list_name = 'ALMBFrot/' + 'd_alm_err_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Exact_cover_alm_list_name = 'ALMBFrot/' + 'Exact_cover_alm_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Average_cover_alm_list_name = 'ALMBFrot/' + 'Average_cover_alm_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    L_efa_err_list_name = 'ALMBFrot/' + 'L_efa_err_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Psi_efa_err_list_name = 'ALMBFrot/' + 'Psi_efa_err_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Lr_efa_err_list_name = 'ALMBFrot/' + 'Lr_efa_err_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    d_efa_err_list_name = 'ALMBFrot/' + 'd_efa_err_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Exact_cover_efa_list_name = 'ALMBFrot/' + 'Exact_cover_efa_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Average_cover_efa_list_name = 'ALMBFrot/' + 'Average_cover_efa_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    UF_id_list_name = 'ALMBFrot/' + 'UF_id_list_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    
    np.save(L_alm_err_list_name,L_alm_err_list)
    np.save(Psi_alm_err_list_name,Psi_alm_err_list)
    np.save(Lr_alm_err_list_name,Lr_alm_err_list)
    np.save(d_alm_err_list_name,d_alm_err_list)
    np.save(Exact_cover_alm_list_name,Exact_cover_alm_list)
    np.save(Average_cover_alm_list_name,Average_cover_alm_list)
    np.save(L_efa_err_list_name,L_efa_err_list)
    np.save(Psi_efa_err_list_name,Psi_efa_err_list)
    np.save(Lr_efa_err_list_name,Lr_efa_err_list)
    np.save(d_efa_err_list_name,d_efa_err_list)
    np.save(Exact_cover_efa_list_name,Exact_cover_efa_list)
    np.save(Average_cover_efa_list_name,Average_cover_efa_list)
    np.save(UF_id_list_name,UF_id_list)
    
    