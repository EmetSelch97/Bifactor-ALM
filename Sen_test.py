# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:52:13 2024

@author: ZZ
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as rgt
from scipy.optimize import minimize
from tqdm import tqdm
import time

from Permutation import BF_permutation,Compare_Q,Average_Cr,Exact_Cr
from multiprocessing import Pool

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
    time_start = time.time()
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
    time_end = time.time()
    
    time_cost = time_end - time_start
    return x_old,iter_num,dist_val,time_cost

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

def sensitivity_process(J,G,S,n,gamma_0,rho_0,Sigma_set,Theta_set,tol,max_iter,Pair
                        ,P_list,Q_true,L_true,Psi_true,Cov_true,D_true,Repeat):
    print('start')
    
    NS = len(Sigma_set)
    NT = len(Theta_set)
    
    Lr_err_arr = np.zeros([NS,NT])
    L_err_arr = np.zeros([NS,NT])
    Psi_err_arr = np.zeros([NS,NT])
    D_err_arr = np.zeros([NS,NT])
    EMC_arr = np.zeros([NS,NT])
    ACC_arr = np.zeros([NS,NT])
    Time_cost_arr = np.zeros([NS,NT])
    Niter_arr = np.zeros([NS,NT])
    Rep_num_arr = np.zeros([NS,NT])
    Finished_arr = np.zeros([NS,NT])
    for i in range(NS):
        for j in range(NT):
            rho_sigma = Sigma_set[i]
            theta = Theta_set[j]
            
            Nll_alm_list = []
            Unfinished_alm = []

            X_alm_list = []
            
            Niter_list = []
            Time_list = []
            
            for rep in range(Repeat):
                x_init_alm = init_value(J,G)
                x_alm,iter_num,dist_val,time_val =  alm_solve(x_init_alm,J,G,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
                
                Time_list.append(time_val)
                Niter_list.append(iter_num)
                if iter_num < max_iter:
                    L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,G)
                    Nll_alm_list.append(nll(Psi_alm,L_alm,np.diag(d_alm),S,n))
                    X_alm_list.append(x_alm)
                else:
                    Unfinished_alm.append(x_alm)
            
            restart_num = 0 
            while len(Nll_alm_list)< int(Repeat/2) and restart_num<5:
                restart_num = restart_num + 1
                Repeat_UF = []
                for k in range(len(Unfinished_alm)):
                    x_alm,iter_num,dist_val,time_val = alm_solve(Unfinished_alm[k],J,G,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
                    Time_list.append(time_val)
                    Niter_list.append(iter_num)
                    if iter_num < max_iter:   
                        L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,G)
                        nll_alm = nll(Psi_alm,L_alm,np.diag(d_alm),S,n)
                        Nll_alm_list.append(nll_alm)
                        X_alm_list.append(x_alm)
                    else: 
                        Repeat_UF.append(x_alm)
                Unfinished_alm = Repeat_UF
            
            
            
            if len(Nll_alm_list)>0:
                UF_id = 0
                Nll_alm_list = np.array(Nll_alm_list)
                best_alm_val = np.min(Nll_alm_list)
                best_alm_loc = np.where(Nll_alm_list == best_alm_val)[0][0]
                x_alm_est = X_alm_list[best_alm_loc]
                L_alm_est,Psi_alm_est,d_alm_est,Cov_alm_est = para_decompose(x_alm_est,J,G)
            else:
                UF_id = 1
                for k in range(len(Unfinished_alm)):
                    x_alm =  Unfinished_alm[k]
                    L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,G)
                    nll_alm = nll(Psi_alm,L_alm,np.diag(d_alm),S,n)
                    Nll_alm_list.append(nll_alm)
                Nll_alm_list = np.array(Nll_alm_list)
                best_alm_val = np.min(Nll_alm_list)
                best_alm_loc = np.where(Nll_alm_list == best_alm_val)[0][0]
                x_alm_est = Unfinished_alm[best_alm_loc]
                L_alm_est,Psi_alm_est,d_alm_est,Cov_alm_est = para_decompose(x_alm_est,J,G)
            
            # summary of results
            
            Finished_arr[i,j] = UF_id
            Rep_num_arr[i,j] = len(Time_list)
            Niter_arr[i,j] = np.sum(np.array(Niter_list))
            Time_cost_arr[i,j] = np.sum(np.array(Time_list))
            
            Lr_true = Cov_true - D_true
                
            # estimation error of alm
            Lr_alm_est = L_alm_est @ (Psi_alm_est @ L_alm_est.T)
                
            d_alm_err = np.linalg.norm(d_alm_est - np.diag(D_true))**2/J
            Lr_alm_err = np.linalg.norm(Lr_alm_est-Lr_true,ord = 'fro')**2/J**2 
                

            L_alm_err_list = np.zeros(len(P_list))
            Psi_alm_err_list = np.zeros(len(P_list))
            for s in range(len(P_list)):
                Per_m = np.zeros([(1+G),(1+G)])
                Per_m[0,0] = 1
                Per_m[1:(1+G),1:(1+G)] = P_list[s] 
                L_rot = L_true @ Per_m
                S_rot = np.diag(np.diag(np.sign(L_alm_est.T @ L_rot)))
                L_alm_err_list[s] = np.linalg.norm(L_alm_est - L_rot @ S_rot,ord='fro')**2/J
                    
                Psi_true_rot  = S_rot @ (Per_m.T @ (Psi_true @ (Per_m @ S_rot)))
                Psi_alm_err_list[s] = np.linalg.norm(Psi_alm_est -Psi_true_rot ,ord='fro')/(1+G)**2
            L_alm_err = np.min(L_alm_err_list) 
            loc_alm_err = np.where(L_alm_err_list == L_alm_err)[0][0]
            Psi_alm_err = Psi_alm_err_list[loc_alm_err]
                
            # recovery of bi-factor of alm
            Q_alm_raw = np.where(np.abs(L_alm_est)>tol,1,0)
            Q_alm_raw[:,0] = np.ones(J)
                
            Exact_cover_alm = Exact_Cr(Q_true,Q_alm_raw,P_list)
            Average_cover_alm = Average_Cr(Q_true,Q_alm_raw,P_list)
            
            Lr_err_arr[i,j] = Lr_alm_err
            L_err_arr[i,j] = L_alm_err
            Psi_err_arr[i,j] = Psi_alm_err
            D_err_arr[i,j] = d_alm_err
            EMC_arr[i,j] = Exact_cover_alm
            ACC_arr[i,j] = Average_cover_alm
    print('finished')
    return Lr_err_arr,L_err_arr,Psi_err_arr,D_err_arr,EMC_arr,ACC_arr,Time_cost_arr,Niter_arr,Rep_num_arr,Finished_arr
    
   

if __name__ == '__main__':
    rgt.seed(2024)
    
    J = 15 # 30
    G = 3 # 5
    
    n = 500 # n=2000
    
    tol = 1e-2
    max_iter = 100
    
    rho_0 = 100
    
    Rs = np.array([5,10,15])
    The = np.array([0.25,0.5,0.75])
    #rho_sigma = 10
    #theta = 0.25
    gamma_0 = np.ones([int((G-1)*G/2),J])
    
    Epoch = 100
    Repeat = 50
    
    Q = Q_mat(J,G)
    Permutation_list = BF_permutation(G)
    #gamma_0 = np.ones([int((G-1)*G/2),J])
    
    Pair = []
    for i in range(1,(G+1)):
        for j in range(i+1,(G+1)):
            Pair.append(np.array([i,j]))
            
    S_list = [] 
    D_true,L_true,Phi_true,Psi_true,Cov_true = generator(J,G,Q,n)    
    print(Psi_true)
    Psi_true_name = 'Sentest/' + 'Psi_true_' + str(int(J)) + '_' + str(int(G))
    L_true_name = 'Sentest/' + 'L_true_' + str(int(J)) + '_' + str(int(G))
    
    np.save(Psi_true_name,Psi_true)
    np.save(L_true_name,L_true)
    
    for i in range(Epoch):
        S_list.append(sampling_process(J,Cov_true,n))
    
    params_list = [(J,G,S_list[i],n,gamma_0,rho_0,Rs,The,tol,max_iter,Pair
                            ,Permutation_list,Q,L_true,Psi_true,Cov_true,D_true,Repeat) for i in range(Epoch)]
    
    with Pool(processes= Epoch) as p:
        results = p.starmap(sensitivity_process, params_list)
        
    Lr_err_list = []
    L_err_list = []
    Psi_err_list = []
    D_err_list = []
    
    EMC_list = []
    ACC_list = []
    Time_cost_list = []
    Niter_list = []
    Rep_num_list = []
    Finished_list = []
    
    for Lr_err_arr,L_err_arr,Psi_err_arr,D_err_arr,EMC_arr,ACC_arr,Time_cost_arr,Niter_arr,Rep_num_arr,Finished_arr in results:
        Lr_err_list.append(Lr_err_arr)
        L_err_list.append(L_err_arr)
        Psi_err_list.append(Psi_err_arr)
        D_err_list.append(D_err_arr)
        
        EMC_list.append(EMC_arr)
        ACC_list.append(ACC_arr)
        Time_cost_list.append(Time_cost_arr)
        Niter_list.append(Niter_arr)
        Rep_num_list.append(Rep_num_arr)
        Finished_list.append(Finished_arr)
        
    Lr_err_list = np.stack(Lr_err_list, axis=0)
    L_err_list = np.stack(L_err_list, axis=0)
    Psi_err_list = np.stack(Psi_err_list, axis=0)
    D_err_list = np.stack(D_err_list, axis=0)
    
    EMC_list = np.stack(EMC_list, axis=0)
    ACC_list = np.stack(ACC_list, axis=0)
    Time_cost_list = np.stack(Time_cost_list, axis=0)
    Niter_list = np.stack(Niter_list, axis=0)
    Rep_num_list = np.stack(Rep_num_list, axis=0)
    Finished_list = np.stack(Finished_list, axis=0)
    
    Lr_err_name = 'Sentest/' + 'Lr_err_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    L_err_name = 'Sentest/' + 'L_err_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Psi_err_name = 'Sentest/' + 'Psi_err_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    D_err_name = 'Sentest/' + 'D_err_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    
    EMC_name = 'Sentest/' + 'EMC_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    ACC_name = 'Sentest/' + 'ACC_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Time_cost_name = 'Sentest/' + 'Time_cost_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Niter_name = 'Sentest/' + 'Niter_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Rep_num_name = 'Sentest/' + 'Rep_num_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    Finished_name = 'Sentest/' + 'Finished_' + str(int(J)) + '_' + str(int(G)) + '_' + str(int(n))
    
    np.save(Lr_err_name,Lr_err_list)
    np.save(L_err_name,L_err_list)
    np.save(Psi_err_name,Psi_err_list)
    np.save(D_err_name,D_err_list)
    np.save(EMC_name,EMC_list)
    np.save(ACC_name,ACC_list)
    np.save(Time_cost_name,Time_cost_list)
    np.save(Niter_name,Niter_list)
    np.save(Rep_num_name,Rep_num_list)
    np.save(Finished_name,Finished_list)
    
    
    print(np.mean(L_err_list,axis=0))
    print(np.mean(EMC_list,axis=0))
    
    
    
    
    