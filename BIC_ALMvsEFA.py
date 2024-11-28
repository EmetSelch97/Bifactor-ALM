# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:01:37 2024

@author: ZZ
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as rgt
from scipy.optimize import minimize
from multiprocessing import Pool
#from tqdm import tqdm

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

def init_value(J,G):
    x = np.zeros(J*(1+G) + int(G*(G-1)/2) + J)
    x[0:int(G*(G-1)/2)] = rgt.randn(int(G*(G-1)/2))
    
    x[J*(1+G) + int(G*(G-1)/2) : J*(1+G) + int(G*(G-1)/2) + J] = 1 + rgt.rand(J)
    
    x[int(G*(G-1)/2): int(G*(G-1)/2) + J] = rgt.rand(J)
    
    x[int(G*(G-1)/2)+J : int(G*(G-1)/2) + J*(1+G)] = 0.1*rgt.randn(G*J)
    
    return x

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
    while max(dist_para,dist_val) > tol and iter_num < max_iter:
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
    #tmp = rgt.randn(int(G*(G-1)/2))
    tmp = 0.5*rgt.randn(int(G*(G-1)/2))
    
    Phi_true[1:(G+1),1:(G+1)] = stan_trans(tmp,G)
    Psi_true = Phi_true.T @ Phi_true
    Cov_true = L_true @ (Psi_true @ L_true.T) + D_true
    
    #mean = np.zeros(J)
    #samples = np.random.multivariate_normal(mean, Cov_true, size=n)
    #S = samples.T @ samples /n
    
    return D_true,L_true,Phi_true,Psi_true,Cov_true

def sampling_process(J,Cov,n):
    mean = np.zeros(J)
    samples = np.random.multivariate_normal(mean, Cov, size=n)
    S = samples.T @ samples /n
    return S

def BIC_compare(J,up_r,low_r,Repeat,S,n,rho_0,rho_sigma,theta,tol,max_iter):
    print('Start')
    BIC_alm = np.zeros(int(up_r+1-low_r))
    Loc_alm = np.zeros(int(up_r+1-low_r))
    Unstop_alm = np.zeros(int(up_r+1-low_r))
    NLL_alm = np.zeros(int(up_r+1-low_r))
    
    BIC_efa = np.zeros(int(up_r+1-low_r))
    Loc_efa = np.zeros(int(up_r+1-low_r))
    NLL_efa = np.zeros(int(up_r+1-low_r))
    for g in range(low_r,up_r+1):
        print(g)
        C = g+1
        Pair = []
        for i in range(1,(g+1)):
            for j in range(i+1,(g+1)):
                Pair.append(np.array([i,j]))
        
        gamma_0 = np.ones([int((g-1)*g/2),J])
        
        NLL_alm = []
        #Niter_alm = np.zeros(Repeat)
        Unfinshed_alm = []
        
        
        NLL_efa = np.zeros(Repeat)
        for i in range(Repeat):
            x_alm_init = init_value(J,g)
            x_alm,iter_num,dist_val = alm_solve(x_alm_init,J,g,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
            if iter_num < max_iter:
                L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,g)
                nll_alm = nll(Psi_alm,L_alm,np.diag(d_alm),S,n)
                NLL_alm.append(nll_alm)
            else: 
                Unfinshed_alm.append(x_alm)
            
            x_init_efa = efa_init(J, C)
            efa_result = minimize(efa_nll,x_init_efa,args = (J,C,S,n),method = 'L-BFGS-B',jac= efa_grad)
            NLL_efa[i] = efa_result.fun
        
        best_nll_efa = np.min(NLL_efa)
        Loc_efa[g-low_r] = g
        BIC_efa[g-low_r] = 2*best_nll_efa + (J*(C+1) - g*(g+1)/2)*np.log(n)
        NLL_efa[g-low_r] = best_nll_efa
        
        restart_num = 0 
        while len(NLL_alm)< int(Repeat/2) and restart_num<5:
            restart_num = restart_num + 1
            #Unfinshed_alm = []
            Repeat_UF = []
            for i in range(len(Unfinshed_alm)):
                x_alm,iter_num,dist_val = alm_solve(Unfinshed_alm[i],J,g,S,n,Pair,gamma_0,rho_0,rho_sigma,theta,tol,max_iter)
                if iter_num < max_iter:
                    L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,g)
                    nll_alm = nll(Psi_alm,L_alm,np.diag(d_alm),S,n)
                    NLL_alm.append(nll_alm)
                else: 
                    Repeat_UF.append(x_alm)
            Unfinshed_alm = Repeat_UF
        
        if len(NLL_alm)>0:
            NLL_alm = np.array(NLL_alm)
            best_nll_alm = np.min(NLL_alm)
            bic_alm = 2*best_nll_alm + (3*J + g*(g-1)/2)*np.log(n)
            BIC_alm[g-low_r] = bic_alm
            Loc_alm[g-low_r] = g
            NLL_alm[g-low_r] = best_nll_alm
        else:
            for i in range(len(Unfinshed_alm)):
                x_alm = Unfinshed_alm[i]
                L_alm,Psi_alm,d_alm,Cov_alm=para_decompose(x_alm,J,g)
                nll_alm = nll(Psi_alm,L_alm,np.diag(d_alm),S,n)
                NLL_alm.append(nll_alm)
            NLL_alm = np.array(NLL_alm)
            best_nll_alm = np.min(NLL_alm)
            bic_alm = 2*best_nll_alm + (3*J + g*(g-1)/2)*np.log(n)
            BIC_alm[g-low_r] = bic_alm
            Loc_alm[g-low_r] = g
            Unstop_alm[g-low_r] = 1
            NLL_alm[g-low_r] = best_nll_alm
    print('Finished')
    best_bic_alm = np.min(BIC_alm)
    best_loc_alm = Loc_alm[np.where(BIC_alm ==best_bic_alm)[0][0]]
    check_stop_alm = Unstop_alm[np.where(BIC_alm ==best_bic_alm)[0][0]]
    
    best_bic_efa = np.min(BIC_efa)
    best_loc_efa = Loc_alm[np.where(BIC_efa ==best_bic_efa)[0][0]]
    return best_loc_alm,check_stop_alm,best_loc_efa,BIC_alm,BIC_efa,NLL_alm,NLL_efa

if __name__ == '__main__':
    rgt.seed(2024)
    J = 30
    G = 5
    up_r = G+1
    low_r = G-1
    #Repeat = 20
    Repeat = 50
    tol =1e-2
    max_iter = 1000
    n= 2000
    rho_0 = 100
    rho_sigma = 10
    theta = 0.25
    Epoch = 100
    Q = Q_mat(J,G)
    
    S_list = []
    D_true,L_true,Phi_true,Psi_true,Cov_true = generator(J,G,Q,n)
    print(Psi_true)
    for epoch in range(Epoch):
        S = sampling_process(J,Cov_true,n)
        S_list.append(S)
        
    params_list = [(J,up_r,low_r,Repeat,S_list[i],n,rho_0,rho_sigma,theta,tol,max_iter) for i in range(Epoch)]
    
    with Pool(processes= Epoch) as p:
        results = p.starmap(BIC_compare, params_list)
        
    ALM_loc_list = []
    ALM_stop_list = []
    EFA_loc_list = []
    
    ALM_bic = []
    EFA_bic = []
    
    ALM_NLL = []
    EFA_NLL = []
    for alm_loc,alm_stop,efa_loc,BIC_alm,BIC_efa,NLL_alm,NLL_efa in results:
        ALM_loc_list.append(alm_loc)
        ALM_stop_list.append(alm_stop)
        EFA_loc_list.append(efa_loc)
        ALM_NLL.append(ALM_NLL)
        EFA_NLL.append(EFA_NLL)
    print(ALM_loc_list)   
    print(ALM_stop_list)
    print(EFA_loc_list)
    
    ALM_loc_list = np.array(ALM_loc_list)
    ALM_stop_list = np.array(ALM_stop_list)
    EFA_loc_list= np.array(EFA_loc_list)
    
    print(np.mean(ALM_loc_list==G))
    print(np.mean(EFA_loc_list==G))
    
    print(np.mean(ALM_loc_list))
    print(np.mean(ALM_stop_list))
    print(np.mean(EFA_loc_list))
    ALM_loc_list_name = 'ALMBF_revised/BIC/BIC_ALM_loc_' + str(int(J))+ '_' +  str(int(G))+ '_' +  str(int(n))
    #ALM_loc_list_name = 'ALMsimulation/BIC_ALM_loc_' + str(int(J))+ '_' +  str(int(G))
    np.save(ALM_loc_list_name,ALM_loc_list)
    ALM_stop_list_name = 'ALMBF_revised/BIC/BIC_ALM_stop_' + str(int(J))+ '_' +  str(int(G))+ '_' +  str(int(n))
    #ALM_stop_list_name = 'ALMsimulation/BIC_ALM_stop_' + str(int(J))+ '_' +  str(int(G))
    np.save(ALM_stop_list_name,ALM_stop_list)
    EFA_loc_list_name = 'ALMBF_revised/BIC/BIC_EFA_loc_' + str(int(J))+ '_' +  str(int(G))+ '_' +  str(int(n))
    #EFA_loc_list_name = 'ALMsimulation/BIC_EFA_loc_' + str(int(J))+ '_' +  str(int(G))
    np.save(EFA_loc_list_name,EFA_loc_list)