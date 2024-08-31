# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:28:24 2024

@author: 888
"""

import numpy as np
from itertools import permutations

def Compare_Q(Q,Q_est):
    r = Q.shape[1]
    J = Q.shape[0]
    average_cr = 0
    for i in range(1,r):
        zero = np.where(Q[:,i] == 0)[0]
        zero_est = np.where(Q_est[:,i] == 0)[0]
        zero_common = np.intersect1d(zero, zero_est)

        one = np.where(Q[:,i] == 1)[0]
        one_est = np.where(Q_est[:,i] == 1)[0]
        one_common = np.intersect1d(one, one_est)

        average_cr = average_cr + len(zero_common) + len(one_common)
    return average_cr/((r-1)*J)

def Average_Cr(Q,Q_est,R_list):
    Cr_list = np.zeros(len(R_list))
    for i in range(len(R_list)):
        Q_r = Q_est.copy()
        Q_r[:,1:] = Q_est[:,1:] @ R_list[i]
        Cr_list[i] = Compare_Q(Q,Q_r)
    return np.max(Cr_list)


def Exact_Cr(Q,Q_est,R_list):
    Cr_list = np.zeros(len(R_list))
    for i in range(len(R_list)):
        Q_r = Q_est.copy()
        Q_r[:,1:] = Q_est[:,1:] @ R_list[i]
        Cr_list[i] = np.array_equal(Q, Q_r)
    return np.sum(Cr_list)

def BF_permutation(n):
    elements = list(range(1, n+1))
    all_permutations = permutations(elements)
    
    permutation_matrices = []
    for perm in all_permutations:
        matrix = np.zeros((n, n))
        for i, p in enumerate(perm):
            matrix[i, p-1] = 1
        permutation_matrices.append(matrix)
    
    return permutation_matrices

def Hierarchy_permutation():
    Rotation_list = []
    I1 = np.identity(6)
    Rotation_list.append(I1)
    I2 = np.identity(6)
    I2[2,2] = 0
    I2[2,3] = 1
    I2[3,2] = 1
    I2[3,3] = 0
    Rotation_list.append(I2)
    I3 = np.zeros([6,6])
    I3[0,0] = 1
    I3[1,1] = 1
    I3[2,2] = 1
    I3[3,3] = 1
    I3[4,5] = 1
    I3[5,4] = 1
    Rotation_list.append(I3)
    I4 = np.zeros([6,6])
    I4[0,0] = 1
    I4[1,1] = 1
    I4[2,3] = 1
    I4[3,2] = 1
    I4[4,5] = 1
    I4[5,4] = 1
    Rotation_list.append(I4)
    I5 = np.zeros([6,6])
    I5[0,1] = 1
    I5[1,0] = 1
    I5[2,4] = 1
    I5[3,5] = 1
    I5[4,2] = 1
    I5[5,3] = 1
    Rotation_list.append(I5)
    I6 = np.zeros([6,6])
    I6[0,1] = 1
    I6[1,0] = 1
    I6[2,5] = 1
    I6[3,4] = 1
    I6[4,3] = 1
    I6[5,2] = 1
    Rotation_list.append(I6)
    I7 = np.zeros([6,6])
    I7[0,1] = 1
    I7[1,0] = 1
    I7[2,4] = 1
    I7[3,5] = 1
    I7[4,3] = 1
    I7[5,2] = 1
    Rotation_list.append(I7)
    I8 = np.zeros([6,6])
    I8[0,1] = 1
    I8[1,0] = 1
    I8[2,5] = 1
    I8[3,4] = 1
    I8[4,2] = 1
    I8[5,3] = 1
    Rotation_list.append(I8)
    return Rotation_list