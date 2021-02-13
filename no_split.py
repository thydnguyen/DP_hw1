# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:58:22 2021

@author: Thy Nguyen
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:01:35 2021

@author: Thy Nguyen
"""
import numpy as np
from cvxopt import matrix, solvers
from numpy import linalg as la
import argparse


solvers.options['show_progress'] = False
threshold = 0.5
thresholdHint = 0.5
n  = 20
d = 100
alpha = 1
output = []

for j in range(n):
    s = np.random.rand(d)
    
    z = np.random.rand(d)
    s = [1 if ss > threshold else 0  for ss in s  ]
    s_hint = np.array([ss if np.random.rand() > thresholdHint else 1-ss for ss in s])
    z =  [1 if zz > 0.5 else 0  for zz in z  ]
    
    
    a = np.cumsum(s) + z
    B = np.append(a,-a)
    A = np.append(np.tri(d,d), np.diag([1]*d), axis = 1)
    A = np.append(A,-A, axis = 0)
    A = np.append(A, np.reshape([-1]* (2*d), (2*d,1)), axis = 1)
    
    a1 = np.zeros((d,2*d + 1))
    for i in range(d):
        a1[i,i] = 1
    
    a1a2 = np.append(a1,-a1, axis = 0)
    A = np.append(A,a1a2, axis = 0)
    B = np.append(B,[1]*d)
    B = np.append(B,[0]*d)
    
    
    a1 = np.zeros((d,2*d + 1))
    for i in range(d):
        a1[i,i+d] = 1
    
    a1a2 = np.append(a1,-a1, axis = 0)
    A = np.append(A,a1a2, axis = 0)
    B = np.append(B,[1]*d)
    B = np.append(B,[0]*d)
    
    C = [0]*(2*d + 1)
    C[-1] = 1
    C = matrix(C,tc='d')
    A = matrix(A,tc='d')
    B = matrix(B,tc='d')
    
    # hint = matrix(hint, tc = 'd')
    # hint_s = matrix([1] * (3*d), tc = 'd')
    
    sol=solvers.lp(C,A,B,  solver = None)
    
    s_hat = np.array(sol['x']).flatten()[:d]
    s_hat = np.round(alpha*s_hat + (1-alpha) * s_hint)
    print((d - la.norm(s-s_hat, ord=1)) / d)
    output.append((d - la.norm(s-s_hat, ord=1)) / d)
print("d:", d)
print("\n\n\n")
print(np.mean(output))
print(np.std(output))

