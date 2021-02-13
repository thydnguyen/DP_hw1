# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 21:03:15 2021

@author: Thy Nguyen
"""

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
from scipy.sparse import csr_matrix


solvers.options['show_progress'] = False
threshold = 0.5
thresholdHint = 1/3
n  = 20
d = 50000
alpha = 0.7
output = []
index = 500

def part( a, s_hint):
    d = len(a)
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
    return s_hat

    
for j in range(n):
    s = np.random.rand(d)
    
    z = np.random.rand(d)
    s = [1 if ss > threshold else 0  for ss in s  ]
    s_hint = np.array([ss if np.random.rand() > thresholdHint else 1-ss for ss in s])
    z =  [1 if zz > 0.5 else 0  for zz in z  ]
    
    
    a = np.cumsum(s) + z
    if  (d//index) * index == d:
        split_list = list(range(1, d // index + 1))
    else:
        split_list = list(range(1, d // index + 1 )) + [d / index]
    first_result = [part( a[0 :index ] , s_hint[0 :index ])] 
    result = [part( a[index * split_list[k] : int(index * split_list[k+1]) ] - a[index * split_list[k]  -1 ] , s_hint[index * split_list[k] : int(index * split_list[k+1]) ]) for k in range(len(split_list) -1 )  ]
    result_ = []
    for r in result:
        result_ = result_ + r.tolist()
    s_hat = np.append(first_result, result_).flatten()
    output.append((d - la.norm(s-s_hat, ord=1)) / d)
    print((d - la.norm(s-s_hat, ord=1)) / d)

    
print("d:", d)
print("\n\n\n")
print(np.mean(output))
print(np.std(output))

