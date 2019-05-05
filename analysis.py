import numpy as np
import scipy
import math

def ostrowski(I,A,B,groups,epsilon=1.0e-8,tracker=0):
    
    assert(A.shape==B.shape)
    
    u0 = [] 
    v0 = []
    for i in range(groups*(I+1)):
        i = np.random.random()
        u0 = np.append(u0,i)
        v0 = np.append(v0,i)
    while not np.dot(v0,u0):
        i = np.random.random()
        u0 += i
        v0 += i
    
    u0 /= np.linalg.norm(u0)
    v0 /= np.linalg.norm(v0)
    
    C = np.matmul(np.linalg.inv(A),B)
    converged = 0
    iteration = 1
    if tracker:
        trace = np.array([])
    rho0 = 0    
    while not converged:
        rho1 = np.dot(v0,np.dot(C,u0))/np.dot(v0,u0)
        if tracker:
            trace = np.append(trace,rho1)
        temp = C - rho1*np.eye(A.shape[0])
        try:
            u1 = np.linalg.solve(temp,u0)
            v1 = (np.linalg.solve(temp.T,v0))
        except LinAlgErr as err:
            u1 = np.linalg.solve(temp,np.zeros([u0.shape[0],1]))
            v1 = (np.linalg.solve(temp.T,np.zeros([v0.shape[0],1])))
            phi = np.reshape(u1,(I+1,groups),order='F')
            return phi, 1/rho1
        print('iteration:',iteration,'rho:',rho1)
        converged = (np.fabs(rho0-rho1) < epsilon)  
        u1 /= np.linalg.norm(u1)
        v1 /= np.linalg.norm(v1)
        rho0 = rho1
        u0 = u1
        v0 = v1
        iteration += 1
    phi = np.reshape(u1,(I+1,groups),order='F')
    if tracker:
        return phi,1/rho1,trace
    return phi, 1/rho1
    
def rayleighQuotient(I,A,B,groups,epsilon=1.0e-8,tracker=0):
    assert(A.shape == B.shape)
    
    # Initial Guess of x0
    x0 = [] 
    for i in range(groups*(I+1)):
        i = np.random.random()
        x0 = np.append(x0,i)
        
    # Normalize x0
    l_old = np.linalg.norm(x0)
    x0 /= l_old
    C = np.dot(np.linalg.inv(A),B)
    converged = 0 
    iteration = 1
    if tracker:
        trace = np.array([])

    while not(converged):
        x0 = x0/np.linalg.norm(x0)
        lam = np.dot(x0,np.dot(C,x0))
        temp = C-np.dot(lam,np.eye(A.shape[0]))
        x1 = np.linalg.solve(temp,x0)
        phi_change = np.linalg.norm(x1-x0)
        print('iteration:',iteration,'lam:',lam)
        converged = (abs(l_old-lam) < epsilon)
        x0 = x1
        if tracker:
            trace = np.append(trace, lam)
        l_old = lam
        iteration += 1
    phi = np.reshape(x1,(I+1,groups),order='F')
    if tracker:
        return phi,1/lam,trace
    return phi, 1/lam

def inversePower(I,A,B,groups,epsilon=1.0e-8, LOUD=0, tracker=0):
    ''' Solve the generalized eigenvalue problem 
    Ax = (1/k)Bx using inverse power iteration
    Inputs:
        I: the number of cells, (I+1) = N
        A: left-side (groups*N)x(groups*N) matrix
        B: right-side (groups*N)x(groups*N) matrix
        groups: for dimensional purposes 
    Outputs:
        l_new: the smallest eigenvalue of the problem
        phi: the associated eigenvector, broken up into Nxgroups matrix
    '''
    assert(A.shape == B.shape)
    
    # Initial Guess of x0
    x0 = [] 
    for i in range(groups*(I+1)):
        i = np.random.random()
        x0 = np.append(x0,i)

    # Normalize x0
    l_old = np.linalg.norm(x0)
    x0 /= l_old

    converged = 0 
    iteration = 1
    if tracker:
        trace = np.array([])
    while not(converged):
        if tracker:
            trace = np.append(trace,1/l_old)
        x1 = np.linalg.solve(A,np.dot(B,x0))
        l_new = np.linalg.norm(x1)/np.linalg.norm(x0)
        x1 = x1/np.linalg.norm(x1)
        converged = (np.fabs(l_new-l_old) < epsilon) 
        x0 = x1
        l_old = l_new
        if (LOUD>0):
            print('Iteration:',iteration,'\tMagnitude of l',l_new)
        iteration += 1
    phi = np.reshape(x1,(I+1,groups),order='F')
    if tracker:
        return phi, 1/l_new,trace
    return phi, 1/l_new
 
def power(I,A,B,groups,epsilon=1.0e-8, tracker=0):
    # Initial Guess of x0
    x0 = [] 
    for i in range(groups*(I+1)):
        i = np.random.random()
        x0 = np.append(x0,i)
        
    # Normalize x0
    l_old = np.linalg.norm(x0)
    x0 /= l_old
    
    # Combine A and B into A^-1 B
    c = np.linalg.inv(A)@B

    converged = 0 
    iteration = 1
    if tracker:
        trace = np.array([])
        
    while not(converged):
        if tracker:
            trace = np.append(trace,l_old)
        y1 = c @ x0
        mu = np.linalg.norm(y1)
        x1 = y1/mu
        l_new = np.linalg.norm(y1)/np.linalg.norm(x0)
        
        print('Iteration:',iteration,'\tMagnitude of l',l_new)
        converged = (np.fabs(l_new-l_old) < epsilon) 
        
        x0 = x1
        l_old = l_new
        iteration += 1
    phi = np.reshape(x1,(I+1,groups),order='F')
    if tracker:
        return phi, 1/l_new,1/trace
    return phi, 1/l_new

def qr(I,A,B,groups,max_iterations,all_val=0):
    C = np.linalg.inv(A) @ B
    # intialize values
    eigenvalue = np.zeros(max_iterations)
    [rows,columns] = C.shape
    # convert matrix to hessenberg matrix 
    H = scipy.linalg.hessenberg(C)
    Q,R = scipy.linalg.qr(H)
    H = R@Q
    for i in range (max_iterations):
        #check for smallest eigenvalue
        smallest_eigenvalue = H[0,0]
        for k in range(2,rows):
            if H[k,k] < smallest_eigenvalue and H[k,k] > 0:
                smallest_eigenvalue = H[k,k]
        eigenvalue[i] = smallest_eigenvalue
    if all_val:
        all_val = np.array([])
        for i in range(len(H)):
            all_val = np.append(all,H[i,i])
        return smallest_eigenvalue,all_val
    return smallest_eigenvalue
