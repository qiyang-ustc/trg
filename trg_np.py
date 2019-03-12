import numpy as np
import numpy.linalg as la
import math

def trg():   
    Jcp = 0.5
    n = 30
    dim = 2
    dim_cut = 24
    epsilon = 1E-12
    
    T = np.array([[math.exp(Jcp), math.exp(-Jcp)], [math.exp(-Jcp), math.exp(Jcp)]])
    U, S, V = la.svd(T)
    
    U = U * np.sqrt(S)
    V = V * np.sqrt(S)
    
    T = np.einsum('ai,aj,ak,al->ijkl', U, U, V, V)
    
    lnZ = 0
    for iter in range(n):
    
        #maxval = T.__abs__().max()
        maxval = np.abs(T).max()
        T = T/maxval
        lnZ += 2**(n-iter)*math.log(maxval)
    
        A = np.transpose(T, (1, 2, 3, 0))
        B = np.transpose(T, (2, 3, 1, 0))
        A = A.reshape(dim ** 2, dim ** 2)
        B = B.reshape(dim ** 2, dim ** 2)
        #A = np.array([[T[i, j, k, l] for i in D for k in D] for i in D for l in D])
        #B = np.array([[T[i, j, k, l] for l in D for k in D] for j in D for i in D])
    
        Ua, Sa, Va = la.svd(A)
        Ub, Sb, Vb = la.svd(B)
        Va = Va.T
        Vb = Vb.T
    
        dim_new = min(min(dim ** 2, dim_cut), min((Sa > epsilon).sum(), (Sb > epsilon).sum()))
    
        T3 = (Ua[:, :dim_new] * np.sqrt(Sa[:dim_new])).reshape(dim, dim, dim_new)  #12
        T4 = (Ub[:, :dim_new] * np.sqrt(Sb[:dim_new])).reshape(dim, dim, dim_new)  #23
        T1 = (Va[:, :dim_new] * np.sqrt(Sa[:dim_new])).reshape(dim, dim, dim_new)  #34
        T2 = (Vb[:, :dim_new] * np.sqrt(Sb[:dim_new])).reshape(dim, dim, dim_new)  #jk 
    
       # T = np.einsum('ajc,daf,gdi,gjl->cfil', T3, T4, T1, T2)
        T3 = np.tensordot(T3, T4, axes=(0, 1))   #jc,df,gdi,gjl
        T1 = np.tensordot(T1, T2, axes=(0, 0))   #jc,df,di,jl
        T = np.tensordot(T1, T3, axes=(2, 0))    #jc,f,i,jl
        T = np.einsum('jcfijl->cfil', T)
    
    
        dim = dim_new
    
        #print('T:FullTrack', (math.log(np.einsum('ijij->', T ))+lnZ/2**(n-iter-1))/2**(iter+1))
    
    lnZ += math.log(np.einsum('ijij->', T ))
    
    print(lnZ/2**n)
    

#print('Level1: Spin:2', math.log(np.einsum('ai,ai,ai,ai->', U, U, V, V)))
#print('Level2:', math.log(np.einsum('ijkl,alcj,keig,cgea->',T,T,T,T))/4.0)

if __name__ == "__main__":
    trg()
