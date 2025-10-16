import numpy as np 
import matplotlib.pyplot as plt
from reduction_partitio.matrice import M, n  
import random as rd 

def p(t):
    if t<100:
        return 0
    elif t<400:
        return 0.01*t 
    else : 
        return 4
   

def Tracage_bifurcation(delta, phi0, M, K, T, N):
    n = len(M) # get the size of the matrix M
    X = np.zeros((N+1, n)) # fix the shape of X array
    Y = np.zeros((N+1, n))
    h = T / (N+1)
    def calcul_energie(X,Y,k):
        s = 0

        for i in range(n):
            si = 0
            for j in range(n):
                si+= M[i,j]*X[k,j]

            s += - 0.25*si*X[k,i]
        """for i in range(n):
           s +=delta*0.5*Y[k,i]**2 #+K*0.25*X[k,i]**2+(delta-p(k*h))/2*X[k,i]**2
        return s"""  
    
    def calculate_sum_of_column(X, k):
        """Calculate the sum of the k-th column of X array."""
        return np.sum(X[k,:])
            
    def contrainte(x):
            criteremin= 1 #on pourra changer
            if abs(x)<criteremin:
                return x/criteremin
            else:
                return int(x>0)-int(x<0)
    for i in range(n):
        X[0, i] = rd.random()
        for k in range(N):
            X[k+1,i] =  contrainte(X[k,i] + h * delta * Y[k,i] )
            Y[k+1,i] = Y[k,i] - h * ((K * X[k,i]**2 - p(k*h) + delta) * X[k,i] + phi0 * calculate_sum_of_column(X,k))
    
    Temps = np.linspace(0, T, N+1)  # fix the linspace call
    P = [p(t) for t in Temps ]
    E = [calcul_energie(X,Y,k) for k in range(N+1)]

    plt.plot(Temps,P,label='P(t)')
    #plt.plot(Temps,E,label ='E(t)')
    plt.plot(Temps, X[:,1],label = 'X2(t)')  # plot the first column of X
    plt.plot(Temps,X[:,0],label='X1(t)')

    plt.legend()
    plt.show()

#on cherche a determiner les xi pour lesquels l'energie est minimale 

#il faut voir a quoi correspond les delta , k , phi 0 
N = 1000
delta = 0.5
phi0 = 0.1
K = 1
T=500
Tracage_bifurcation(delta,phi0,M,K,T,N)