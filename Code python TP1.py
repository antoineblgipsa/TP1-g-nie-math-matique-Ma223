# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 23:27:10 2021

@author: 33752
"""

import numpy as np 
import time
from numpy.linalg import norm
#from matplotlib.pylab import plot
import matplotlib.pyplot as plt
from copy import deepcopy

# L'algorithme de Gauss
#Question1



Aaug1 = np.array([[1, 1, 1, 1, 1], [2, 4, -3, 2, 1], [-1, -1, 0, -3, 2], [1, -1, 4, 9, -8]])

def ReductionGauss(Aaug):
    
    ligne = np.shape(Aaug)[0]
    colonne = np.shape(Aaug)[1]
    if ligne+1 == colonne :
        #print("la matrice d'entrée est augmentée")
        for j in range(0, colonne-1):
            for i in range (0, ligne):
                #print ("colonne:",j, "ligne:", i)
                if i>j :
                    Aaug[i] = Aaug[i] -(Aaug[i][j]/Aaug[j][j])*Aaug[j]
                #print (Aaug,"\n")
        #print ("la solution est:\n",Aaug)
    else :
        print("La matrice d'entrée n'est pas augmentée")
    return Aaug

print(ReductionGauss(Aaug1))

#question 2

Taug1 = np.array([[1, 1, 1, 1, 1], [0, 2, -5, 0, -1], [0, 0, 1, -2, 3],[0, 0, 0, 4, -4]])




def ResolutionSystTriSup(Taug):
    l = np.shape(Taug)[0]
    c = np.shape(Taug)[1]
    X = np.zeros((l, 1))
    X[l-1] = Taug[l-1][c-1]/ Taug[l-1][c-2]
    for i in range (l-2,-1,-1):
        X[i] = Taug[i][c-1]
        for j in range (i+1,c-1):
            X[i] = X[i]-Taug[i][j]*X[j]
        X[i]= X[i]/Taug[i][i]
    return X



print(ResolutionSystTriSup(Taug1))

#question 3
def Gauss(A,B):
    C = np.concatenate((A,B),axis=1)
    D = ReductionGauss(C)
    sol =(ResolutionSystTriSup(D))
    return sol


a= np.array([[1,1,1,1], 
             [2,4,-3,2],
             [-1,-1,0,-3],
             [1,-1,4,9]])
b = np.array([[1], 
               [1],
               [2], 
               [-8]])
                                                       
    
print(Gauss(a,b))
"""
A1 = ([[0,2,-3], [-2,0,-3], [6,4,4]])
b1 = ([[2],
       [-5],
       [16]])
 
print(Gauss(A1,b1))
matrice qui marche pas avec Gauss simple mais marche avec pivot partiel
"""
#Question 4
Taille = list()

Temps1 = list()
Norme1 = list()


for n in range(50,100,50):
    A = np.random.rand(n,n)
    #print(A)
    B =  np.random.rand(n,1)
    #print(B)
    
    t1= time.time()
    Gauss(A,B)
    t2= time.time()
    t = t2-t1
    Temps1.append(t)
    Taille.append(n)
    
    N = Gauss(A,B)
    n1 = norm(np.ravel(A.dot(N))-np.ravel(B))
    Norme1.append(n1)
    
    
X= Taille
Y1 = Norme1
Y2 = Temps1



plt.plot(X, Y1,label="Gauss")
plt.xlabel('taille de la matrice ', color='r')
plt.ylabel('erreur ', color='r')
plt.title('Evolution erreur en fonction de la valeur de n')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(X, Y2,label="Gauss")
plt.xlabel('taille de la matrice ', color='r')
plt.ylabel('temps de calcul en secondes ', color='r')
plt.title('Evolution temps de calcul en fonction de la valeur de n')
plt.grid(True)
plt.legend()
plt.show()

# Décomposition LU
A = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
B = np.array([[7], [12], [3]])

def DecompositionLU(A):
    
    """
    Application de la méthode de Gauss
    """
    
    n = len(A)
    L = np.eye(n)
    for k in range(0, n-1):
        for i in range(k+1, n):   
            g = A[i,k]/A[k,k]
            A[i] = A[i] - g*A[k]
            L[i,k] = g
    
    U = A            
          
    return U, L

def ResolutionSystTriSup(Taug):
    
    n, m = np.shape(Taug)
    X = np.zeros((n, 1))
    X[n-1] = Taug[n-1][m-1]/ Taug[n-1][m-2]
    for i in range (n-2,-1,-1):
        X[i] = Taug[i][m-1]
        for j in range (i+1, n):
            X[i] = X[i]-Taug[i][j]*X[j]
        X[i]= X[i]/Taug[i][i]
    return X


def ResolutionSystTriInf(L):
    
    n, m = np.shape(L)
    Y = np.zeros((n, 1))
    Y[0] = L[0][m-1]/ L[0][0]
    for i in range(1, n):
        Y[i] = L[i][n]
        for j in range (0, i):
            Y[i] = Y[i]-L[i][j]*Y[j]
        Y[i]= Y[i]/L[i][i]
    return Y
    
def ResolutionLU(A, B):
    n = len(A)
    U, L = DecompositionLU(A)
    Y = ResolutionSystTriInf(np.concatenate((L, B), axis = 1))
    Y = np.asarray(Y).reshape(n, 1)
    X = ResolutionSystTriSup(np.concatenate((U, Y), axis = 1))
    
    return X
        
    
        
print(ResolutionLU(A, B))
        

#Variantes de l'algorithme de Gauss
#pivot partiel

def dilatation(A, i, a):
    """L[i] <-- a*L[i]"""
    A[i, :] *= a
    
def echange(A, i, j):
    """L[i] <--> L[j]"""
    L=A[i, :].copy()
    A[i, :] = A[j, :]
    A[j, :] = L

def transvection(A, i, j, m):
    """L[i] <- L[i]+m*L[j]"""
    A[i, :] += m * A[j, :]
    
def pivot_partiel(A, j):
    """Recherche de l'indice tel que |A[k,j]| soit maximal pour k>j"""
    indice = j
    for k in range(j + 1, A.shape[0]):
        if abs(A[k, j]) > abs(A[indice, j]):
            indice = k
    return indice

 

def GaussChoixPivotPartiel(A, B):
    n,p = A.shape
    assert n == p # A doit être carrée.
    # création la matrice C=(A B) avec conversion en flottants pour
    # éviter les problèmes de type.
    C = np.concatenate((A.astype(float), B.astype(float)), axis=1)
    # Echelonnement
    for j in range(n - 1):
        k = pivot_partiel(C, j)
        if k != j: echange(C, j, k) # Si le pivot est déjà à sa place,
        # inutile de faire l'échange.
        for i in range(j + 1, n):
            assert C[j, j] != 0 # A doit être inversible.
            mu = - C[i, j] / C[j, j]
            transvection(C, i, j, mu)
# Les coefficients diagonaux sont changés en 1.
    for i in range(n): dilatation(C, i, 1 / C[i, i])
# Fin de la résolution.
    for j in range(n - 1, 0, -1):
        for i in range(j): transvection(C, i, j, -C[i,j])
# Extraction de la solution
    return C[:, n:]
A1 = np.array( [[0,2,-3], [-2,0,-3], [6,4,4]])
b1 = np.array([[2],
               [-5],
               [16]])
print(GaussChoixPivotPartiel(A1,b1))

#pivot total
def GaussChoixPivotTotal(A,B):
    # vecteur de pivotation des solutions

    a=[[j for j in i] for i in A]
    b=[j for j in B]

    n=len(b)
    pivot_sol=[0.]*n
    x=[0.]*n

    for i in range(0,n,1):
        pivot_sol[i]=i

    for k in range(0,n-1,1):
        # max pour le pivot total
        ref=0.0
        for i in range(k,n,1):
            for j in range(k,n,1):
                if ref<a[i][j]:
                    ref=a[i][j]
                    ligne=i
                    colonne=j
                elif ref< -a[i][j]:
                    ref=-a[i][j]
                    ligne=i
                    colonne=j

        # pivotations
        for j in range(k,n,1):
            a[k][j], a[ligne][j] = a[ligne][j], a[k][j]
        b[k], b[ligne] = b[ligne], b[k]
        
        for i in range(0,n,1):
            a[i][k], a[i][colonne] = a[i][colonne], a[i][k]

        # remplissage du vecteur accorde aux pivotations
        pivot_sol[k], pivot_sol[colonne] = pivot_sol[colonne], pivot_sol[k]

        if a[k][k]==0.0:
            return []

        # reduction
        for i in range(k+1,n,1):
            p=a[i][k]/a[k][k]
            for j in range(k,n,1):
                a[i][j] -= p*a[k][j]
            b[i] -= p*b[k]

    # resolution
    for i in range(n-1,-1,-1):
        s=0.0
        for j in range(i+1,n,1):
            s+=a[i][j]*b[j]
        b[i]=(b[i]-s)/a[i][i]

    # pivotation des solutions
    for i in range(0,n,1):
        x[pivot_sol[i]]=b[i]

    return x

A = ([[1, 1, 1, 1], [2, 4, -3, 2, 1], [-1, -1, 0, -3, 2], [1, -1, 4, 9, -8]])
B = ([1,2,3,4 ])
print(GaussChoixPivotTotal(A, B))

a= np.array([[3,1,1,1], 
             [0,4,-3,2],
             [0,-1,0,-3],
             [0,-1,4,9]])
b = np.ravel( np.array([[1], 
               [1],
               [2], 
               [-8]]))
                                                       
    
print(GaussChoixPivotTotal(a, b))
 


