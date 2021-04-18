# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:58:00 2021

@author: BLANCHEGORGE ANTOINE
"""
import numpy as np
from math import sqrt
import time
from numpy.linalg import norm
#from matplotlib.pylab import plot
import matplotlib.pyplot as plt 

 

def Cholesky(A):
    n, m = np.shape(A)
    L = np.zeros((n, n))
    for k in range(0,n):
        somme = 0
        for j in range (0,k):
            
            b = (L[k][j])*(L[k][j])
            
            somme += b
           
        L[k][k]= sqrt(A[k][k]-somme)
        #print("les termes diagonaux")
        #print(L[k][k])
        for i in range (k+1,n):
            
            somme2 = 0
            for j in range(0,k):
               
                B = L[i][j]*L[k][j]
                somme2 += B
               
            L[i][k]= (A[i][k]-somme2)/L[k][k]
            #print("les termes non diagonaux")
            #print(L[i][k])
       
       
    return L
        
        

 

        
        
O = np.array([[4,-2,-4],[-2,10,5],[-4,5,6]])
print(Cholesky(O)) 

 

P= np.array([[1,1,1,1],[1,5,5,5],[1,5,14,14],[1,5,14,15]])
print(Cholesky(P))      

 

"""
matrice non inversible    
Q = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
print(Cholesky(Q))
""" 
def ResolutionSystTriSup(N):
    
    n, m = np.shape(N)
    X = np.zeros((n, 1))
    X[n-1] = N[n-1][m-1]/ N[n-1][m-2]
    for i in range (n-2,-1,-1):
        X[i] = N[i][m-1]
        for j in range (i+1, n):
            X[i] = X[i]-N[i][j]*X[j]
        X[i]= X[i]/N[i][i]
    return X

 


def ResolutionSystTriInf(M):
    
    n, m = np.shape(M)
    Y = np.zeros((n, 1))
    Y[0] = M[0][m-1]/ M[0][0]
    for i in range(1, n):
        Y[i] = M[i][n]
        for j in range (0, i):
            Y[i] = Y[i]-M[i][j]*Y[j]
        Y[i]= Y[i]/M[i][i]
    return Y   

 


def ResolCholesky(A,B):
    n = len(A)
    L = Cholesky(A)
    Y = ResolutionSystTriInf(np.concatenate((L, B), axis = 1))
    Y = np.asarray(Y).reshape(n, 1)
    X = ResolutionSystTriSup(np.concatenate((np.transpose(L), Y), axis = 1))
   
    return X

 

O = np.array([[4,-2,-4],[-2,10,5],[-4,5,6]])
R = np.array([[6],[-9],[-7] ])
print(ResolCholesky(O,R))
K = np.array([[4,-2,-4],[-2,10,5],[-4,5,6]])
H = np.transpose(K)
print("Transposée")
print(H)
print(ResolCholesky(H,R))

 

def ResolCholeskyMachine(A,B):
    n = len(A)
    L = np.linalg.cholesky(A)
    Y = ResolutionSystTriInf(np.concatenate((L, B), axis = 1))
    Y = np.asarray(Y).reshape(n, 1)
    X = ResolutionSystTriSup(np.concatenate((np.transpose(L), Y), axis = 1))
    return X

 

O = np.array([[4,-2,-4],[-2,10,5],[-4,5,6]])
R = np.array([[6],[-9],[-7] ])
print(ResolCholeskyMachine(O,R))
print(np.linalg.solve(O,R))

 

#LU
def DecompositionLU(A):
    
    """
    Application de la méthode de Gauss
    """
    C = np.copy(A)
    n = len(C)
    L = np.eye(n)
    for k in range(0, n-1):
        for i in range(k+1, n):   
            g = C[i,k]/C[k,k]
            C[i] = C[i] - g*C[k]
            L[i,k] = g
    
    U = C            
          
    return U, L

 


def ResolutionLU(A, B):
    n = len(A)
    U, L = DecompositionLU(A)
    Y = ResolutionSystTriInf(np.concatenate((L, B), axis = 1))
    Y = np.asarray(Y).reshape(n, 1)
    X = ResolutionSystTriSup(np.concatenate((U, Y), axis = 1))
    
    return X

#Courbes

Taille = list()


Temps1 = list()
Norme1 = list()
Temps2 = list()
Norme2 = list()
Temps3 = list()
Norme3 = list()
Temps4 = list()
Norme4 = list()
Temps5 = list()
Norme5 = list()

for n in range(50,500,50):
    M = np.random.rand(n,n)
    A = np.dot(np.transpose(M),M)
    C = A 
    B = np.random.rand(n,1)
    """
    print(M)
    print(A)
    print(B)
    """
    t1= time.time()
    #print("A1",A)
    V = ResolCholesky(A,B)
    t2= time.time()
    t = t2-t1
    t3= time.time()
    Y = ResolutionLU(A,B)
    #print("A2",A)
    t4= time.time()
    u = t4-t3
    t5= time.time()
    Z = np.linalg.solve(A,B)
    #print("A3",A)
    t6= time.time()
    v = t6-t5
    t7= time.time()
    T = ResolCholeskyMachine(A,B)
    t8= time.time()
    w = t8-t7
    Temps1.append(t)
    Temps2.append(u)
    Temps3.append(v)
    Temps4.append(w)
    Taille.append(n)
   

 
    n1 = norm((A.dot(np.ravel(V)))-np.ravel(B))
    #print("n1",n1)
    n2 = norm((A.dot(np.ravel(Y)))-np.ravel(B))
    #print("n2",n2)
    n3 = norm((A.dot(np.ravel(Z)))-np.ravel(B))
    #print("n3",n3)
    n4 = norm((A.dot(np.ravel(T)))-np.ravel(B))
    #print("n4",n4)
    
    
    Norme1.append(n1)
    Norme2.append(n2)
    Norme3.append(n3)
    Norme4.append(n4)
   
    
 

 
"""
on enlève de commentaire pour afficher les courbes des normes et mettre un commentaire 
au niveau des variables de temps
Y1 = Norme1
Y2 = Norme2
Y3 = Norme3
Y4 = Norme4
"""
X= Taille
Y1 = Temps1
Y2 = Temps2
Y3 = Temps3
Y4 = Temps4
X2 = np.log(X)
Y5 = np.log(Temps1)
Y6 = np.log(Temps2)
Y7 = np.log(Temps3)
Y8 = np.log(Temps4)


 


plt.plot(X, Y1,label="Cholesky")
plt.plot(X, Y2,label="ResolutionLU")
plt.plot(X, Y3, label="linalg.solve")
plt.plot(X, Y4, label="linalg.cholesky")
plt.xlabel('taille de la matrice ', color='r')
plt.ylabel('temps en secondes ', color='r')
plt.title('Evolution temps de calcul en fonction de la valeur de n')
plt.grid(True)
plt.legend()
plt.show()


# courbe log 
plt.plot(X2, Y5,label="Cholesky")

#plt.plot(X2, Y6,label="ResolutionLU")

#plt.plot(X2, Y7, label="linalg.solve")

plt.plot(X2, Y8, label="linalg.cholesky")
plt.xlabel('ln de taille de la matrice ', color='r')
plt.ylabel('ln du temps en secondes ', color='r')
plt.title('Evolution lnt  en fonction de la valeur de lnn')
plt.grid(True)
plt.legend()
plt.show()


from scipy import stats
slope, intercept, r_value, p_value, std_error = stats.linregress(X2,Y5)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)



slope, intercept, r_value, p_value, std_error = stats.linregress(X2,Y8)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)


# courbe semi log 
plt.plot(X, Y5,label="Cholesky")

#plt.plot(X, Y6,label="ResolutionLU")

#plt.plot(X, Y7, label="linalg.solve")

plt.plot(X, Y8, label="linalg.cholesky")
plt.xlabel('Taille de la matrice ', color='r')
plt.ylabel('ln du temps en secondes ', color='r')
plt.title('Evolution lnt en fonction de la valeur de n')
plt.grid(True)
plt.legend()
plt.show()




slope, intercept, r_value, p_value, std_error = stats.linregress(X,Y5)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)




slope, intercept, r_value, p_value, std_error = stats.linregress(X,Y8)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)
