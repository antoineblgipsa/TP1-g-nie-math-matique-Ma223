# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:27:28 2021

@author: BLANCHEGORGE ANTOINE
"""

import numpy as np
from codepythonTP1 import Gauss
from codepythonTP1 import GaussChoixPivotPartiel
from codepythonTP1 import GaussChoixPivotTotal
from codepythonTP1 import ResolutionLU

 

 
import time
from numpy.linalg import norm
#from matplotlib.pylab import plot
import matplotlib.pyplot as plt
from copy import deepcopy
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

 

for n in range(50,1001,50):
    A = np.random.rand(n,n)
    #print(A)
    B =  np.random.rand(n,1)
    #print(B)
    
    t1= time.time()
    Gauss(A,B)
    t2= time.time()
    t = t2-t1
    t3= time.time()
    ResolutionLU(A, B)
    t4= time.time()
    u = t4-t3
    t5= time.time()
    GaussChoixPivotPartiel(A,B)
    t6= time.time()
    v = t6-t5
    t7= time.time()
    np.linalg.solve(A,B)
    t8= time.time()
    w = t8-t7
    t9= time.time()
    GaussChoixPivotTotal(A,B)
    t10= time.time()
    x= t10-t9
    Temps1.append(t)
    Temps2.append(u)
    Temps3.append(v)
    Temps4.append(w)
    Temps5.append(x)
    
    Taille.append(n)
    
    
    
    
    
    N = Gauss(A,B)
    n1 = norm(np.ravel(A.dot(N))-np.ravel(B))
    Norme1.append(n1)
    
    Y= ResolutionLU(A,B)
    n2 = norm(np.ravel(A.dot(Y))-np.ravel(B))
    Norme2.append(n2)
    
    Z=  GaussChoixPivotPartiel(A,B)
    n3 = norm(np.ravel(A.dot(Z))-np.ravel(B))
    Norme3.append(n3)
              
    L= np.linalg.solve(A,B)
    n4 = norm(np.ravel(A.dot(L))-np.ravel(B))
    Norme4.append(n4)
    
    U =  GaussChoixPivotTotal(A,B)
    n5 = norm(np.ravel(A.dot(U))-np.ravel(B))
    Norme5.append(n5)
    
   
    
   
    
X= Taille
X2= np.log(Taille)

 

"""
Enlever ce commentaire dans le cas de la norme 
Y1 = Norme1
Y2 = Norme2

 

Y3 = Norme3
Y4 = Norme4
Y5 = Norme5

 

Y6 = np.log(Norme1)
Y7 = np.log(Norme2)

 

Y8 = np.log(Norme3)
Y9 = np.log(Norme4)
Y11 = np.log(Norme5)

 

"""

 

Y1 = Temps1
Y2 = Temps2
Y3 = Temps3
Y4 = Temps4
Y5 = Temps5

 


Y6= np.log(Temps1)
Y7= np.log(Temps2)
Y8= np.log(Temps3)
Y10= np.log(Temps5)

 


plt.plot(X, Y1,label="Gauss")
plt.plot(X, Y2,label="ResolutionLU")
plt.plot(X, Y3, label="Pivotpartiel")
plt.plot(X, Y4, label="linalg")
plt.plot(X, Y5, label="pivottotal")
plt.xlabel('taille de la matrice ', color='r')
plt.ylabel('erreur ', color='r')
plt.title('Evolution erreur en fonction de la valeur de n')
plt.grid(True)
plt.legend()
plt.show()

 

 


#ln (pour le temps)
plt.plot(X2,Y6,label="Gauss")
plt.plot(X2,Y7,label="ResolutionLU")
plt.plot(X2,Y8,label="Pivotpartiel")
plt.plot(X2,Y10,label="pivottotal")
plt.xlabel('lnn ', color='r')
plt.ylabel( 'lnt en secondes', color='r')
plt.title('Evolution de lnt en fonction de lnn')
plt.grid(True)
plt.legend()
plt.show()

 

"""
Enlever ce commentaire pour afficher ln de l'erreur 
plt.plot(X2, Y6,label="Gauss")
plt.plot(X2, Y7,label="ResolutionLU")
plt.plot(X2, Y8, label="Pivotpartiel")
plt.plot(X2, Y9, label="linalg")
plt.plot(X2, Y11, label="pivot total")
plt.xlabel('ln taille de la matrice ', color='r')
plt.ylabel('ln erreur ', color='r')
plt.title('Evolution erreur en fonction de la valeur de n')
plt.grid(True)
plt.legend()
plt.show()
"""

 

#utilisé pour le temps
from scipy import stats
slope, intercept, r_value, p_value, std_error = stats.linregress(X2,Y6)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)

 


slope, intercept, r_value, p_value, std_error = stats.linregress(X2,Y7)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)

 

slope, intercept, r_value, p_value, std_error = stats.linregress(X2,Y8)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)

 

slope, intercept, r_value, p_value, std_error = stats.linregress(X2,Y10)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)

 

#demi-ln (pour le temps)

 

plt.plot(X,Y6,label="Gauss")
plt.plot(X,Y7,label="ResolutionLU")
plt.plot(X,Y8,label="Pivotpartiel")
plt.plot(X,Y10,label="pivottotal")
plt.xlabel('n ', color='r')
plt.ylabel( 'lnt en secondes', color='r')
plt.title('Evolution de lnt en fonction de n')
plt.grid(True)
plt.legend()
plt.show()

 

#utilisé pour le temps
slope, intercept, r_value, p_value, std_error = stats.linregress(X,Y6)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)

 


slope, intercept, r_value, p_value, std_error = stats.linregress(X,Y7)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)

 

slope, intercept, r_value, p_value, std_error = stats.linregress(X,Y8)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)

 

slope, intercept, r_value, p_value, std_error = stats.linregress(X,Y10)
print ('pente :',slope)
print ("ordonnée à l'origine :", intercept)
print ('coefficient de corrélation r :',r_value)