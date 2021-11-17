#!/usr/bin/env python
# coding: utf-8

# # <font color=red>TP1 :  Résolution de l'équation non-linéaire </font>

# ##  <font color=green>Exercice 1 : </font>

# In[5]:


def f(x):
    return   x**6-6*x**5+15*x**4-20*x**3+15*x**2-6*x+1


# In[6]:


def p(x):
    return (x-1)**6


# In[7]:


p(0.998)


# In[8]:


f(0.998)


# <b> p(x) et f(x) sont deux fonctions égaux mais ils n'ont pas les même résultats se qui montre que l'ordinateur ne fait pas le calcule correctement car l'ordinateur code seulement sur 32 ou 64 bits alors il va coder seulement une partie du nombre calculé il fait l'arrondit de cette valeur c'est l'erreur de codage d'où les valeurs calculer par l'ordinateur sont fausses . 

# ###  <font color=purple> la méthode Horner
#  </font>
# 

# In[9]:


def horner(a,x):
    n=len(a)
    valeur=a[1]
    for i in range(2,n):
        valeur=valeur*x+a[i]
    return valeur


# In[10]:


def f(x):
    return   x**6-6*x**5+15*x**4-20*x**3+15*x**2-6*x+1


# In[11]:


def p(x):
    return (x-1)**6


# In[12]:


T=[1,-6,15,-20,15,-6,1]
print(T)
print('--------------p(i)----------')
ii=[0.998,0.001,1.002]
for i in ii:
    print('ans=',p(i)) 
print('--------------f(i)----------')
ii=[0.998,0.001,1.002]
for i in ii:
    print('ans=',f(i)) 
print('----------horner---------')
ii=[0.998,0.001,1.002]
for i in ii:
    print('ans=',horner(T,i)) 


# <b> si on remplace la valeur à calculer directement dans la fonction donc on va faire presque 26 opérations pour cette raison le mathématicien Horner utilise son algorithme pour calculer la valeur donnée avec la moitié des opérations utilisé

# ###  <font color=red> Représentation graphique de 𝑓(𝑥) et 𝑝(𝑥)
#  </font>

# ####  <font color=green> Représentation graphique de la fonction px
#  </font>

# In[14]:


import matplotlib.pyplot as plt
import numpy as np
x1= np.arange(0.998,1.002,0.0001)
plt.xlabel("x")
plt.ylabel("px")
plt.grid(True)
plt.plot(x1,p(x1))


# ####  <font color=green> Représentation graphique de la fonction fx
#  </font>

# In[15]:


import matplotlib.pyplot as plt
import numpy as np
x2= np.arange(0.998,1.002,0.0001)
plt.xlabel("x")
plt.ylabel("fx")
plt.grid(True)
plt.plot(x2,f(x2),color='r')
plt.plot(x1,p(x1),'*')


# <b> l'erreur d'arrondi est la différence entre la valeur approchée calculée d'un nombre et sa valeur mathématique exacte. 
#     
#    Des erreurs d'arrondi naissent généralement lorsque des nombres exacts sont représentés dans un système incapable de les exprimer exactement.
#   Les erreurs d'arrondi se propagent au cours des calculs avec des valeurs approchées ce qui peut augmenter l'erreur du résultat final.
#     
# La représentation graphique de p(x) : plus clair , plus lisible que celle de f(x)

# ##  <font color=green>Exercice 2 : </font>
# 

# <b> On considère l'équation :f(𝑥)=𝑥3+𝑥2−3𝑥−3=0
# 

# In[16]:


def f(x):
    return  x**3+x**2-3*x-3


# <b> 1) Dessiner la courbe de 𝑓 dans l'intervalle [−2;2] , puis trouver des intervalles convenables pour appliquer la méthode de >bisection.

# In[17]:


import matplotlib.pyplot as plt
from numpy import linspace
f=lambda x: x**3+x**2-3*x-3
T=linspace(-2,2,41)
plt.title("solution dans [-2:2] ");
plt.xlabel("x")
plt.ylabel ("f(x)")
plt.grid(True)
plt.plot(T,f(T))


# <b> Daprés le graph de $f$ il existe 3 solutions :
#     
# $\alpha_1 \in ]-2;-1.5[$
#     
# $\alpha_2 \in ]-1.5;-0.5[$
#     
# $\alpha_3 \in ]1.5;2[$

# <b> 2) Utiliser un programme dicho.m qui permettra de trouver les solutions de l'équation 𝑓(𝑥)=0 avec 𝑒𝑝𝑠=0.001 .

# 
# ###  <font color=red> la methode dichotomie
#  </font>
# 

# In[18]:


def dicho(a,b,f,n):
    m=(a+b)/2
    err=abs(b-a)
    while err>n:
        if m==0:
            break
        if f(a)*f(b)<0:
            b=m
        else:
            a=m
            m=(a+b)/2
            err=abs(b-a)
    return m


# In[19]:


x3=dicho(-2,-1.5,f,10)
#x10=dicho(-2,-1.5,f,10)
#x20=dicho(-2,-1.5,f,20)
#print(x3,x10,x20)
print(x3)


# 
# ###  <font color=red> la 2éme methode dichotomie
#  </font>
# 

# In[20]:


def madichotomie(ff,a,b,e):
    xg=a
    xd=b
    while (xd-xg)>e:
        xm=(xg+xd) / 2
        if f(xg)*f(xm) >0:
            xg=xm
        else:
            xd=xm
    sol=(xg+xd)/2
    return sol
    


# In[21]:


res=madichotomie(f,-2,-1.5,0.001)
res


# <b> 3) Modifier le programme dicho.m pour qu'il donne le nombre d'itération nécessaire pour avoir la solution à 𝑒𝑝𝑠 près.

# In[22]:


def dichow(a,b,f,eps):
    cp=0
    while (b-a)>eps:
        m=(a+b)/2
        if f(m)==0:
            return m
        if f(a)*f(m)<0 :
            b=m
        else :
            a=m
        cp+=1
    return m,cp


# $\alpha_1 \in ]-2;-1.5[$
# 
# $\alpha_2 \in ]-1.5;-0.5[$
# 
# $\alpha_3 \in ]1.5;2[$

# In[23]:


dichow(-2,-1.5,f,0.0001)


# In[24]:


interval=[[-2,-1.5],[-1.5,-0.5],[1.5,2]]
for u in interval:
    print(dichow(u[0],u[1],f,10**(-3)))
################################################   
print(dichow(-2,-1.5,f,10**(-3)))
print(dichow(-1.5,-0.5,f,10**(-3)))
print(dichow(1.5,2,f,10**(-3)))


# <b> Le nombre d'itération N nécéssaire à la méthode de la dichotomie pour trouver $\alpha$ à $10^{-p}$ prés

# In[25]:


#alpha 1
dichow(-2,-1.5,f,10**(-5))
#à 10^-8


# In[26]:


#alpha 2
dichow(-1.51,-0.5,f,10**(-5))
#à 10^-8


# <b> Transformer le code précédent pour trouver desvaleurs approchées des solutions $\alpha_1$, $\alpha_2$ et $\alpha_3$ de l'equation (E)

# ###  <font color=red> Conclusion
#  </font>
# 

# <b> La méthode de dichotomie ou méthode de la bissection est un algorithme de recherche d'un zéro d'une fonction qui consiste à répéter des partages d’un intervalle en deux parties puis à sélectionner le sous-intervalle dans lequel existe un zéro de la fonction.

# ##  <font color=green>Exercice 3 : </font>
# 

# <b> 1) Dessiner la courbe de 𝑓 dans l'intervalle [1;2] , puis trouver une valeur approché de la solution à 0.001 près (En >utilisant la fonction zoom du graphique)

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
#import mpld3
#mpld3.enable_notebook()
import numpy as np 
import matplotlib.pyplot as plt

f=lambda x:x**3+4*x**2-10

t=np.linspace(1,2,100)
plt.subplot(2,2,1)
plt.plot(t,f(t),'r')
plt.grid(True)
##########################################
t=np.linspace(1.2,1.4,100)
plt.subplot(2,2,2)
plt.plot(t,f(t),'r')
plt.grid(True)
##########################################
t=np.linspace(1.35,1.4,100)
plt.subplot(2,2,3)
plt.plot(t,f(t),'r')
plt.grid(True)
#########################################
t=np.linspace(1.365,1.367,100)
plt.subplot(2,2,4)
plt.plot(t,f(t),'r')
plt.grid(True)


# In[28]:


from sympy import *
import numpy as np
x, y, z = symbols('x y z')
init_printing(use_unicode=True)


# In[29]:


def g1(x):
    return 1/2*sqrt(10-x**3)

g1=lambda x: 1/2*sqrt(10-x**3)
dg1=lambdify(x,diff(g1(x),x,1))
dg1(1.365)


# <b> Que pouvez vous conclure à propos des convergences de g1:

# Puisque la valeur de g1 est négatife et la valeur absolu de g1 |g1(x)| est inférieur à 1 donc la méthode de point fixe converge totalemen, c'est à dire, il exixteun locale I dans l'intervale [1,2].
# 
# 

# In[30]:


from math import sqrt

g1=lambda x: 1/2*sqrt(10-x**3)
g2=lambda x: sqrt(10./(x+4))
g3=lambda x: x-x**3-4*x**2 + 10


# In[31]:


def pointfixe(g,x0,eps,nmax):
    zero=x0
    for niter in range(0,nmax):
        x=zero
        zero=g1(x)
        erreur=abs(zero-x)
        if erreur<eps: 
            return x


# In[32]:


print(pointfixe(g1,1.5,0.001,50))


# In[33]:


print(pointfixe(g2,1.5,0.001,50))


# In[34]:


print(pointfixe(g3,1.5,0.001,50))


# ##  <font color=green>Exercice 4 : </font>
# 

# In[35]:


def newton(a,eps):
    x1=a
    x2=x1-f(x1)/fprime(x1)
    
    while abs(x1-x2)>eps:
        x1=x2
        x2=x1-f(x1)/fprime(x1)
        sol=x2
    return sol


# In[36]:


#importation
import matplotlib.pyplot as plt
from numpy import linspace
from math import exp, expm1
import numpy as np 


# In[37]:


def f(x):
    return exp(-x)-x

def fprime(x):
    return -exp(-x)-1

f2 = np.vectorize(f)
t=np.linspace(0,1,100)
plt.plot(t,f2(t),'y')
plt.show()


# In[38]:


print(newton(0,0.001))


# In[39]:


print(newton(1,0.001))


# ###  <font color=red> Conclusion
#  </font>

# <b> Après les trois méthodes que nous avons déjà vues (Dichotomie,Point fixe et Newton), nous remarquons que la méthode de Newton est la plus rapide et la plus simple, ce qui nous donne un taux d'erreur très faible, tandis que la méthode de point fixe donne un taux d'erreur moyen et la dernière, qui est la méthode de dichotomie la plus lente qui est très lente et prend beaucoup de temps Pour les calculer en plus d'un grand nombre d'erreurs.

# In[ ]:




