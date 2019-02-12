import numpy as np
import matplotlib.pyplot as plt
import math

def tail_prob(chi,t):
    tail_Pr = []
    for i in range(0,len(t)):
		tail_Pr.append(chi[chi>(t[i]+np.mean(chi))].shape[0]/(1.0*len(chi)))#smaple mean = 0
    return tail_Pr

n=3
k = 10000
t = np.linspace(0,50,1e4)
data = np.zeros((k,))

###########################################################
#chi2 = Sigma(Xi^2) Xi -> N(0,1)
###########################################################

#data generation emperically
for i in range(n):
    x = np.random.normal(0,1,(k,))
    data = data + x**2


tail_probs = np.array(tail_prob(data,t))#computing the tail probability 
gaussian = np.random.normal(0,np.sqrt(2*n),(k,))
bound    =np.array(tail_prob(gaussian,t))#computing the bound

fig = plt.figure()
plt.plot(t,tail_probs,'b',label='Chi-square')
plt.plot(t,bound,'r',label='Reference Gaussian')
labels = ['Chi-square','Reference Gaussian']
plt.legend(labels)
fig.suptitle('N = '+str(n))
plt.xlabel('t')
plt.ylabel('Pr({X-u >t})')
plt.show()
