import numpy as np
import matplotlib.pyplot as plt
import math

def tail_prob(dist,t):
    tail_Pr = []
    for i in range(0,len(t)):
		tail_Pr.append(dist[dist>(t[i]+np.mean(dist))].shape[0]/(1.0*len(dist)))
		#subtracting the smaple mean from the random variable to make it centered 
    return tail_Pr

np.random.seed(42)
k = 10000
t = np.linspace(0,30,1e4)
data = np.zeros((k,))

###########################################################

#data generation emperically
data = np.random.normal(0,1,(k,))


###########################################################
tail_probs = np.array(tail_prob(data,t))#computing the tail probability 
gaussian = np.random.normal(0,2,(k,))
bound    =np.array(tail_prob(gaussian,t))#computing the bound


###########################################################

fig = plt.figure()
plt.plot(t,tail_probs,'b')
plt.plot(t,bound,'r')
labels = ['Gaussain N(0,1)','Reference Gaussian']
plt.legend(labels)
fig.suptitle('Gaussian N(0,1)')
plt.xlabel('t')
plt.ylabel('Pr({X-u >t})')
plt.show()
