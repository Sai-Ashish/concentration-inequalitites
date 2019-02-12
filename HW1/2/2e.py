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
t = np.linspace(0.5,1000,1e5)
data = np.zeros((k,))

###########################################################

#data generation emperically
a=10
n=10
for i in range(n):
    x = np.random.uniform(-a,a,(k,))
    data = data + x

###########################################################

tail_probs = np.array(tail_prob(data,t))#computing the tail probability 
std=10*n#distribution standard deviation
gaussian = np.random.normal(0,std,(k,))
bound    =np.array(tail_prob(gaussian,t))#computing the bound

###########################################################

fig = plt.figure()
plt.plot(t,tail_probs,'b')
plt.plot(t,bound,'r')
labels = ['Sum of bounded random variables(Uniform [-10,10])','Reference Gaussian']
plt.legend(labels)
fig.suptitle('Sum of bounded random variables with zero mean is sub gaussian')
plt.xlabel('t')
plt.ylabel('Pr({X-u >t})')
plt.show()
