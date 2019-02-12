import numpy as np
import matplotlib.pyplot as plt
import math

def tail_prob(dist,t):
    tail_Pr = []
    for i in range(0,len(t)):
		tail_Pr.append(dist[dist>(t[i]+np.mean(dist))].shape[0]/(1.0*len(dist)))
		#subtracting the smaple mean from the random variable to make it centered 
    return tail_Pr

    
np.random.seed(44)

k = 10000
t = np.linspace(0,150,1e5)
data = np.zeros((k,))

###########################################################

#data generation emperically
p=0.75
n=500
for i in range(n):
    x = np.random.binomial(1, p, (k,))
    data = data + x

###########################################################

tail_probs = np.array(tail_prob(data,t))#computing the tail probability 

#distribution standard deviation
v=n*p*(1-p)

b=1-p
u=(b/v)*t

lamda=0.155
mgf=(np.exp(-1.0*lamda*p)*(1.0-p+p*np.exp(lamda)))

chernoff_bound= (np.exp(-1.0*lamda*t))*mgf

bennett_bound=np.exp(-1.0*(v/b**2)*((1+u)*np.log(1+u)-u))

hoeffdings_bound=np.exp(-2.0*(t**2)/(n))

###########################################################

fig = plt.figure()
plt.plot(t,tail_probs,'b')
plt.plot(t,bennett_bound,'r')
plt.plot(t,hoeffdings_bound,'g')
plt.plot(t,chernoff_bound,'c')
labels = ['Tail probability Pr({X-u >t}','Bennetts Bound','Hoeffdings Bound','Chernoff bound']
plt.legend(labels)
fig.suptitle('Sharpness of Bounds')
plt.xlabel('t')
plt.ylabel('Tail probabilities Pr({X-u >t} and bounds')
plt.show()
