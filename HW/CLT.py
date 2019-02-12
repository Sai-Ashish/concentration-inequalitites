import numpy as np
import matplotlib.pyplot as plt
##########################################
#Central limit theorem
# suppose {X1,X2.....} is a sequence of i.i.d random variables with E[Xi]=mean and Var[Xi]=sigma^2 <infinty 
#then as n approaches infinity the random variable [sqrt(n)](Sn- mean) converges in distribution to a normal N(0,sigma^2)
#Sn=sigma(X)/n
##########################################
np.random.seed(42)

# Determine the number of IIDs taken for CLT approximation
n = 1000
# Number of samples being called for each IIDs
k = 10000

# Uniform Random Variable
uniform_rand = np.zeros((k,))
mean=0
for i in range(n):
    uniform_rand += np.random.uniform(-100,100,(k,))
uniform_rand = (uniform_rand-mean*n)/np.sqrt(n)


# Rayleigh RV with scale 1
Rayleigh = np.zeros((k,))
mean = np.sqrt(np.pi/2)
for i in range(n):
    Rayleigh += np.random.rayleigh(1, (k,))
Rayleigh = (Rayleigh-mean*n)/np.sqrt(n)

# Poisson RV with Lambda = 1 
Poisson = np.zeros((k,))
mean=1
for i in range(n):
    Poisson += np.random.poisson(1,(k,))
Poisson = (Poisson-n*mean)/np.sqrt(n)

#Exponential mean =0.9
exponential = np.zeros((k,))
mean=0.9
for i in range(n):
	exponential += np.random.exponential(0.9,(k,)) 
exponential = (exponential-n*mean)/np.sqrt(n)

clts = [uniform_rand,Rayleigh,Poisson,exponential]

fig, ax = plt.subplots(4, 1)

fig.suptitle('Central Limit Theorem (sqrt(n)(Sn-E[Xi])) -> N(0,Var(Xi))')
hist, bins = np.histogram(clts[0], bins=50, normed=True)
bin_centers = (bins[1:]+bins[:-1])*0.5
ax[0].plot(bin_centers, hist)
ax[0].set_title('Uniform', y=.6)


hist, bins = np.histogram(clts[1], bins=50, normed=True)
bin_centers = (bins[1:]+bins[:-1])*0.5
ax[1].plot(bin_centers, hist)
ax[1].set_title('Rayleigh', y=.6)

hist, bins = np.histogram(clts[2], bins=50, normed=True)
bin_centers = (bins[1:]+bins[:-1])*0.5
ax[2].plot(bin_centers, hist)
ax[2].set_title('Poisson', y=.6)

hist, bins = np.histogram(clts[3], bins=50, normed=True)
bin_centers = (bins[1:]+bins[:-1])*0.5
ax[3].plot(bin_centers, hist)
ax[3].set_title('exponential', y=.6)

plt.show()