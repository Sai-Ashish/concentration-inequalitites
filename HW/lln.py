import numpy as np 
import matplotlib.pyplot as plt 


#We show that as the number of samples increase Sn/n -> mean of Xi (weak law of large numbers)

n=10000
gaussian = []
exponential = []
uniform = []
laplacian =[]

#Gaussian
mu = 0
for i in range(1,n+1):
	data = np.random.normal(mu,1.0,i)
	gaussian.append(np.mean(data))


#Exponential
mu = 0.9
for i in range(1,n+1):
	data = np.random.exponential(mu,i)
	exponential.append(np.mean(data)) 


#Uniform
mu = 0
for i in range(1,n+1):
	data = np.random.uniform(-1,1,i)
	uniform.append(np.mean(data))

#Laplacian
mu = 0
for i in range(1,n+1):
	data = np.random.laplace(mu,1.0,i)
	laplacian.append(np.mean(data))

figure,axs = plt.subplots(4,sharex=True )
figure.suptitle('Law of Large Numbers')


###############
#plot
###############

#Gaussian
#X_n= sigma(X_i)/n
axs[0].plot(gaussian,label='Sn/n')
#actual mean
axs[0].plot(np.zeros(n),label='Actual mean')
axs[0].legend(loc="upper right")
axs[0].set_title('Gaussian', y=.6)

#Exponential
#X_n= sigma(X_i)/n
axs[1].plot(exponential,label='Sn/n')
#actual mean
axs[1].plot(0.9*np.ones(n),label='Actual mean')
axs[1].legend(loc="upper right")
axs[1].set_title('Exponential',y=.6)

#laplacian
#X_n= sigma(X_i)/n
axs[2].plot(laplacian,label='Sn/n')
#actual mean
axs[2].plot(np.zeros(n),label='Actual mean')
axs[2].legend(loc="upper right")
axs[2].set_title('Laplacian', y=.6)

#uniform
#X_n= sigma(X_i)/n
axs[3].plot(uniform,label='Sn/n')
#actual mean
axs[3].plot(np.zeros(n),label='Actual mean')
axs[3].legend(loc="upper right")
axs[3].set_title('Uniform', y=.6)

plt.show()
