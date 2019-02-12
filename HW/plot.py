import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

x=np.linspace(-1/2.0,1/2.0,1000)

plt.plot(x,np.exp(-1*x)/((1-x)),'r')
# plt.plot(x,np.exp(-1*x)/((np.sqrt(1-2*x))**4),'r')
# plt.plot(x,np.exp(-1*x)/(np.sqrt(1-2*x))**10,'r')
# plt.plot(x,np.exp(-1*x)/(np.sqrt(1-2*x))**100,'r')

plt.plot(x,np.exp(x**2),'b')

plt.show()