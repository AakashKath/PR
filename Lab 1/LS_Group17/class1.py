import matplotlib.pyplot as plt
import numpy as np

x, y=np.loadtxt('Class1.txt', delimiter=' ', unpack=True)
plt.plot(x, y, 'ro', label='Loaded from file!')

x, y=np.loadtxt('Class2.txt', delimiter=' ', unpack=True)
plt.plot(x, y, 'bo')

#x, y=np.loadtxt('Class2.txt', delimiter=' ', unpack=True)
#plt.plot(x, y, 'go')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Class1')
plt.legend()
plt.savefig(filename='Class1.png')
