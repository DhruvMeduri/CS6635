import numpy as np
import matplotlib.pyplot as plt
x1 = np.linspace(0, 1, 100)
y1 = np.sin(10*np.pi*x1) + np.sin(20*np.pi*x1)
add_noise = np.zeros([len(y1), len(y1)])
for row in range(len(y1)):
    for i, y in enumerate(y1):
        add_noise[row][i] = y + np.random.normal(0, abs(y)/2, 1)[0]
    plt.plot(x1, add_noise[row])
plt.title("Data with Noise")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

mean = np.mean(add_noise, axis=0)
std = np.std(add_noise, axis=0)
plt.plot(x1, mean, label='mean')
plt.fill_between(x1, mean - std, mean + std, alpha=0.3, label='std')
plt.title("Mean and standard deviation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
