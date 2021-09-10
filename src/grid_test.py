import numpy as np
import matplotlib.pyplot as plt


def gauss(mu, sigma, x, y):
    mu = np.array(mu)
    x = x - mu[0]
    y = y - mu[1]
    dist = np.sqrt(x * x + y * y)
    return np.exp(-(dist ** 2 / (2.0 * sigma ** 2)))

def mexhat(mu, x, y):
    mu = np.array(mu)
    x = x - mu[0]
    y = y - mu[1]
    dist = np.sqrt(x * x + y * y)

    sigma1 = 4
    sigma2 = 2

    g1 = np.exp(-(dist ** 2 / (2.0 * sigma1 ** 2)))
    g2 = np.exp(-(dist ** 2 / (2.0 * sigma2 ** 2)))

    return g1 - g2

def annulus(mu, x, y):
    mu = np.array(mu)
    x = x - mu[0]
    y = y - mu[1]
    dist = np.sqrt(x * x + y * y)

    f = np.zeros_like(dist)
    f[dist < 4] = 1
    f[dist < 2] = -np.inf
    return f

x, y = np.meshgrid(np.linspace(-50, 50, 1000), np.linspace(-50, 50, 1000))
a = annulus([0, 0], x, y)

x_idxs, y_idxs = np.where(np.random.rand(*a.shape) < a)
i = np.random.choice([i for i in range(len(x_idxs))])
x_idx = x_idxs[i]
y_idx = y_idxs[i]
x_val = x[0, x_idx]
y_val = y[y_idx, 0]

a1 = annulus([0, 0], x, y)
a2 = annulus([x_val, y_val], x, y)
a = a1 + a2
a[a == -np.inf] = 0
plt.imshow(a)
plt.show()

