import numpy as np


class Cov(object):
    def __init__(self):
        self.n, self.xm1, self.ym1, self.xym2 = 0, 0.0, 0.0, 0.0

    def step(self, x, y):
        n1 = self.n
        self.n += 1
        dx, dy = x - self.xm1, y - self.ym1
        self.xm1 += dx / self.n
        self.ym1 += dy / self.n
        self.xym2 += dx * dy * n1 / self.n

    def value(self):
        return self.xym2 / (self.n - 1)

if __name__ == '__main__':
    N = 1024
    x = np.random.rand(N)
    y = np.random.rand(N)
    cov = Cov()
    for k in range(N):
        cov.step(x[k], y[k])
    print cov.value()
    xm = x.mean()
    ym = y.mean()
    ref = np.dot(x - xm, y - ym) / (N - 1)
    print ref
