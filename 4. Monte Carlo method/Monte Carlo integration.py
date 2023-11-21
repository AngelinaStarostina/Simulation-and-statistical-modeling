import math
import random
from scipy import integrate
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def f1(x):
    return math.exp(-x**4) * math.sqrt(1 + x**4)


def f2(x, y):
    return 1 / (x**2 + y**2)


def MonteCarloMethod(n, f):
    fn = []
    for i in range(n):
        el = random.normalvariate(0, 1)
        fn.append(f(el) / stats.norm.pdf(el))
    sum = 0
    for i in range(n):
        sum += fn[i]
    return 1 / n * sum


def MonteCarloMethodRing(n, f, D, d):
    S = math.pi * (D**2 - d**2)
    sum = 0
    k = n
    while k > 0:
        el1 = random.uniform(-D, D)
        el2 = random.uniform(-D, D)
        if 1 <= el1**2 + el2**2 < 4:
            sum += f(el1, el2)
            k -= 1
    return S / n * sum


print("Решение методом Монте-Карло: ", MonteCarloMethod(1000, f1))
r = integrate.quad(f1, -math.inf, math.inf)[0]
print("Приближенное значение интеграла: ", r)
f = [math.fabs(MonteCarloMethod(i, f1) - r) for i in range(100, 2000, 10)]
plt.plot([i for i in range(100, 2000, 10)], f)
plt.show()

print()
print("Решение методом Монте-Карло: ", MonteCarloMethodRing(1000, f2, 2, 1))
print("Приближенное значение интеграла: ", 2 * math.pi * math.log(2))
f = [math.fabs(MonteCarloMethodRing(i, f2, 2, 1) - 2 * math.pi * math.log(2)) for i in range(100, 5000, 10)]
plt.plot([i for i in range(100, 5000, 10)], f)
plt.show()
