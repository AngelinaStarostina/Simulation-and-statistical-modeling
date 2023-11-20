import numpy as np
from scipy import stats
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class MultiplicativeCongruentMethod:

    def __init__(self, a, b, m):
        self.prev_el = a
        self.beta = b
        self.M = m

    def next_element(self):
        z = self.beta * self.prev_el
        self.prev_el = z - self.M * int(z / self.M)
        return self.prev_el / self.M

    def generate_n(self, num_el):
        return [self.next_element() for _ in range(num_el)]


class PoissonModeling:

    def __init__(self, a0, beta, M, l):
        self.bsv_modeling = MultiplicativeCongruentMethod(a0, beta, M)
        self.lambda_p = l
        self.a = []

    def next_element(self):
        m = 0
        A = -self.lambda_p - math.log(self.bsv_modeling.next_element())
        while A < 0:
            m = m + 1
            A -= math.log(self.bsv_modeling.next_element())
        return m

    def generate_n(self, num_el):
        return [self.next_element() for _ in range(num_el)]


class GeometricModeling:

    def __init__(self, a0, beta, M, p):
        self.bsv_modeling = MultiplicativeCongruentMethod(a0, beta, M)
        self.p = p
        self.q = 1 - self.p

    def next_element(self):
        a = self.bsv_modeling.next_element()
        return math.ceil(math.log(a) / math.log(self.q))

    def generate_n(self, num_el):
        return [self.next_element() for _ in range(num_el)]


class Estimates:

    def __init__(self):
        self.E = 0
        self.D = 0

    def UnbiasedE(self, seq):
        sum = 0
        for i in range(len(seq)):
            sum += seq[i]
        self.E = sum / len(seq)
        return self.E

    def UnbiasedD(self, seq):
        sum = 0
        for i in range(len(seq)):
            sum += (seq[i] - self.E)**2
        self.D = sum / (len(seq) - 1)
        return self.D

    def GeometricE(self, p):
        return 1/p

    def GeometricD(self, p):
        return (1-p)/p**2


def sum_chi(data, expected_frequences):
    s = 0
    for i in range(len(expected_frequences)):
        s += expected_frequences[i]
    diff = len(data) - s
    diff /= len(expected_frequences)
    for i in range(len(expected_frequences)):
        expected_frequences[i] += diff
    return expected_frequences


def ChiSquere(data, l, dist):
    observed_frequences = np.bincount(data)
    if dist == 'p':
        expected_frequences = [round(len(data) * stats.poisson.pmf(x, l)) for x in range(min(data), max(data) + 1)]
        expected_frequences = sum_chi(data, expected_frequences)
    elif dist == 'g':
        expected_frequences = [int(len(data) * stats.geom.pmf(x, l)) for x in range(min(data)-1, max(data) + 1)]
        expected_frequences = sum_chi(data, expected_frequences)
    res_test = stats.chisquare(observed_frequences, expected_frequences)
    return res_test[1]


def decision(res, e):
    if res > e:
        return "гипотеза принимается"
    else:
        return "гипотеза не принимается"


def type_1_error(generator, param, m, n, dist, e):
    error = 0
    for _ in range(m):
        if ChiSquere(generator.generate_n(n), param, dist) < e:
            error += 1
    print("Ошибка 1 рода: ", error / m)


n = 1000
lambda_poisson = 0.7
p = 0.2
a0 = 16807
beta = 16807
M = 2 ** 31
e = 0.05

est = Estimates()
print("Распределение Пуассона:")
poisson = PoissonModeling(a0, beta, M, lambda_poisson)
seq_p = poisson.generate_n(n)
print(seq_p)
print("Математическое ожидание: ", lambda_poisson, "; Оценка: ", est.UnbiasedE(seq_p))
print("Дисперсия: ", lambda_poisson, "; Оценка: ", est.UnbiasedD(seq_p))
print("Тест хи квадрат: ", decision(ChiSquere(seq_p, lambda_poisson, 'p'), e))
type_1_error(poisson, lambda_poisson, 1000, 1000, 'p', 0.05)

print()
print("Геометрическое распределение:")
geometric = GeometricModeling(a0, beta, M, p)
seq_g = geometric.generate_n(n)
print(seq_g)
print("Математическое ожидание: ", est.GeometricE(p), "; Оценка: ", est.UnbiasedE(seq_g))
print("Дисперсия: ", est.GeometricD(p), "; Оценка: ", est.UnbiasedD(seq_g))
print("Тест хи квадрат: ", decision(ChiSquere(seq_g, p, 'g'), e))
type_1_error(geometric, p, 2000, 1000, 'g', 0.05)

