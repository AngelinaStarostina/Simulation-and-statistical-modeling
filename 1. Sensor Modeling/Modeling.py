from scipy.stats import stats
from numpy import random


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


class McLarenMarsagliaMethod:

    def __init__(self, k, gen1, gen2):
        self.generator_1 = gen1
        self.generator_2 = gen2
        self.K = k
        self.V = [self.generator_1.next_element() for _ in range(self.K)]

    def next_element(self):
        s = int(self.generator_2.next_element() * self.K)
        a = self.V[s]
        self.V[s] = self.generator_1.next_element()
        return a

    def generate_n(self, num_el):
        return [self.next_element() for _ in range(num_el)]


def check_accuracy(seq, e):
    print("Критерий согласия Колмогорова: ", end='')
    res = stats.kstest(seq, 'uniform')
    decision(res[1], e)
    print("Критерий Хи-квадрат: ", end='')
    res = stats.chisquare(seq, random.uniform(0, 1))
    decision(res[1], e)


def decision(res, e):
    if res > e:
        print("гипотеза принимается")
    else:
        print("гипотеза не принимается")


a0 = 16807
beta = 16807
M = 2 ** 31
K = 64
n = 1000
eps = 0.05
print("Мультипликативный конгруэнтный метод")
generator_MCM = MultiplicativeCongruentMethod(a0, beta, M)
sequence_MCM = generator_MCM.generate_n(n)
print(sequence_MCM)
check_accuracy(sequence_MCM, eps)
print()

print("Метод Макларена-Марсальи")
generator_MM = McLarenMarsagliaMethod(K, generator_MCM, MultiplicativeCongruentMethod(79507, 79507, M))
sequence_MM = generator_MM.generate_n(n)
print(sequence_MM)
check_accuracy(sequence_MM, eps)

