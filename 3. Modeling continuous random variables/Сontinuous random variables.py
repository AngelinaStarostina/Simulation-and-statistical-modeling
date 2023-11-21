import math
import numpy as np
from scipy import stats
from scipy.stats import norm


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


class NormModeling:

    def __init__(self, a0, beta, M, sigma, E):
        self.bsv_modeling = MultiplicativeCongruentMethod(a0, beta, M)
        self.N = 12
        self.sigma = math.sqrt(sigma)
        self.E = E

    def next_element_stand(self):
        a = self.bsv_modeling.generate_n(self.N)
        x = 0
        for i in range(self.N):
            x += a[i]
        return x - 6

    def next_element_m_sigma(self):
        a = self.bsv_modeling.generate_n(self.N)
        x = 0
        for i in range(self.N):
            x += a[i]
        x = x - 6
        return self.E + self.sigma * x

    def generate_n(self, num_el):
        return [self.next_element_m_sigma() for _ in range(num_el)]

    def cdf(self, x):
        return norm.cdf(x, loc=self.E, scale=self.sigma)


class LogNormModeling:

    def __init__(self, a0, beta, K, sigma, m):
        self.E = m
        self.sigma = sigma
        self.nsv_modeling = NormModeling(a0, beta, K, sigma, math.log(m))

    def next_element(self):
        return math.exp(self.nsv_modeling.next_element_stand()) * self.sigma + self.E

    def generate_n(self, num_el):
        return [self.next_element() for _ in range(num_el)]

    def cdf(self, x):
        return stats.lognorm.cdf(x, s=1, loc=self.E, scale=self.sigma)


class ExpModeling:

    def __init__(self, a0, beta, M, a):
        self.bsv_modeling = MultiplicativeCongruentMethod(a0, beta, M)
        self.lambda_exp = a

    def next_element(self):
        el = self.bsv_modeling.next_element()
        return -1 / self.lambda_exp * math.log(el)

    def generate_n(self, num_el):
        return [self.next_element() for _ in range(num_el)]

    def cdf(self, x):
        return stats.expon.cdf(x, loc=0, scale= 1/self.lambda_exp)


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

    def LognormE(self, m, s2):
        return m * math.exp(s2)

    def LognormD(self, m, s2):
        w = math.exp(s2)
        return m ** 2 * w * (w-1)

    def ExpE(self, a):
        return 1 / a

    def ExpD(self, a):
        return 1 / (a ** 2)


class ChiTest:
    def __init__(self, e, cdf, distribution):
        self.e = e
        self.cdf = cdf
        self.distribution = distribution

    def find_intervals(self, seq):
        n = 1 + int(math.log2(len(seq)))
        max_el = np.max(seq)
        min_el = np.min(seq)
        h = (max_el - min_el) / n
        a = np.zeros(n)
        for i in range(n - 1):
            a[i] = min_el + (i + 1) * h
        a[-1] = np.Inf
        return a

    def observed_frequencies(self, seq, intervals):
        n = len(intervals)
        freq = np.zeros(n)
        sort_seq = np.sort(seq)
        i = 0
        k = 0
        while i < n:
            while k < len(seq) and sort_seq[k] < intervals[i]:
                freq[i] += 1
                k += 1
            i += 1
        return freq

    def expected_frequencies(self, intervals, l):
        n = len(intervals)
        exp_freq = np.zeros(n)
        exp_freq[0] = l * self.cdf(intervals[0])
        for i in range(1, n):
            exp_freq[i] = l * (self.cdf(intervals[i]) - self.cdf(intervals[i - 1]))
        return exp_freq

    def chisquare_test(self, seq, print_res):
        intervals = self.find_intervals(seq)
        obs_freq = self.observed_frequencies(seq, intervals)
        exp_freq = self.expected_frequencies(intervals, len(seq))
        k = len(intervals)
        stat = stats.chisquare(obs_freq, exp_freq)
        critical_value = stats.chi2.ppf(1 - self.e, k + 2)
        res = 1 if stat.statistic < critical_value else 0
        if print_res:
            print(decision(res, self.e))
        return res


def decision(res, e):
    if res > e:
        return "гипотеза принимается"
    else:
        return "гипотеза отклоняется"


def decision_error(res, e):
    if res > e:
        return 1
    else:
        return 0


def type_1_error(generator, n, e):
    xi_res = 0
    k_res = 0
    for _ in range(n):
        tmp = generator.generate_n(n)
        xi_res += ChiTest(e, generator.cdf, 'norm').chisquare_test(tmp, False)
        k_res += decision_error(stats.kstest(tmp, generator.cdf)[1], e)

    xi_error = 1 - (xi_res / n)
    k_error = 1 - (k_res / n)
    print("Вероятность ошибки первого рода критерий Колмогорова: ", k_error)
    print("Вероятность ошибки первого рода критерий хи-квадрат: ", xi_error)


n = 1000
e = 0.05
a0 = 16209
beta = 16209
M = 2 ** 31
m = 1
s2 = 9
lambda_exp = 2


est = Estimates()
print("Нормальное распределение:")
norm_ = NormModeling(a0, beta, M, s2, m)
seq_p = norm_.generate_n(n)
print(seq_p)
print("Математическое ожидание: ", m, "; Оценка: ", est.UnbiasedE(seq_p))
print("Дисперсия: ", s2, "; Оценка: ", est.UnbiasedD(seq_p))
print("Критерий согласия Колмогорова: ", end='')
print(decision(stats.kstest(seq_p, norm_.cdf)[1], e))
print("Критерий хи-квадрат: ", end='')
ChiSquare = ChiTest(e, norm_.cdf, 'norm')
ChiSquare.chisquare_test(seq_p, True)
type_1_error(norm_, 100, e)


print()
print("Логнормальное распределение:")
lognorm_ = LogNormModeling(a0, beta, M, s2, m)
seq_p = lognorm_.generate_n(n)
print(seq_p)
print("Математическое ожидание: ", est.LognormE(m, math.sqrt(s2)), "; Оценка: ", est.UnbiasedE(seq_p))
print("Дисперсия: ", est.LognormD(m, math.sqrt(s2)), "; Оценка: ", est.UnbiasedD(seq_p))
print("Критерий согласия Колмогорова: ", end='')
res = stats.kstest(seq_p, lognorm_.cdf)
print(decision(res[1], e))
print("Критерий хи-квадрат: ", end='')
ChiSquare = ChiTest(e, lognorm_.cdf, 'lognorm')
ChiSquare.chisquare_test(seq_p, True)
type_1_error(lognorm_, 100, e)


print()
print("Экспоненциальное распределение:")
exp_ = ExpModeling(a0, beta, M, lambda_exp)
seq_p = exp_.generate_n(n)
print(seq_p)
print("Математическое ожидание: ", est.ExpE(lambda_exp), "; Оценка: ", est.UnbiasedE(seq_p))
print("Дисперсия: ", est.ExpD(lambda_exp), "; Оценка: ", est.UnbiasedD(seq_p))
print("Критерий согласия Колмогорова: ", end='')
res = stats.kstest(seq_p, exp_.cdf)
print(decision(res[1], e))
print("Критерий хи-квадрат: ", end='')
ChiSquare = ChiTest(e, exp_.cdf, 'exp')
ChiSquare.chisquare_test(seq_p, True)
type_1_error(exp_, 100, e)
