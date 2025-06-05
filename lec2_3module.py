## Линейная регрссия
## Задача - на основе наблюдаемых точек построить прямую, которая отображает связь между двумя или более перемнными
# Регрессия пытается "подогнать" функцию к наблюдаемым данным, чтобы спрогнощировать новые данные
# Линейная регрессия подгоняет данные к прямой линии, пытаемся установить линейную связь между перемнными и предскахать новые данные

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr
import random


# features, target = make_regression(n_samples = 200, n_features = 1, n_informative = 1, n_targets = 1, noise = 20, random_state = 1)

# print(features.shape)

# model = LinearRegression().fit(features, target)

# plt.scatter(features, target)

# x = np.linspace(features.min(), features.max(), 100)

# plt.plot(x, model.coef_[0]*x + model.intercept_, color='orange')

# plt.show()


# Простая линейная регрессия
# Линейная -> линейная зависимость
# + :
# Прогнозирование на новых данных
# Анализ взамного влияния переменных друг на друга
# -:
# Точки обучаемых данных не будут лежать на прямой (шум) - область погрешности
# Не позволяет делать прогнозы вне диапазона имеющихся данных

# Данные, на основании которых разрабатывается модель - выборка из некоторой мовокупности
# Хотелось бы, чтобы это была репрезентативная выборка

data = np.array(
    [
        [1, 5],
        [2, 7],
        [3, 7],
        [4, 10],
        [5, 11],
        [6, 14],
        [7, 17],
        [8, 19],
        [9, 22],
        [10, 28]
    ]
)

x = data[:,0]
y = data[:,1]
n = len(x)

# Метод наименьших квадратов

# w_1 = n*sum(x[i]*y[i] for i in range(n)) - sum(num for num in x)*sum(num for num in y) 
# w_1 /= ( n*(sum(num**2 for num in x)) - sum(num for num in x)**2 )

# w_0 = sum(num for num in y)/n - w_1*sum(num for num in x)/n

# # plt.scatter(x,y)
# # plt.plot(x, w_1*x + w_0, linestyle = '--')

# # plt.show()

# # Метод обратных матриц

# x_1 = np.vstack([x, np.ones(len(x))]).T
# w = inv(x_1.transpose() @ x_1 ) @ (x_1.transpose() @ y)

# print(w)
# print(w_1, w_0)

# # Разложение матриц (QR)

# Q, R = qr(x_1)
# w = (inv(R) @ Q.T) @ y
# print(w)

# Градиентный спуск (метод оптимизации)

# def f(x):
#     return 0.1*(x-3)**2 - 4

# def df(x):
#     return 0.2*x - 0.6

# x = np.linspace(-10,10, 1000)

# L = 0.01
# iterations = 100_000

# t = np.random.randint(0,5)

# for i in range(iterations):
#     dx = df(t)
#     t -= L*dx

# print(t)

# ax = plt.gca()

# ax.xaxis.set_major_locator(plt.MultipleLocator(1))
# plt.grid()
# # plt.plot(x, f(x))
# # plt.plot(x, df(x))

# plt.show()


# Линейная регрессия

# x = data[:,0]
# y = data[:,1]
# n = len(x)
# w1, w0 = 0, 0
# L = 0.00001
# iterations = 100
# for i in range(iterations):
#     D_w0 = 2 * sum(y[i] - w0 - w1*x[i] for i in range(n))
#     D_w1 = 2 * sum(x[i] * (-y[i] - w0 - w1*x[i]) for i in range(n))
#     w0 -= L*D_w0
#     w1 -= L*D_w1

# print(w1, w0)

W1 = np.linspace(-10, 10, 100)
W0 = np.linspace(-10, 10, 100)

def E(w1,w0, x, y):
    n = len(x)
    return sum((y[i] - (w0 + w1 * x[i]))**2 for i in range(n))

EW = E(W1, W0, x, y)
print(EW)
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.plot_surface(W1, W0, EW)

plt.show()
