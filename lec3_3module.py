import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd


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

# Градиентный спуск - пакетный градиентный спуск. Для работы используются
# все обучающие данные. На практике используется стохастический градиентный спуск
#  На каждой итерации обучаемся только на одной выборке из данных

# - сокращение числа ввычислений
#  - восим смещение -> боремся с переобучением

# Мини-пакетный градиентный спуск

# x = data[:,0]
# y = data[:,1]
# n = len(x)
# w1, w0 = 0, 0
# L = 0.001

# sample_size = 3  # Размер выборки
# iterations = 100_000

# for i in range(iterations):
#     idx = np.random.choice(n, sample_size, replace= False)
#     D_w0 = 2*(-y[idx] + w0 + w1 * x[idx]) 
#     D_w1 = 2 * (x[idx] * (-y[idx] + w0 + w1*x[idx]))
#     w0 -= L*D_w0
#     w1 -= L*D_w1

# print(sum(w1)/sample_size, sum(w0)/sample_size)

# Как оценить, насколько сильно промахиваются прогнозы при использовании
# линейной регрессии

# Для оценки степени взаимосвязи между двумя переменными мы использовали
# линейный коэффициент корреляции

# data_df = pd.DataFrame(data)
# print(data_df.corr(method='pearson'))

# data_df[1] = data_df[1].values[::-1]
# print(data_df.corr(method='pearson'))

# К-т корреляции помогает понять, есть ли связь между двумя переменными

# Обучающие и тестовые выборки
# Основной метод борьбы с переобучением
# Заключается в том, что набор данных делится на обучающую и тестовую выборки

# Во всех видаз машинного обучения с учителем это встречается

# Обычная пропорция: 2/3 - обучение, 1/3 - на тест (4/5 и 1/5) (9/10 и 1/10)

# data_df = pd.DataFrame(data)

# # 3-кратная перекрестная валидация
# kfold = KFold(n_splits= 3, random_state=1, shuffle=True)

# X = data_df.values[:,:-1]
# Y = data_df.values[:,1]

# # X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3)

# # print(X_train)
# # print(Y_train)

# # print(X_test)
# # print(Y_test)

# model = LinearRegression()
# # model.fit(X_train, Y_train)

# results = cross_val_score(model, X, Y, cv=kfold) # Перекреестная валидция
# print(results) 
# print(results.mean(), results.std())
# К-т детерминации r^2
# print(r**0.5)

# Метрики показывают, насколько единообразно ведет себя модель
# на разных выборках

# Возмоно использование поэлементной перекрестной валидации 
# (в случае, если мало данных)
# Можно делать случайную валидацию

# Иногда часть выборки выделяют как валидационную выборку 
# (сравнение различных моделей или конфигураций)


# Многомерная линейная регрессия

data_df = pd.read_csv('multiple_independent_variable_linear.csv')

# print(data_df.head())

X = data_df.values[:,:-1]
Y = data_df.values[:,-1]

model = LinearRegression().fit(X, Y)

print(model.coef_, model.intercept_)

x1, x2= X[:, 0], X[:, 1]
y = Y

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, y)

x1_1 = np.linspace(min(x1), max(x1), 100)
x2_1 = np.linspace(min(x2), max(x2), 100)
X1_, X2_ = np.meshgrid(x1_1, x2_1)
Y_ = model.intercept_ + model.coef_[0]*X1_ + model.coef_[1]*X2_

ax.plot_surface(X1_, X2_, Y_, cmap='Greys', alpha=0.5)

plt.show()

