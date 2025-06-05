# Наивная байессовская классификация
# Набор моделей, которые предлагают быстрые и простые алгоритмы классификации
# Первое приближенное решение задачи классификации

# Гауссовский наивный байессовский классификатор
# Допущение состоит в том, что данные всех категорий взяты из простого нормального распределения
# Причем распределения независимы

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

iris = sns.load_dataset('iris')
# print(iris.head())

# sns.pairplot(iris, hue='species')


data = iris[['sepal_length', 'petal_length', 'species']]
# print(data)

# setosa / versicolor

data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]
X = data_df[['sepal_length', 'petal_length']] 
Y = data_df['species'] # Метки
# print(data_df.shape)
# sns.pairplot(data_df, hue='species')

model = GaussianNB()
model.fit(X, Y)

print(model.theta_[0]) # Мат ожидание
print(model.var_[0]) # Дисперсия

print(model.theta_[1])
print(model.var_[1])

theta_0, theta_1 = model.theta_[0], model.theta_[1]
var_0, var_1 = model.var_[0], model.var_[1]



data_df_setosa = data_df[data_df['species']=='setosa']
data_df_versicolor = data_df[data_df['species']=='versicolor']

# plt.scatter(data_df_setosa['sepal_length'], data_df_setosa['petal_length'])
# plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 400)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 400)

# Подготовка данных

# Создание пар
X1_p, X2_p = np.meshgrid(x1_p, x2_p)

# Подготовка данных комбинаций признаков
X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
                   columns=['sepal_length', 'petal_length'])

Z1 = 1 / (1*np.pi*(var_0[0]*var_0[1])**0.5) * np.exp(-0.5*(X1_p - theta_0[0])**2/var_0[0] - 0.5*(X2_p - theta_0[1])**2/var_0[1])
Z2 = 1 / (1*np.pi*(var_1[0]*var_1[1])**0.5) * np.exp(-0.5*(X1_p - theta_1[0])**2/var_1[0] - 0.5*(X2_p - theta_1[1])**2/var_1[1])
plt.contour(X1_p, X2_p, Z1)
plt.contour(X1_p, X2_p, Z2)

# print(X_p.head())
# Предсказание
Y_p = model.predict(X_p)
X_p['species'] = Y_p

X_p_setosa = X_p[X_p['species']=='setosa']
X_p_versicolor = X_p[X_p['species']=='versicolor']
cls = [(0.2, 0.4, 0.5), (0.6, 0.4, 0.2)]
plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], linewidths=5, alpha=0.2, color=cls[0])
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], linewidths=5, alpha=0.2, color=cls[1])
plt.scatter(data_df_setosa['sepal_length'], data_df_setosa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1_p, X2_p, Z1, levels=200)
ax.contour3D(X1_p, X2_p, Z2, levels=100)
# setosa / verginica


plt.show()


