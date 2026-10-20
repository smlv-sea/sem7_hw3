# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:06:41 2026

@author: Ольга
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight


# 1. Генерация выборок
N = 2000 # Количество точек для каждого класса
# N = 5000

# Класс 1: Точки внутри шара радиуса 0.5 с центром (0, -0.5, 0) ---
# Генерируем точки в кубе, охватывающем шар, и оставляем только те, что внутри
x1 = np.random.uniform(-0.5, 0.5, N)
y1 = np.random.uniform(-1.0, 0.0, N)
z1 = np.random.uniform(-0.5, 0.5, N)
mask_ball = (x1**2 + (y1 + 0.5)**2 + z1**2) <= 0.5**2
class1 = np.column_stack((x1[mask_ball], y1[mask_ball], z1[mask_ball]))

# Класс 2: Точки между плоскостями 0.2x + 0.2y + z = 1 и -1 ---
# То есть |0.2x + 0.2y + z| < 1
# Сгенерируем точки в кубе [-2, 2] и отфильтруем
x2 = np.random.uniform(-2, 2, N)
y2 = np.random.uniform(-2, 2, N)
z2 = np.random.uniform(-2, 2, N)
val_plane = 0.2*x2 + 0.2*y2 + z2
mask_plane = np.abs(val_plane) < 1
class2 = np.column_stack((x2[mask_plane], y2[mask_plane], z2[mask_plane]))

# Класс 3: Точки вне параллелепипеда (0.5, 1, 1.5) с центром (-1, 2, 1) ---
# Границы: x in [-1.25, -0.75], y in [1.5, 2.5], z in [0.25, 1.75]
x3 = np.random.uniform(-3, 1, N) # Берем куб побольше
y3 = np.random.uniform(0, 4, N)
z3 = np.random.uniform(-1.5, 3.5, N)
inside_box = ((x3 >= -1.25) & (x3 <= -0.75) & 
              (y3 >= 1.5) & (y3 <= 2.5) & 
              (z3 >= 0.25) & (z3 <= 1.75))
class3 = np.column_stack((x3[~inside_box], y3[~inside_box], z3[~inside_box]))

# Объединяем классы:
# Класс 0 = шар (class1)
# Класс 1 = плоскости (class2) + вне параллелепипеда (class3)
X_class0 = class1
y_class0 = np.zeros(len(X_class0))

X_class1 = np.vstack((class2, class3))
y_class1 = np.ones(len(X_class1))

# Формируем общую выборку
X = np.vstack((X_class0, X_class1))
y = np.concatenate((y_class0, y_class1))

# Перемешаем данные 
shuffle_idx = np.random.permutation(len(y))
X = X[shuffle_idx]
y = y[shuffle_idx]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# Визуализация формы классов
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Рисуем только часть точек
idx0 = np.random.choice(len(X_class0), 500, replace=False)
idx1 = np.random.choice(len(X_class1), 500, replace=False)
ax.scatter(X_class0[idx0, 0], X_class0[idx0, 1], X_class0[idx0, 2], c='b', label='Класс 0 (Шар)')
ax.scatter(X_class1[idx1, 0], X_class1[idx1, 1], X_class1[idx1, 2], c='r', label='Класс 1 (Плоскости + Вне)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title("Визуализация классов")
plt.show()


# Обучение сетей и сбор метрик

# Вычисляем веса для балансировки 
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Массивы для результатов
neurons = list(range(5, 55, 5)) # 5, 10, 15, ..., 50
times = []
# Списки для метрик КЛАССА 1 (плоскости + вне)
prec_train_1, prec_test_1 = [], []
rec_train_1, rec_test_1 = [], []

# Списки для метрик КЛАССА 0 (шар)
prec_train_0, prec_test_0 = [], []
rec_train_0, rec_test_0 = [], []

print("\nНачинаем обучение...")

for n in neurons:
    # Создаем сеть. Число нейронов во втором слое оставим фиксированным (например, тоже n)
    # или можно сделать (n, 10). Для чистоты эксперимента сделаем (n, n//2 + 2)
    mlp = MLPClassifier(
        hidden_layer_sizes=(n, 5), 
        activation='relu',
        max_iter=1000,
        #max_iter=2000,
        random_state=42,
        # early_stopping=True # Включаем раннюю остановку для борьбы с переобучением
        early_stopping=False
    )
    
    # Засекаем время
    start_time = time.time()
    mlp.fit(X_train, y_train, sample_weight=sample_weights)
    end_time = time.time()
    
    times.append(end_time - start_time)
    
    # Предсказания
    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)
    
    # Метрики для КЛАССА 1
    p_train_1 = precision_score(y_train, y_pred_train, pos_label=1, zero_division=0)
    r_train_1 = recall_score(y_train, y_pred_train, pos_label=1, zero_division=0)
    p_test_1  = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
    r_test_1  = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)

    prec_train_1.append(p_train_1)
    rec_train_1.append(r_train_1)
    prec_test_1.append(p_test_1)
    rec_test_1.append(r_test_1)
    
    # Метрики для КЛАССА 0 
    p_train_0 = precision_score(y_train, y_pred_train, pos_label=0, zero_division=0)
    r_train_0 = recall_score(y_train, y_pred_train, pos_label=0, zero_division=0)
    p_test_0  = precision_score(y_test, y_pred_test, pos_label=0, zero_division=0)
    r_test_0  = recall_score(y_test, y_pred_test, pos_label=0, zero_division=0)

    prec_train_0.append(p_train_0)
    rec_train_0.append(r_train_0)
    prec_test_0.append(p_test_0)
    rec_test_0.append(r_test_0)

    print(f"\nНейронов: {n:2d} | Время: {times[-1]:.3f} сек")
    print(f"  Класс 1 (Train): Precision={p_train_1:.3f}, Recall={r_train_1:.3f}")
    print(f"  Класс 1 (Test): Precision={p_test_1:.3f}, Recall={r_test_1:.3f}")
    print(f"  Класс 0 (Train): Precision={p_train_0:.3f}, Recall={r_train_0:.3f}")
    print(f"  Класс 0 (Test): Precision={p_test_0:.3f}, Recall={r_test_0:.3f}")
    


# Построение графиков

# График времени обучения
plt.figure(figsize=(8, 5))
plt.plot(neurons, times, marker='o', linestyle='-', color='b')
plt.title('Зависимость времени обучения от числа нейронов')
plt.xlabel('Число нейронов в первом скрытом слое')
plt.ylabel('Время обучения (сек)')
plt.grid(True)
plt.show()

# График Precision для класса 1
plt.figure(figsize=(8, 5))
plt.plot(neurons, prec_train_1, marker='o', label='Обучающая выборка (Train)')
plt.plot(neurons, prec_test_1, marker='s', label='Тестовая выборка (Test)')
plt.title('Зависимость Precision (для класса 1) от числа нейронов')
plt.xlabel('Число нейронов в первом скрытом слое')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()

# График Recall для класса 1
plt.figure(figsize=(8, 5))
plt.plot(neurons, rec_train_1, marker='o', label='Обучающая выборка (Train)')
plt.plot(neurons, rec_test_1, marker='s', label='Тестовая выборка (Test)')
plt.title('Зависимость Recall (для класса 1) от числа нейронов')
plt.xlabel('Число нейронов в первом скрытом слое')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
plt.show()


# График Precision для класса 0
plt.figure(figsize=(8, 5))
plt.plot(neurons, prec_train_0, marker='o', label='Обучающая выборка (Train)')
plt.plot(neurons, prec_test_0, marker='s', label='Тестовая выборка (Test)')
plt.title('Зависимость Precision (для класса 0) от числа нейронов')
plt.xlabel('Число нейронов в первом скрытом слое')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()

# График Recall для класса 0
plt.figure(figsize=(8, 5))
plt.plot(neurons, rec_train_0, marker='o', label='Обучающая выборка (Train)')
plt.plot(neurons, rec_test_0, marker='s', label='Тестовая выборка (Test)')
plt.title('Зависимость Recall (для класса 0) от числа нейронов')
plt.xlabel('Число нейронов в первом скрытом слое')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
plt.show()