# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:35:13 2026

@author: Ольга
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.utils.class_weight import compute_sample_weight

# В задании 2 работы 3 требуется определить число нейронов, требуемое
# для решения задачи классификации точек двумерного пространства.


def logistic(x, s=10):
    # логистическая функция активации
    return 1 / (1 + np.exp(-s*x))

def sign(x):
    # пороговая функция активации
    return (x > 0) * 1.

def relu(x):
    # ReLU-функция активации
    return x if x > 0 else 0.

class ManualMLP:
    # чтобы не лезть в детали реализации многослойного персептрона
    # в scikit-learn, сделаем свой маленький класс для него (без метода обучения,
    # т.е. только с ручным заданием коэффициентов)
    def __init__(self, coeffs, intercepts, act_func):
        # создать сеть по списку коэффициентов и функции активации
        self.coeffs = coeffs  # синаптические коэффициенты при входах
        self.intercepts = intercepts  # коэффициенты - постоянные (смещения нуля)
        self.act_func = act_func  # функция активации
        
    def predict(self, X):
        # получить выходы всех слоёв, а не только последнего
        curr_val = X
        results = []
        for i in range(len(self.coeffs)):
            # для каждого слоя провести линейное преобразование предыдущего
            curr_val = np.matmul(curr_val, self.coeffs[i]) + self.intercepts[i]
            # поместить его выход в результаты
            results.append(curr_val)
            # применить к этому же выходу нелинейное преобразование
            curr_val2 = self.act_func(curr_val)
            # поместить его тоже в результаты
            curr_val = curr_val2
            results.append(curr_val2)
        # выдать из функции и выход, и список результатов каждого слоя
        return curr_val, results

# создадим массивы коэффициентов, которые были в явном виде рассчитаны
# на лекции для задачи классификации
layer1_c = [[0, -1, -1],
            [1, 0, -1]]
layer1_i = [-2, 2, 7]

layer2_c = [[1, 0, -1, 0],
            [-1, 1, 0, 0],
            [1, 0, 0, -1]]
layer2_i = [-1.5, 0, 0.5, 0.5]

layer3_c = [[1, 0],
            [0, 1],
            [0, 1],
            [0, 1]]
layer3_i = [0, 0]


# создадим сеть с этими коэффициентами и пороговыми функциями активации всех слоёв
mlp = ManualMLP([np.array(layer1_c), np.array(layer2_c), np.array(layer3_c)],
                [np.array(layer1_i), np.array(layer2_i), np.array(layer3_i)],
                sign)

# обучать сеть не надо, потому что коэффициенты для неё мы рассчитали вручную;
# можно просто вывести её результаты для рассматриваемой области данных

def draw_all_layers_output(net, min_x, max_x, min_y, max_y, dx, dy):
    # функция вывода результатов работы всех нейронов сети net;
    # принимает она два числа -- первое принимает значения
    # от min_x до max_x с шагом dx, второе --- от min_y до max_y с шагом dy
    # Вычислим размер входных данных
    nx = round((max_x - min_x) / dx) + 1
    ny = round((max_y - min_y) / dy) + 1
    # составим из них решётку --- пару матриц vx,vy, в которых 
    # сочетаются все возможные пары x и y
    px = np.linspace(min_x, max_x, nx)
    py = np.linspace(min_x, max_x, ny)
    vx, vy = np.meshgrid(px, py)
    test_set = np.vstack([np.ravel(vx), np.ravel(vy)]).T
    # предскажем классы всех элементов решётки на нейросети
    preds,interm = net.predict(test_set)
    
    # выведем эти классы
    fig, axs = plt.subplots(nrows=4, ncols=6)
    # размеры слоёв сети
    szs = [interm[i].shape[1] for i in range(len(interm))]
    
    for i,sz in enumerate(szs):
        for j in range(sz):
            # для каждого используемого нейрона вывести выход в виде изображения
            reshaped_data = np.reshape(interm[i][:,j], (nx, ny))
            dt = np.flip(reshaped_data, 0)
            axs[j, i].imshow(dt, cmap=cm.coolwarm)
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])
            # неиспользуемыми оставить пустыми
            for j in range(sz, max(szs)):
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])
    plt.show()

# выведем выходы всех нейронов сети mlp в диапазонах x и y [-3,10]
# с шагом x и y 0.1. Слои выводятся слева направо; для каждого нейрона 
# выводится сначала сумма входов, а потом результат действия функции активации
# на эту сумму.

draw_all_layers_output(mlp, -3, 10, -3, 10, 0.1, 0.1)




# Теперь попытаемся обучить сеть с тем числом нейронов, которое было получено
# на прошлом этапе, и с близкими к нему, чтобы понять, насколько оправдаются
# предсказания относительно требуемого числа нейронов. Входов и выходов у сети 
# всегда будет по два, а число нейронов скрытых слоёв будем варьировать.


N = 50000
# x и y равномерно покрывают плоскость
xs = np.random.uniform(-3,10,size=(N,1))
ys = np.random.uniform(-3,10,size=(N,1))
# выделим классы по той же логике, что в лекции
line1 =  ys - 2 > 0
line2 = -1*xs + 2 > 0
line3 = -1*xs - ys +7 > 0

# выпуклые области получим сочетанием линейных разбиений, как в лекции
area = np.logical_and(np.logical_and(np.logical_not(line2), line1), line3)
# если точка относится к area, то относится к классу 1, иначе - к классу 2
class_ids = (1 * area).ravel()

# выборка будет состоять из точек points и номеров их классов class_ids
points = np.hstack((xs,ys))

# разделим единую начальную выборку на обучающую и тестовую. 
# Классификатор будет обучаться только на первой из них, а качество 
# работы будет оцениваться по второй. Это нужно для распознавания переобучения:
# если классификатор намного лучше сработает на обучающей выборке,
# чем на тестовой, он слишком приспособился к конкретным примерам
# и потерял способность обобщать данные.

points_train, points_test, classes_train, classes_test = train_test_split(points, class_ids,
                                                                          train_size=0.8,
                                                                          random_state=17,
                                                                          shuffle=True)

# нарисуем обучающую и тестовую выборки
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
idx = np.random.choice(len(points_train), size=1000, replace=False)
ax.scatter(points_train[idx, 0], points_train[idx, 1], classes_train[idx], c='b')
idx_test = np.random.choice(len(points_test), size=1000, replace=False)
ax.scatter(points_test[idx_test, 0], points_test[idx_test, 1], classes_test[idx_test], color='r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('номер класса')
plt.show()

# проверяемые размеры слоёв
layer1_sizes = [2,3,4,5]
layer2_sizes = [2,3,4,5,6]


# # массивы для результатов
precisions = np.zeros((len(layer1_sizes), len(layer2_sizes)))
recalls = np.zeros((len(layer1_sizes), len(layer2_sizes)))

# # число повторов проверок (т.е. число обучаемых нейросетей каждой конфигурации)
NE = 3

for e_num in range(NE):
    for i,layer1_size in enumerate(layer1_sizes):
        for j,layer2_size in enumerate(layer2_sizes):
            # сделаем сеть с числом нейронов скрытых слоёв,
            # равным (layer1_size, layer2_size). Так как 
            # функция активация будет отличаться от sign, результат вполне может
            # быть другим. Можно проверить его, заменив 'logistic' на 'relu' или 'tanh'
            mlp = MLPClassifier(
                hidden_layer_sizes=(layer1_size, layer2_size),
                activation='logistic',
                max_iter=5000,
                random_state=42)  
            sample_weights = compute_sample_weight(class_weight='balanced', y=classes_train)

            
            # обучаем классификатор на обучающей выборке
            mlp.fit(points_train,np.ravel(classes_train), sample_weight=sample_weights)
            # предсказываем классы для тестовой выборки
            # предсказываем классы для тестовой выборки
            classes_predicted = mlp.predict(points_test)
            # вырабатываем оценку качества
            rep = classification_report(classes_test, classes_predicted,output_dict=True)
            # выберем из результатов precision и recall для класса 1
            precisions[i,j] += rep['1']['precision']   
            recalls[i,j] += rep['1']['recall']         
            print(f'Network with l1={layer1_size} and l2={layer2_size} processed in experiment {e_num+1}')


# усредняем собранные данные
precisions /= NE
recalls /= NE
# выводим их в явном виде на всякий случай
print(precisions)
print(recalls)

# выведем тепловые карты данных. Определим границы выводимых значений,
# предполагая при этом, что количества нейронов взяты подряд
extents = [layer2_sizes[0]-0.5,layer2_sizes[-1]+0.5,
           layer1_sizes[0]-0.5,layer1_sizes[-1]+0.5]
plt.imshow(precisions,origin='lower',extent=extents)
plt.colorbar()
plt.title('Precision')
plt.xlabel('Первый слой')
plt.ylabel('Второй слой')
plt.show()
plt.imshow(recalls,origin='lower',extent=extents)
plt.colorbar()
plt.title('Recalls')
plt.xlabel('Первый слой')
plt.ylabel('Второй слой')
plt.show()


# Обучение сети (5,5), вывод весов и тест на избыточность

print("\n--- Обучение сети (5, 5) ---")

# 1. Создаем сеть с 5 нейронами в первом и 5 во втором слое
# Используем relu и early_stopping, чтобы она хорошо обучилась
mlp_5_5 = MLPClassifier(
    hidden_layer_sizes=(5, 5),
    activation='relu', 
    max_iter=2000,
    early_stopping=True,
    random_state=42
)

# 2. Обучаем её ОДИН РАЗ (используем веса, чтобы побороть дисбаланс классов)
# Переменная sample_weights должна быть уже создана в вашем коде ранее
mlp_5_5.fit(points_train, classes_train, sample_weight=sample_weights)



# Вывод коэффициентов

print("\n=== Коэффициенты обученной сети ===")
print("\n[Слой 1] (5 нейронов):")
print("Веса (coefs_):\n", np.round(mlp_5_5.coefs_[0], 4))
print("Смещения (intercepts_):", np.round(mlp_5_5.intercepts_[0], 4))

print("\n[Слой 2] (5 нейронов):")
print("Веса (coefs_):\n", np.round(mlp_5_5.coefs_[1], 4))
print("Смещения (intercepts_):", np.round(mlp_5_5.intercepts_[1], 4))

print("\n[Выходной слой]:")
print("Веса (coefs_):\n", np.round(mlp_5_5.coefs_[2], 4))
print("Смещения (intercepts_):", np.round(mlp_5_5.intercepts_[2], 4))



# Тест на избыточность (обнуление весов)

print("\n=== Тест на избыточность (обнуление весов) ===")

# Оценка ДО обнуления
preds_orig = mlp_5_5.predict(points_test)
rep_orig = classification_report(classes_test, preds_orig, output_dict=True)
print(f"До обнуления:       Precision={rep_orig['1']['precision']:.3f}, Recall={rep_orig['1']['recall']:.3f}")

# Обнуляем 25% (1 нейрон из 5) в предпоследнем слое.
# Индекс -2 означает предпоследний слой (второй скрытый слой).
# Символ : означает "все входящие связи", а 0 — это индекс конкретного нейрона.
mlp_5_5.coefs_[-2][:, 0] = 0 
preds_25 = mlp_5_5.predict(points_test)
rep_25 = classification_report(classes_test, preds_25, output_dict=True)
print(f"После обнуления 25%: Precision={rep_25['1']['precision']:.3f}, Recall={rep_25['1']['recall']:.3f}")

# Обнуляем еще 25% (итого 50% - 2 нейрона из 5)
mlp_5_5.coefs_[-2][:, 1] = 0 
preds_50 = mlp_5_5.predict(points_test)
rep_50 = classification_report(classes_test, preds_50, output_dict=True)
print(f"После обнуления 50%: Precision={rep_50['1']['precision']:.3f}, Recall={rep_50['1']['recall']:.3f}")



# Сравнение функций активации для сети (3, 4) с 3 нейронами на 1 слое и 4 нейронами на втором слое

print("\n--- Сравнение функций активации для сети (3, 4) ---")

# Список функций активации, которые мы хотим проверить
# (logistic - это сигмоида, tanh - гиперболический тангенс, relu - выпрямитель)
activations = ['relu', 'logistic', 'tanh']

results = {} # Словарь для хранения результатов

for act in activations:
    print(f"\nОбучение сети (3, 4) с функцией активации: '{act}'...")
    
    # Создаем и обучаем сеть
    mlp_act = MLPClassifier(
        hidden_layer_sizes=(3, 4),
        activation=act,
        max_iter=3000,           # Даем больше итераций для сходимости
        early_stopping=True,     # Защита от переобучения
        random_state=42          # Одинаковый seed для честного сравнения
    )
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=classes_train)
    mlp_act.fit(points_train, classes_train, sample_weight=sample_weights)
    
    # Оцениваем качество
    preds = mlp_act.predict(points_test)
    rep = classification_report(classes_test, preds, output_dict=True)
    
    # Сохраняем метрики
    precision = rep['1']['precision']
    recall = rep['1']['recall']
    f1_score = rep['1']['f1-score']
    
    results[act] = {'precision': precision, 'recall': recall, 'f1': f1_score}
    

# Вывод итоговой таблицы

print("\n" + "="*50)
print(f"{'Функция активации':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10}")
print("-"*50)
for act, metrics in results.items():
    print(f"{act:<20} | {metrics['precision']:<10.3f} | {metrics['recall']:<10.3f} | {metrics['f1']:<10.3f}")
print("="*50)