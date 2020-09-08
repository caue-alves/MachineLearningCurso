uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'
import pandas as pd
data = pd.read_csv(uri)

map = {
    'unfinished':'nao_finalizado',
    'expected_hours':'horas_esperadas',
    'price':'preco'
}
data = data.rename(columns = map)

trocar = {
    0:1,
    1:0
}
data['finalizado'] = data.nao_finalizado.map(trocar)

!pip install seaborn==0.9.0
import seaborn as sns
sns.scatterplot(x="horas_esperadas", y="preco", data=data)

sns.scatterplot(x="horas_esperadas", y="preco", hue=teste_y,  data=data)

x = data[['horas_esperadas', 'preco']]
y = data['finalizado']

from sklearn.model_selection import train_test_split 

SEED = 5
np.random.seed(SEED)

test = train_test_split(x, y, test_size=0.25, stratify=y)
print(len(test))
treino_x = test[0]
teste_x = test[1]
treino_y = test[2]
teste_y = test[3]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(treino_x)
ss.fit(teste_x)
treino_x = ss.transform(treino_x)
teste_x = ss.transform(teste_x)

from sklearn.svm import SVC
model = SVC()
model.fit(treino_x, treino_y)
result = model.predict(teste_x)
from sklearn.metrics import accuracy_score
print(f"{accuracy_score(teste_y, result) * 100}%")

print(treino_y.value_counts(), teste_y.value_counts(), sep="\n")

x_min = data.horas_esperadas.min()
x_max = data.horas_esperadas.max()
y_min = data.preco.min()
y_max = data.preco.max()
print(f"X Min: {x_min}, X Max: {x_max}")
print(f"Y Min:{y_min}, Y Max: {y_max}")

import numpy as np
baseline = np.ones(540)

pixels = 100
x_index = np.arange(x_min, x_max, (x_max - x_min) / pixels)
y_index = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(x_index, y_index)
pontos = np.c_[xx.ravel(), yy.ravel()]
pontos

Z = model.predict(pontos)
Z = Z.reshape(xx.shape)
Z

import matplotlib.pyplot as plt
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(data.horas_esperadas, data.preco, s=1)