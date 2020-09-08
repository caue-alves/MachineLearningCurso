import pandas as pd
uri = 'https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'
data = pd.read_csv(uri)
data.head()

map = {
    "home" : "tela_inicial",
    "how_it_works" : "nosso_trabalho",
    'contact' : 'contato',
    "bought" : 'comprou'
}
data.rename(columns = map)

x = data[['home', 'how_it_works', 'contact']]
y = data['bought']
print(data)

data.shape

from sklearn.model_selection import train_test_split

SEED = 24

test = train_test_split(x, y, random_state = SEED, test_size=0.25, stratify=y)
print(len(test))
treino_x = test[0]
teste_x = test[1]
treino_y = test[2]
teste_y = test[3]

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(treino_x, treino_y)
result = model.predict(teste_x)
from sklearn.metrics import accuracy_score
print(f"{accuracy_score(teste_y, result) * 100}%")

print(treino_y.value_counts(), teste_y.value_counts(), sep="\n")