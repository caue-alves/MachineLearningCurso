# 3 features:
# Pelo é longo?
# Perna é curta?
# Faz au-au?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

train_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
train_y = [1,1,1, 0, 0, 0]

from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(train_x, train_y)

sera1 = [1, 1, 1]
sera2 = [0, 1, 0]
sera3 = [1, 1, 0]
Xtest = [sera1, sera2, sera3]
Ytest = [0, 1, 1]
predicts = model.predict(Xtest)
print(predicts)

from sklearn.metrics import accuracy_score

print(f"{accuracy_score(predicts, Ytest) * 100}%")