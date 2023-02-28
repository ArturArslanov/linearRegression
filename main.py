import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class DummyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def add_weight(self, X):
        n, k = X.shape
        x1 = X
        if self.fit_intercept:
            x1 = np.hstack((np.ones((n, 1)), X))
        return x1

    def fit(self, X, y):
        # фукнкция обучения - вычисляет параметры модели (веса) по данной выборке
        X_training = self.add_weight(X)
        self.w = np.dot(np.dot(np.linalg.inv(np.dot(X_training.T, X_training)), X_training.T), y)

    def predict(self, X):
        # функция предсказания - предсказывает ответы модели по данной выборке
        X_training = self.add_weight(X)
        y_pred = X_training @ self.w  # < напишите код здесь >

        return np.round(y_pred)


class DummyLinearRegressionWithRegularization(DummyLinearRegression):
    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept)

    def fit(self, X, y):
        # функция обучения - вычисляет параметры модели (веса) по данной выборке, c учетом регуляризации
        X_training = self.add_weight(X)
        n, k = X.shape
        E = np.eye(k + 1) if self.fit_intercept else np.eye(k)
        lambda_ = 1.1
        self.w = np.linalg.inv(X_training.T @ X_training + lambda_ * lambda_ * E) @ X_training.T @ y  # < добавьте слагаемое из формулы выше >
        return self


def minmax_scale(s1):
    array = np.divide((s1 - s1.min(axis=0)), s1.max(axis=0) - s1.min(axis=0) + 0.00000001)
    array[:][array >= 0.99999999] = 1
    return pd.DataFrame(array)


def onehot_encoding(s1: np.array):
    set1 = np.unique(s1)
    set1 = np.sort(set1)
    array = np.zeros(len(set1) * len(s1))
    array = array.reshape(len(s1), len(set1))
    for i in range(len(s1)):
        for j in range(len(set1)):
            if set1[j] == s1[i]:
                array[i][j] = 1
    array = array.astype("int8")
    return pd.DataFrame(array)


df = pd.read_csv("travel insurance.csv")
df = df.drop("Destination", axis=1)
df = df.drop("Product Name", axis=1)
df = df.drop("Gender", axis=1)
df = df.loc[df["Duration"] > 0]
df = df.loc[df["Duration"] < 4000]

df = df.dropna()
cat_features_mask = (df.dtypes == "object").values
headers = df.keys()
for i in range(len(cat_features_mask)):
    df[headers[i]].replace('', np.nan, inplace=True)
    df = df.dropna(subset=headers[i])
    if cat_features_mask[i]:
        if headers[i] == 'Claim':
            continue
        onehot_vector = onehot_encoding(df[headers[i]].values)
        df = pd.concat([df, onehot_vector], axis=1)
        df = df.drop(headers[i], axis=1)
    else:
        df[headers[i]] = minmax_scale(df[headers[i]].values)

df = df.dropna()
y = np.where(df['Claim'] == 'Yes', 1, 0)
df = df.drop("Claim", axis=1)
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)
model = DummyLinearRegression()
model.fit(X_train, y_train)
result = model.predict(X_test)

model2 = DummyLinearRegressionWithRegularization()
model2.fit(X, y)
l2_result = model2.predict(X_test)

print(f"MSE без регуляризации: {mean_squared_error(y_test, result)}")
print(f"MSE с регуляризацией: {mean_squared_error(y_test, l2_result)}")

# во время обработки данных я заметил что большая часть у гендера
# не определенна в следствии чего, решил удалить чтобы не влияло не резултьат
# так же из за слишком большого количества стран я решил удалить параметр Destination ведь там он большой разброс и малое количество информации
# и product name так же имел сликшом много возможных значений
# у параметра duration(длительность поездки) были некоторые отрицательные значение что невозможно
# и некоторые значение большие 4000 которых было немного и они могли повлиять на распределение веса, по этому я их удалил
