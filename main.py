from typing import Union

from fastapi import FastAPI

app = FastAPI()


import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier

# Чтоб FutureWarning и DeprecationWarning не вылезали
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# Dummy encoding
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разбивка данных на тестовые и тренировочные
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.5, random_state=2)

n_features = X.shape[1]
n_classes = Y.shape[1]

def create_model():
    model = Sequential(
        [Dense(8, input_dim=n_features, activation='relu'),
         Dense(8, input_dim=n_features, activation='relu'),
         Dense(n_classes, activation='softmax')],
        name="model"
        )

    model.compile(loss='mse',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model

model = create_model()

model.summary()

history_dict = {}

cb = TensorBoard()

print('Имя модели:', model.name)
history_callback = model.fit(X_train, Y_train,
                                batch_size=5,
                                epochs=50,
                                verbose=0,
                                validation_data=(X_test, Y_test),
                                callbacks=[cb])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Потери при тестировании:', score[0])
print('Точность тестирования:', score[1])


estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=5, verbose=0)
scores = cross_val_score(estimator, X_scaled, Y, cv=10)
print("Прогноз : {:0.2f} (+/- {:0.2f})".format(scores.mean(), scores.std()))



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}