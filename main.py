from tabnanny import verbose
from typing import Dict, Annotated
from enum import Enum
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, Form
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
from keras.layers import Dense, Input, Conv2D
from keras.callbacks import TensorBoard

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
        [Input(shape=(4,)),
         Dense(16, input_shape=(X_train[0].shape), activation='relu'),
         Dense(16, activation='relu'),
         Dense(n_classes, activation='softmax')],
        name="model")

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
                                epochs=150,
                                verbose=1,
                                validation_data=(X_test, Y_test),
                                callbacks=[cb])
score = model.evaluate(X_test, Y_test, verbose=1)
print('Потери при тестировании: {}'.format(score[0]))
print('Точность тестирования: {}'.format(score[1]))


pred = model.predict(X_test)

prognosis_array = np.around(pred) == Y_test
unique, counts = np.unique(prognosis_array, return_counts=True)
prognosis_percent = dict(zip(unique, counts))[True] / len(prognosis_array) * 10
print(str(prognosis_percent)+"%")


class ModelData(str, Enum):
    loss = str(score[0])
    acc = str(score[1])
    scores = str(prognosis_percent)+"%"
    model_name = model.name


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "loss": ModelData.loss.value, "acc": ModelData.acc.value, "scores": ModelData.scores.value, "model_name": ModelData.model_name.value})
    

@app.post("/model_query")
async def process_data(request: Request, seplength: Annotated[str, Form()], sepwidth: Annotated[str, Form()], petlength: Annotated[str, Form()], petwidth: Annotated[str, Form()]):
    results = model.predict([[float(seplength), float(sepwidth), float(petlength), float(petwidth)]], verbose=1)
    result = ""
    if np.around(results)[0][0] == 1: result = "setosa"
    elif np.around(results)[0][1] == 1: result = "versicolor"
    elif np.around(results)[0][2] == 1: result = "virginica"

    return templates.TemplateResponse("model_responce.html", {"request": request, "result": result, "loss": ModelData.loss.value, "acc": ModelData.acc.value, "scores": ModelData.scores.value, "model_name": ModelData.model_name.value})