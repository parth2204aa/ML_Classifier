from tkinter import _test
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def train_model(data):
    x_train, y_train = data
    model = make_pipeline(StandardScaler(),
    LogisticRegression(max_iter=200))
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, data):
    x_train, y_train = data
    y_pred = model.predict(_test)
    return accuracy_score(_test, y_pred),sklearn