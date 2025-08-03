from preprocess import load_data, clean_data, split_data
from model import train_model
from sklearn.pipeline import Pipeline

def test_model_training():
    df = clean_data(load_data('data/iris.csv'))
    x_train, x_test, y_train, y_test = split_data(df)
    model = train_model((x_train, y_train))
    assert isinstance (model, Pipeline)
    assert hasattr(model, 'predict')
