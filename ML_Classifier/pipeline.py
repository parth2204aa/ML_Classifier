from preprocess import load_data, clean_data, split_data
from model import train_model, evaluate_model

def run_pipeline(path):
    df = clean_data(load_data(path))
    x_train,x_test, y_train, y_test = split_data(df)
    model =train_model((x_train, y_train))
    accuracy = evaluate_model(model, (x_test, y_test))
    return accuracy